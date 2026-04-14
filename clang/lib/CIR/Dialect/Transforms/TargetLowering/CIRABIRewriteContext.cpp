//===- CIRABIRewriteContext.cpp - CIR-specific ABI rewriting --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRABIRewriteContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include <limits>

using namespace cir;
using namespace mlir;
using namespace mlir::abi;

/// Emit a value coercion between two types.  For scalar-to-scalar
/// (e.g. integer sign extension), a direct cir.cast is sufficient.
/// When one of the types is a record (struct), LLVM IR's bitcast
/// cannot reinterpret between aggregate and scalar types, so we go
/// through memory: alloca srcTy -> store src -> bitcast ptr -> load
/// dstTy.
static Value emitCoercion(OpBuilder &rewriter, Location loc, Type dstTy,
                          Value src) {
  Type srcTy = src.getType();
  if (srcTy == dstTy)
    return src;

  bool needsMemory =
      mlir::isa<cir::RecordType, cir::ComplexType>(srcTy) ||
      mlir::isa<cir::RecordType, cir::ComplexType>(dstTy) ||
      (mlir::isa<cir::VectorType>(srcTy) != mlir::isa<cir::VectorType>(dstTy));

  if (!needsMemory)
    return cir::CastOp::create(rewriter, loc, dstTy, cir::CastKind::bitcast,
                               src);

  auto srcPtrTy = cir::PointerType::get(srcTy);
  auto dstPtrTy = cir::PointerType::get(dstTy);

  auto alloca =
      cir::AllocaOp::create(rewriter, loc, srcPtrTy, srcTy,
                            /*name=*/rewriter.getStringAttr("coerce"),
                            /*alignment=*/rewriter.getI64IntegerAttr(8));

  cir::StoreOp::create(rewriter, loc, src, alloca,
                       /*isVolatile=*/mlir::UnitAttr(),
                       /*alignment=*/mlir::IntegerAttr(),
                       /*sync_scope=*/cir::SyncScopeKindAttr(),
                       /*mem_order=*/cir::MemOrderAttr());

  auto ptrCast = cir::CastOp::create(rewriter, loc, dstPtrTy,
                                     cir::CastKind::bitcast, alloca);

  return cir::LoadOp::create(rewriter, loc, dstTy, ptrCast,
                             /*isDeref=*/mlir::UnitAttr(),
                             /*isVolatile=*/mlir::UnitAttr(),
                             /*alignment=*/mlir::IntegerAttr(),
                             /*sync_scope=*/cir::SyncScopeKindAttr(),
                             /*mem_order=*/cir::MemOrderAttr());
}

/// Insert coercion before each cir.return to coerce the return value
/// from the original type to the ABI type.
static void insertReturnCoercion(FunctionOpInterface funcOp, Type origRetTy,
                                 Type coercedRetTy, OpBuilder &rewriter) {
  SmallVector<cir::ReturnOp> returnOps;
  funcOp->walk([&](cir::ReturnOp retOp) { returnOps.push_back(retOp); });

  for (cir::ReturnOp retOp : returnOps) {
    if (retOp.getInput().empty())
      continue;

    Value origVal = retOp.getInput()[0];
    if (origVal.getType() == coercedRetTy)
      continue;

    rewriter.setInsertionPoint(retOp);
    Value coerced =
        emitCoercion(rewriter, retOp.getLoc(), coercedRetTy, origVal);
    retOp->setOperand(0, coerced);
  }
}

/// Rewrite each cir.return to store the return value through the sret
/// pointer (first block argument) and return void.
static void insertSRetStores(FunctionOpInterface funcOp, Type origRetTy,
                             OpBuilder &rewriter) {
  Value sretPtr = funcOp.getArguments()[0];

  SmallVector<cir::ReturnOp> returnOps;
  funcOp->walk([&](cir::ReturnOp retOp) { returnOps.push_back(retOp); });

  for (cir::ReturnOp retOp : returnOps) {
    if (retOp.getInput().empty())
      continue;

    Value retVal = retOp.getInput()[0];
    rewriter.setInsertionPoint(retOp);
    cir::StoreOp::create(rewriter, retOp.getLoc(), retVal, sretPtr,
                         /*isVolatile=*/mlir::UnitAttr(),
                         /*alignment=*/mlir::IntegerAttr(),
                         /*sync_scope=*/cir::SyncScopeKindAttr(),
                         /*mem_order=*/cir::MemOrderAttr());
    cir::ReturnOp::create(rewriter, retOp.getLoc());
    retOp->erase();
  }
}

/// For each argument that requires ABI coercion (Extend or Direct
/// with a coerced type), insert a cast at the function entry and
/// replace all uses of the block argument with the cast result.
static void insertArgAdaptation(FunctionOpInterface funcOp,
                                const FunctionClassification &fc,
                                unsigned sretOffset, OpBuilder &rewriter) {
  Region &body = funcOp->getRegion(0);
  if (body.empty())
    return;

  Block &entryBlock = body.front();
  Operation *lastInserted = nullptr;

  for (auto [idx, argClass] : llvm::enumerate(fc.argInfos)) {
    if (!argClass.coercedType)
      continue;

    if (argClass.kind != ArgKind::Extend && argClass.kind != ArgKind::Direct)
      continue;

    // Skip flattened args -- handled separately.
    if (argClass.canFlatten && argClass.coercedType)
      if (auto cr = dyn_cast<cir::RecordType>(argClass.coercedType))
        if (cr.getMembers().size() > 1)
          continue;

    unsigned blockIdx = idx + sretOffset;
    BlockArgument blockArg = entryBlock.getArgument(blockIdx);
    Type oldArgTy = blockArg.getType();
    Type newArgTy = argClass.coercedType;

    if (oldArgTy == newArgTy)
      continue;

    blockArg.setType(newArgTy);

    if (lastInserted)
      rewriter.setInsertionPointAfter(lastInserted);
    else
      rewriter.setInsertionPointToStart(&entryBlock);

    Value adapted;
    SmallPtrSet<Operation *, 4> coercionOps;

    if (argClass.kind == ArgKind::Extend) {
      auto cast = cir::CastOp::create(rewriter, funcOp.getLoc(), oldArgTy,
                                      cir::CastKind::integral, blockArg);
      adapted = cast;
      coercionOps.insert(cast.getOperation());
    } else {
      auto srcPtrTy = cir::PointerType::get(newArgTy);
      auto dstPtrTy = cir::PointerType::get(oldArgTy);
      Location loc = funcOp.getLoc();

      auto alloca =
          cir::AllocaOp::create(rewriter, loc, srcPtrTy, newArgTy,
                                /*name=*/rewriter.getStringAttr("coerce"),
                                /*alignment=*/rewriter.getI64IntegerAttr(8));

      auto store = cir::StoreOp::create(rewriter, loc, blockArg, alloca,
                                        /*isVolatile=*/mlir::UnitAttr(),
                                        /*alignment=*/mlir::IntegerAttr(),
                                        /*sync_scope=*/cir::SyncScopeKindAttr(),
                                        /*mem_order=*/cir::MemOrderAttr());

      auto ptrCast = cir::CastOp::create(rewriter, loc, dstPtrTy,
                                         cir::CastKind::bitcast, alloca);

      auto load = cir::LoadOp::create(rewriter, loc, oldArgTy, ptrCast,
                                      /*isDeref=*/mlir::UnitAttr(),
                                      /*isVolatile=*/mlir::UnitAttr(),
                                      /*alignment=*/mlir::IntegerAttr(),
                                      /*sync_scope=*/cir::SyncScopeKindAttr(),
                                      /*mem_order=*/cir::MemOrderAttr());

      adapted = load;
      coercionOps.insert(alloca.getOperation());
      coercionOps.insert(store.getOperation());
      coercionOps.insert(ptrCast.getOperation());
      coercionOps.insert(load.getOperation());
    }
    lastInserted = adapted.getDefiningOp();

    blockArg.replaceAllUsesExcept(adapted, coercionOps);
  }
}

LogicalResult CIRABIRewriteContext::rewriteFunctionDefinition(
    FunctionOpInterface funcOp, const FunctionClassification &fc,
    OpBuilder &rewriter) {
  ArrayRef<Type> oldArgTypes = funcOp.getArgumentTypes();
  ArrayRef<Type> oldResultTypes = funcOp.getResultTypes();
  bool isDecl = funcOp.isDeclaration();

  bool hasSRet =
      fc.returnInfo.kind == ArgKind::Indirect && !oldResultTypes.empty();
  bool returnCoerced = false;
  bool hasArgChanges = false;
  SmallVector<unsigned> ignoredArgIndices;

  // Compute new argument types.
  SmallVector<Type> newArgTypes;

  if (hasSRet) {
    auto ptrTy = cir::PointerType::get(oldResultTypes[0]);
    newArgTypes.push_back(ptrTy);
  }

  for (auto [idx, argClass] : llvm::enumerate(fc.argInfos)) {
    Type origTy = oldArgTypes[idx];
    switch (argClass.kind) {
    case ArgKind::Direct:
    case ArgKind::Extend:
      if (argClass.canFlatten && argClass.coercedType) {
        if (auto coercedRec = dyn_cast<cir::RecordType>(argClass.coercedType)) {
          if (coercedRec.getMembers().size() > 1) {
            for (mlir::Type fieldTy : coercedRec.getMembers())
              newArgTypes.push_back(fieldTy);
            hasArgChanges = true;
            break;
          }
        }
      }
      newArgTypes.push_back(argClass.coercedType ? argClass.coercedType
                                                 : origTy);
      if (argClass.coercedType && argClass.coercedType != origTy)
        hasArgChanges = true;
      break;
    case ArgKind::Indirect:
      newArgTypes.push_back(cir::PointerType::get(origTy));
      hasArgChanges = true;
      break;
    case ArgKind::Ignore:
      ignoredArgIndices.push_back(idx);
      hasArgChanges = true;
      break;
    case ArgKind::Expand:
      newArgTypes.push_back(origTy);
      break;
    }
  }

  // Compute new result type.  CIR's FuncType::clone expects exactly
  // one result type (VoidType for void-returning functions).
  auto voidTy = cir::VoidType::get(funcOp->getContext());
  Type origRetTy = oldResultTypes.empty() ? voidTy : oldResultTypes[0];
  Type newRetTy = origRetTy;

  if (hasSRet) {
    newRetTy = voidTy;
  } else if (fc.returnInfo.kind == ArgKind::Direct ||
             fc.returnInfo.kind == ArgKind::Extend) {
    if (fc.returnInfo.coercedType && !oldResultTypes.empty() &&
        fc.returnInfo.coercedType != oldResultTypes[0]) {
      newRetTy = fc.returnInfo.coercedType;
      returnCoerced = true;
    }
  } else if (fc.returnInfo.kind == ArgKind::Ignore) {
    newRetTy = voidTy;
  }

  SmallVector<Type> newResultTypes = {newRetTy};

  // If nothing changed, skip the rewrite -- unless we have
  // Extend args/returns that need signext/zeroext attrs.
  bool hasExtend = fc.returnInfo.kind == ArgKind::Extend;
  for (auto &argClass : fc.argInfos)
    if (argClass.kind == ArgKind::Extend)
      hasExtend = true;
  if (!hasSRet && !hasArgChanges && !hasExtend && !returnCoerced &&
      newRetTy == origRetTy && newArgTypes == SmallVector<Type>(oldArgTypes))
    return success();

  // Body modifications only apply to definitions.
  if (!isDecl) {
    if (hasSRet) {
      auto ptrTy = cir::PointerType::get(oldResultTypes[0]);
      Region &body = funcOp->getRegion(0);
      body.insertArgument(0u, ptrTy, funcOp.getLoc());
      insertSRetStores(funcOp, oldResultTypes[0], rewriter);
    }

    unsigned sretOffset = hasSRet ? 1 : 0;
    if (hasArgChanges)
      insertArgAdaptation(funcOp, fc, sretOffset, rewriter);

    // For Indirect args, the block argument type was changed to a
    // pointer.  Insert a load at function entry so existing uses see
    // the original value type.
    if (!funcOp->getRegion(0).empty()) {
      Block &entry = funcOp->getRegion(0).front();
      for (auto [idx, argClass] : llvm::enumerate(fc.argInfos)) {
        if (argClass.kind != ArgKind::Indirect)
          continue;
        unsigned blockIdx = idx + sretOffset;
        BlockArgument blockArg = entry.getArgument(blockIdx);
        Type origArgTy = oldArgTypes[idx];
        auto ptrTy = cir::PointerType::get(origArgTy);
        blockArg.setType(ptrTy);

        rewriter.setInsertionPointToStart(&entry);
        auto load =
            cir::LoadOp::create(rewriter, funcOp.getLoc(), origArgTy, blockArg,
                                /*isDeref=*/mlir::UnitAttr(),
                                /*isVolatile=*/mlir::UnitAttr(),
                                /*alignment=*/mlir::IntegerAttr(),
                                /*sync_scope=*/cir::SyncScopeKindAttr(),
                                /*mem_order=*/cir::MemOrderAttr());
        SmallPtrSet<Operation *, 1> loadOps;
        loadOps.insert(load.getOperation());
        blockArg.replaceAllUsesExcept(load, loadOps);
      }
    }

    // For flattened Direct args (canFlatten + RecordType coercion),
    // replace the single struct block arg with N scalar block args
    // and reconstruct the struct at function entry.
    if (!funcOp->getRegion(0).empty()) {
      Block &entry = funcOp->getRegion(0).front();
      for (int i = fc.argInfos.size() - 1; i >= 0; --i) {
        auto &argClass = fc.argInfos[i];
        if (argClass.kind != ArgKind::Direct || !argClass.canFlatten ||
            !argClass.coercedType)
          continue;
        auto coercedRec = dyn_cast<cir::RecordType>(argClass.coercedType);
        if (!coercedRec || coercedRec.getMembers().size() <= 1)
          continue;

        unsigned origBlockIdx = i + sretOffset;
        Type origArgTy = oldArgTypes[i];
        auto members = coercedRec.getMembers();
        unsigned numFields = members.size();

        BlockArgument origArg = entry.getArgument(origBlockIdx);

        for (unsigned f = 0; f < numFields; ++f)
          entry.insertArgument(origBlockIdx + 1 + f, members[f],
                               funcOp.getLoc());

        rewriter.setInsertionPointToStart(&entry);
        Type coercedTy = argClass.coercedType;
        auto coercedPtrTy = cir::PointerType::get(coercedTy);
        auto alloca = cir::AllocaOp::create(
            rewriter, funcOp.getLoc(), coercedPtrTy, coercedTy,
            /*name=*/rewriter.getStringAttr("coerce"),
            /*alignment=*/rewriter.getI64IntegerAttr(8));

        for (unsigned f = 0; f < numFields; ++f) {
          BlockArgument scalarArg = entry.getArgument(origBlockIdx + 1 + f);
          auto memberPtr = cir::GetMemberOp::create(
              rewriter, funcOp.getLoc(), cir::PointerType::get(members[f]),
              alloca, /*name=*/"", f);
          cir::StoreOp::create(rewriter, funcOp.getLoc(), scalarArg, memberPtr,
                               /*isVolatile=*/mlir::UnitAttr(),
                               /*alignment=*/mlir::IntegerAttr(),
                               /*sync_scope=*/cir::SyncScopeKindAttr(),
                               /*mem_order=*/cir::MemOrderAttr());
        }

        auto origPtrTy = cir::PointerType::get(origArgTy);
        auto ptrCast = cir::CastOp::create(rewriter, funcOp.getLoc(), origPtrTy,
                                           cir::CastKind::bitcast, alloca);
        auto loaded =
            cir::LoadOp::create(rewriter, funcOp.getLoc(), origArgTy, ptrCast,
                                /*isDeref=*/mlir::UnitAttr(),
                                /*isVolatile=*/mlir::UnitAttr(),
                                /*alignment=*/mlir::IntegerAttr(),
                                /*sync_scope=*/cir::SyncScopeKindAttr(),
                                /*mem_order=*/cir::MemOrderAttr());

        origArg.replaceAllUsesWith(loaded);
        entry.eraseArgument(origBlockIdx);
      }
    }

    // Erase block arguments for Ignore'd args (in reverse to keep
    // indices valid).  Replace any remaining uses with undef first.
    if (!ignoredArgIndices.empty()) {
      Region &body = funcOp->getRegion(0);
      if (!body.empty()) {
        Block &entry = body.front();
        unsigned sretOff = hasSRet ? 1 : 0;
        for (int i = ignoredArgIndices.size() - 1; i >= 0; --i) {
          unsigned blockIdx = ignoredArgIndices[i] + sretOff;
          if (blockIdx < entry.getNumArguments()) {
            BlockArgument arg = entry.getArgument(blockIdx);
            if (!arg.use_empty()) {
              rewriter.setInsertionPointToStart(&entry);
              auto ptrTy = cir::PointerType::get(arg.getType());
              auto alloca = cir::AllocaOp::create(
                  rewriter, funcOp.getLoc(), ptrTy, arg.getType(),
                  /*name=*/rewriter.getStringAttr("ignored"),
                  /*alignment=*/rewriter.getI64IntegerAttr(1));
              auto load = cir::LoadOp::create(
                  rewriter, funcOp.getLoc(), arg.getType(), alloca,
                  /*isDeref=*/mlir::UnitAttr(),
                  /*isVolatile=*/mlir::UnitAttr(),
                  /*alignment=*/mlir::IntegerAttr(),
                  /*sync_scope=*/cir::SyncScopeKindAttr(),
                  /*mem_order=*/cir::MemOrderAttr());
              arg.replaceAllUsesWith(load);
            }
            entry.eraseArgument(blockIdx);
          }
        }
      }
    }

    if (returnCoerced)
      insertReturnCoercion(funcOp, origRetTy, fc.returnInfo.coercedType,
                           rewriter);

    // When the return type is Ignore (empty struct), rewrite all
    // return ops to drop their operand so they return void.
    if (fc.returnInfo.kind == ArgKind::Ignore && !oldResultTypes.empty()) {
      funcOp.walk([&](cir::ReturnOp retOp) {
        if (retOp.getNumOperands() > 0) {
          rewriter.setInsertionPoint(retOp);
          cir::ReturnOp::create(rewriter, retOp.getLoc());
          retOp->erase();
        }
      });
    }
  }

  Type newFnTy = funcOp.cloneTypeWith(newArgTypes, newResultTypes);
  funcOp.setFunctionTypeAttr(TypeAttr::get(newFnTy));

  // Attach sret / byval / signext / zeroext attributes.
  {
    MLIRContext *ctx = funcOp->getContext();
    unsigned numArgs = newArgTypes.size();
    bool needsAttrs = hasSRet;
    bool hasIgnoredArgs = !ignoredArgIndices.empty();
    for (auto &argClass : fc.argInfos) {
      if (argClass.kind == ArgKind::Indirect)
        needsAttrs = true;
      if (argClass.kind == ArgKind::Extend)
        needsAttrs = true;
    }
    if ((hasIgnoredArgs || hasArgChanges) && funcOp->hasAttr("arg_attrs"))
      needsAttrs = true;

    if (needsAttrs) {
      SmallVector<Attribute> argAttrDicts(numArgs, DictionaryAttr::get(ctx));

      // Preserve existing arg_attrs, shifting by sretOff and
      // skipping Ignore'd args.
      unsigned sretOff = hasSRet ? 1 : 0;
      if (auto existingAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs")) {
        unsigned newIdx = sretOff;
        for (unsigned oldIdx = 0; oldIdx < existingAttrs.size(); ++oldIdx) {
          if (oldIdx < fc.argInfos.size() &&
              fc.argInfos[oldIdx].kind == ArgKind::Ignore)
            continue;
          if (newIdx < numArgs)
            argAttrDicts[newIdx] = existingAttrs[oldIdx];
          if (oldIdx < fc.argInfos.size()) {
            auto &ac = fc.argInfos[oldIdx];
            if (ac.canFlatten && ac.coercedType)
              if (auto coercedRec = dyn_cast<cir::RecordType>(ac.coercedType))
                if (coercedRec.getMembers().size() > 1) {
                  newIdx += coercedRec.getMembers().size();
                  continue;
                }
          }
          ++newIdx;
        }
      }

      if (hasSRet) {
        SmallVector<NamedAttribute> sretAttrs;
        sretAttrs.push_back(rewriter.getNamedAttr(
            "llvm.sret", TypeAttr::get(oldResultTypes[0])));
        sretAttrs.push_back(rewriter.getNamedAttr(
            "llvm.align",
            rewriter.getI64IntegerAttr(fc.returnInfo.indirectAlign.value())));
        if (!funcOp.isDeclaration())
          sretAttrs.push_back(
              rewriter.getNamedAttr("llvm.noalias", rewriter.getUnitAttr()));
        sretAttrs.push_back(
            rewriter.getNamedAttr("llvm.writable", rewriter.getUnitAttr()));
        sretAttrs.push_back(rewriter.getNamedAttr("llvm.dead_on_unwind",
                                                  rewriter.getUnitAttr()));
        argAttrDicts[0] = DictionaryAttr::get(ctx, sretAttrs);
      }

      unsigned sretOff2 = hasSRet ? 1 : 0;
      for (auto [idx, argClass] : llvm::enumerate(fc.argInfos)) {
        if (argClass.kind == ArgKind::Indirect) {
          if (argClass.byVal) {
            SmallVector<NamedAttribute> byvalAttrs;
            byvalAttrs.push_back(rewriter.getNamedAttr(
                "llvm.byval", TypeAttr::get(oldArgTypes[idx])));
            byvalAttrs.push_back(rewriter.getNamedAttr(
                "llvm.align",
                rewriter.getI64IntegerAttr(argClass.indirectAlign.value())));
            byvalAttrs.push_back(
                rewriter.getNamedAttr("llvm.noundef", rewriter.getUnitAttr()));
            if (passByValueIsNoAlias)
              byvalAttrs.push_back(rewriter.getNamedAttr(
                  "llvm.noalias", rewriter.getUnitAttr()));
            argAttrDicts[idx + sretOff2] = DictionaryAttr::get(ctx, byvalAttrs);
          } else {
            SmallVector<NamedAttribute> indirectAttrs;
            indirectAttrs.push_back(
                rewriter.getNamedAttr("llvm.noundef", rewriter.getUnitAttr()));
            argAttrDicts[idx + sretOff2] =
                DictionaryAttr::get(ctx, indirectAttrs);
          }
        }

        if (argClass.kind == ArgKind::Extend) {
          if (idx + sretOff2 >= numArgs)
            continue;
          auto existing =
              mlir::cast<DictionaryAttr>(argAttrDicts[idx + sretOff2]);
          SmallVector<NamedAttribute> attrs(existing.begin(), existing.end());
          StringRef attrName =
              argClass.signExtend ? "llvm.signext" : "llvm.zeroext";
          attrs.push_back(
              rewriter.getNamedAttr(attrName, rewriter.getUnitAttr()));
          argAttrDicts[idx + sretOff2] = DictionaryAttr::get(ctx, attrs);
        }
      }

      funcOp->setAttr("arg_attrs", ArrayAttr::get(ctx, argAttrDicts));
    }

    // Add signext/zeroext to return value for Extend returns.
    if (fc.returnInfo.kind == ArgKind::Extend) {
      SmallVector<NamedAttribute> retAttrs;
      if (auto existing = funcOp->getAttrOfType<ArrayAttr>("res_attrs"))
        if (existing.size() > 0)
          for (auto attr : mlir::cast<DictionaryAttr>(existing[0]))
            retAttrs.push_back(attr);
      StringRef attrName =
          fc.returnInfo.signExtend ? "llvm.signext" : "llvm.zeroext";
      retAttrs.push_back(
          rewriter.getNamedAttr(attrName, rewriter.getUnitAttr()));
      SmallVector<Attribute> resAttrDicts;
      resAttrDicts.push_back(DictionaryAttr::get(ctx, retAttrs));
      funcOp->setAttr("res_attrs", ArrayAttr::get(ctx, resAttrDicts));
    }
  }

  return success();
}

LogicalResult CIRABIRewriteContext::rewriteCallSite(
    Operation *callOp, const FunctionClassification &fc, OpBuilder &rewriter) {
  auto call = cast<cir::CallOp>(callOp);
  bool hasSRet = fc.returnInfo.kind == ArgKind::Indirect;

  SmallVector<Value> newArgs;
  bool argsChanged = false;
  auto argOperands = call.getArgOperands();

  for (auto [idx, argClass] : llvm::enumerate(fc.argInfos)) {
    if (idx >= argOperands.size())
      break;

    Value arg = argOperands[idx];

    if (argClass.kind == ArgKind::Ignore) {
      argsChanged = true;
      continue;
    }

    if (argClass.kind == ArgKind::Indirect) {
      rewriter.setInsertionPoint(call);
      Type origTy = arg.getType();
      auto ptrTy = cir::PointerType::get(origTy);
      auto alloca = cir::AllocaOp::create(
          rewriter, call.getLoc(), ptrTy, origTy,
          /*name=*/
          rewriter.getStringAttr(argClass.byVal ? "byval" : "indirect"),
          /*alignment=*/
          rewriter.getI64IntegerAttr(argClass.indirectAlign.value()));
      cir::StoreOp::create(rewriter, call.getLoc(), arg, alloca,
                           /*isVolatile=*/mlir::UnitAttr(),
                           /*alignment=*/mlir::IntegerAttr(),
                           /*sync_scope=*/cir::SyncScopeKindAttr(),
                           /*mem_order=*/cir::MemOrderAttr());
      newArgs.push_back(alloca);
      argsChanged = true;
      continue;
    }

    // Flatten multi-register struct args into individual scalars.
    if (argClass.kind == ArgKind::Direct && argClass.canFlatten &&
        argClass.coercedType) {
      if (auto coercedRec = dyn_cast<cir::RecordType>(argClass.coercedType)) {
        if (coercedRec.getMembers().size() > 1) {
          rewriter.setInsertionPoint(call);
          Type origTy = arg.getType();
          auto origPtrTy = cir::PointerType::get(origTy);
          auto alloca = cir::AllocaOp::create(
              rewriter, call.getLoc(), origPtrTy, origTy,
              /*name=*/rewriter.getStringAttr("coerce"),
              /*alignment=*/rewriter.getI64IntegerAttr(8));
          cir::StoreOp::create(rewriter, call.getLoc(), arg, alloca,
                               /*isVolatile=*/mlir::UnitAttr(),
                               /*alignment=*/mlir::IntegerAttr(),
                               /*sync_scope=*/cir::SyncScopeKindAttr(),
                               /*mem_order=*/cir::MemOrderAttr());
          auto coercedPtrTy = cir::PointerType::get(argClass.coercedType);
          auto ptrCast =
              cir::CastOp::create(rewriter, call.getLoc(), coercedPtrTy,
                                  cir::CastKind::bitcast, alloca);
          for (auto [f, fieldTy] : llvm::enumerate(coercedRec.getMembers())) {
            auto memberPtr = cir::GetMemberOp::create(
                rewriter, call.getLoc(), cir::PointerType::get(fieldTy),
                ptrCast, /*name=*/"", f);
            auto fieldVal =
                cir::LoadOp::create(rewriter, call.getLoc(), fieldTy, memberPtr,
                                    /*isDeref=*/mlir::UnitAttr(),
                                    /*isVolatile=*/mlir::UnitAttr(),
                                    /*alignment=*/mlir::IntegerAttr(),
                                    /*sync_scope=*/cir::SyncScopeKindAttr(),
                                    /*mem_order=*/cir::MemOrderAttr());
            newArgs.push_back(fieldVal);
          }
          argsChanged = true;
          continue;
        }
      }
    }

    if ((argClass.kind == ArgKind::Extend ||
         argClass.kind == ArgKind::Direct) &&
        argClass.coercedType && arg.getType() != argClass.coercedType) {
      rewriter.setInsertionPoint(call);
      Value coerced;
      if (argClass.kind == ArgKind::Extend)
        coerced =
            cir::CastOp::create(rewriter, call.getLoc(), argClass.coercedType,
                                cir::CastKind::integral, arg);
      else
        coerced =
            emitCoercion(rewriter, call.getLoc(), argClass.coercedType, arg);
      newArgs.push_back(coerced);
      argsChanged = true;
    } else {
      newArgs.push_back(arg);
    }
  }

  // Pass through any extra operands beyond classified args.
  for (unsigned i = fc.argInfos.size(); i < argOperands.size(); ++i)
    newArgs.push_back(argOperands[i]);

  // Handle indirect return (sret) at call site.
  if (hasSRet && call.getNumResults() > 0) {
    Type origRetTy = call.getResult().getType();
    auto ptrTy = cir::PointerType::get(origRetTy);

    rewriter.setInsertionPoint(call);
    auto alloca =
        cir::AllocaOp::create(rewriter, call.getLoc(), ptrTy, origRetTy,
                              /*name=*/rewriter.getStringAttr("sret"),
                              /*alignment=*/rewriter.getI64IntegerAttr(8));

    SmallVector<Value> sretArgs;
    sretArgs.push_back(alloca);
    sretArgs.append(newArgs.begin(), newArgs.end());

    auto voidTy = cir::VoidType::get(call.getContext());
    auto newCall = cir::CallOp::create(rewriter, call.getLoc(),
                                       call.getCalleeAttr(), voidTy, sretArgs);
    for (NamedAttribute attr : call->getAttrs())
      if (!newCall->hasAttr(attr.getName()))
        newCall->setAttr(attr.getName(), attr.getValue());

    rewriter.setInsertionPointAfter(newCall);
    auto load = cir::LoadOp::create(rewriter, call.getLoc(), origRetTy, alloca,
                                    /*isDeref=*/mlir::UnitAttr(),
                                    /*isVolatile=*/mlir::UnitAttr(),
                                    /*alignment=*/mlir::IntegerAttr(),
                                    /*sync_scope=*/cir::SyncScopeKindAttr(),
                                    /*mem_order=*/cir::MemOrderAttr());

    call.getResult().replaceAllUsesWith(load);
    call->erase();
    return success();
  }

  // Handle direct return coercion.
  bool returnCoerced = false;
  Type coercedRetTy;
  if ((fc.returnInfo.kind == ArgKind::Direct ||
       fc.returnInfo.kind == ArgKind::Extend) &&
      fc.returnInfo.coercedType) {
    returnCoerced = true;
    coercedRetTy = fc.returnInfo.coercedType;
  }

  // Handle Ignore return: replace with void call.
  if (fc.returnInfo.kind == ArgKind::Ignore && call.getNumResults() > 0) {
    rewriter.setInsertionPoint(call);
    auto voidTy = cir::VoidType::get(call.getContext());
    auto newCall = cir::CallOp::create(rewriter, call.getLoc(),
                                       call.getCalleeAttr(), voidTy, newArgs);
    for (NamedAttribute attr : call->getAttrs())
      if (!newCall->hasAttr(attr.getName()))
        newCall->setAttr(attr.getName(), attr.getValue());

    if (!call.getResult().use_empty()) {
      rewriter.setInsertionPointAfter(newCall);
      Type origRetTy = call.getResult().getType();
      auto ptrTy = cir::PointerType::get(origRetTy);
      auto alloca =
          cir::AllocaOp::create(rewriter, call.getLoc(), ptrTy, origRetTy,
                                /*name=*/rewriter.getStringAttr("ignored"),
                                /*alignment=*/rewriter.getI64IntegerAttr(1));
      auto load =
          cir::LoadOp::create(rewriter, call.getLoc(), origRetTy, alloca,
                              /*isDeref=*/mlir::UnitAttr(),
                              /*isVolatile=*/mlir::UnitAttr(),
                              /*alignment=*/mlir::IntegerAttr(),
                              /*sync_scope=*/cir::SyncScopeKindAttr(),
                              /*mem_order=*/cir::MemOrderAttr());
      call.getResult().replaceAllUsesWith(load);
    }
    call->erase();
    return success();
  }

  if (!returnCoerced && !argsChanged)
    return success();

  Type callRetTy;
  Type origRetTy;
  bool hasResult = call.getNumResults() > 0;

  if (hasResult) {
    origRetTy = call.getResult().getType();
    callRetTy = returnCoerced ? coercedRetTy : origRetTy;
  } else {
    callRetTy = cir::VoidType::get(call.getContext());
  }

  rewriter.setInsertionPoint(call);
  auto newCall = cir::CallOp::create(rewriter, call.getLoc(),
                                     call.getCalleeAttr(), callRetTy, newArgs);
  for (NamedAttribute attr : call->getAttrs())
    if (!newCall->hasAttr(attr.getName()))
      newCall->setAttr(attr.getName(), attr.getValue());

  if (hasResult && returnCoerced && origRetTy != coercedRetTy) {
    rewriter.setInsertionPointAfter(newCall);
    Value castBack =
        emitCoercion(rewriter, call.getLoc(), origRetTy, newCall.getResult());
    call.getResult().replaceAllUsesWith(castBack);
  } else if (hasResult) {
    call.getResult().replaceAllUsesWith(newCall.getResult());
  }

  call->erase();
  return success();
}
