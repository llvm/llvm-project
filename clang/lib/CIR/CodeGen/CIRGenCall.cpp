//===--- CIRGenCall.cpp - Encapsulate calling convention details ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function definition used
// to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCall.h"
#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenFunctionInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/TypeSize.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenFunctionInfo *
CIRGenFunctionInfo::create(FunctionType::ExtInfo info, CanQualType resultType,
                           llvm::ArrayRef<CanQualType> argTypes,
                           RequiredArgs required) {
  // The first slot allocated for arg type slot is for the return value.
  void *buffer = operator new(
      totalSizeToAlloc<CanQualType>(argTypes.size() + 1));

  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoParamInfo());

  CIRGenFunctionInfo *fi = new (buffer) CIRGenFunctionInfo();

  fi->noReturn = info.getNoReturn();

  fi->required = required;
  fi->numArgs = argTypes.size();

  fi->getArgTypes()[0] = resultType;
  std::copy(argTypes.begin(), argTypes.end(), fi->argTypesBegin());
  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoExtParamInfo());

  return fi;
}

cir::FuncType CIRGenTypes::getFunctionType(GlobalDecl gd) {
  const CIRGenFunctionInfo &fi = arrangeGlobalDeclaration(gd);
  return getFunctionType(fi);
}

cir::FuncType CIRGenTypes::getFunctionType(const CIRGenFunctionInfo &info) {
  mlir::Type resultType = convertType(info.getReturnType());
  SmallVector<mlir::Type, 8> argTypes;
  argTypes.reserve(info.getNumRequiredArgs());

  for (const CanQualType &argType : info.requiredArguments())
    argTypes.push_back(convertType(argType));

  return cir::FuncType::get(argTypes,
                            (resultType ? resultType : builder.getVoidTy()),
                            info.isVariadic());
}

cir::FuncType CIRGenTypes::getFunctionTypeForVTable(GlobalDecl gd) {
  const CXXMethodDecl *md = cast<CXXMethodDecl>(gd.getDecl());
  const FunctionProtoType *fpt = md->getType()->getAs<FunctionProtoType>();

  if (!isFuncTypeConvertible(fpt))
    cgm.errorNYI("getFunctionTypeForVTable: non-convertible function type");

  return getFunctionType(gd);
}

CIRGenCallee CIRGenCallee::prepareConcreteCallee(CIRGenFunction &cgf) const {
  if (isVirtual()) {
    const CallExpr *ce = getVirtualCallExpr();
    return cgf.cgm.getCXXABI().getVirtualFunctionPointer(
        cgf, getVirtualMethodDecl(), getThisAddress(), getVirtualFunctionType(),
        ce ? ce->getBeginLoc() : SourceLocation());
  }
  return *this;
}

void CIRGenFunction::emitAggregateStore(mlir::Value value, Address dest) {
  // In classic codegen:
  // Function to store a first-class aggregate into memory. We prefer to
  // store the elements rather than the aggregate to be more friendly to
  // fast-isel.
  // In CIR codegen:
  // Emit the most simple cir.store possible (e.g. a store for a whole
  // record), which can later be broken down in other CIR levels (or prior
  // to dialect codegen).

  // Stored result for the callers of this function expected to be in the same
  // scope as the value, don't make assumptions about current insertion point.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(value.getDefiningOp());
  builder.createStore(*currSrcLoc, value, dest);
}

static void addAttributesFromFunctionProtoType(CIRGenBuilderTy &builder,
                                               mlir::NamedAttrList &attrs,
                                               const FunctionProtoType *fpt) {
  if (!fpt)
    return;

  if (!isUnresolvedExceptionSpec(fpt->getExceptionSpecType()) &&
      fpt->isNothrow())
    attrs.set(cir::CIRDialect::getNoThrowAttrName(),
              mlir::UnitAttr::get(builder.getContext()));
}

static void addNoBuiltinAttributes(mlir::MLIRContext &ctx,
                                   mlir::NamedAttrList &attrs,
                                   const LangOptions &langOpts,
                                   const NoBuiltinAttr *nba = nullptr) {
  // First, handle the language options passed through -fno-builtin.
  // or, if there is a wildcard in the builtin names specified through the
  // attribute, disable them all.
  if (langOpts.NoBuiltin ||
      (nba && llvm::is_contained(nba->builtinNames(), "*"))) {
    // -fno-builtin disables them all.
    // Empty attribute means 'all'.
    attrs.set(cir::CIRDialect::getNoBuiltinsAttrName(),
              mlir::ArrayAttr::get(&ctx, {}));
    return;
  }

  llvm::SetVector<mlir::Attribute> nbFuncs;
  auto addNoBuiltinAttr = [&ctx, &nbFuncs](StringRef builtinName) {
    nbFuncs.insert(mlir::StringAttr::get(&ctx, builtinName));
  };

  // Then, add attributes for builtins specified through -fno-builtin-<name>.
  llvm::for_each(langOpts.NoBuiltinFuncs, addNoBuiltinAttr);

  // Now, let's check the __attribute__((no_builtin("...")) attribute added to
  // the source.
  if (nba)
    llvm::for_each(nba->builtinNames(), addNoBuiltinAttr);

  if (!nbFuncs.empty())
    attrs.set(cir::CIRDialect::getNoBuiltinsAttrName(),
              mlir::ArrayAttr::get(&ctx, nbFuncs.getArrayRef()));
}

/// Add denormal-fp-math and denormal-fp-math-f32 as appropriate for the
/// requested denormal behavior, accounting for the overriding behavior of the
/// -f32 case.
static void addDenormalModeAttrs(llvm::DenormalMode fpDenormalMode,
                                 llvm::DenormalMode fp32DenormalMode,
                                 mlir::NamedAttrList &attrs) {
  // TODO(cir): Classic-codegen sets the denormal modes here. There are two
  // values, both with a string, but it seems that perhaps we could combine
  // these into a single attribute?  It seems a little silly to have two so
  // similar named attributes that do the same thing.
}

/// Add default attributes to a function, which have merge semantics under
/// -mlink-builtin-bitcode and should not simply overwrite any existing
/// attributes in the linked library.
static void
addMergeableDefaultFunctionAttributes(const CodeGenOptions &codeGenOpts,
                                      mlir::NamedAttrList &attrs) {
  addDenormalModeAttrs(codeGenOpts.FPDenormalMode, codeGenOpts.FP32DenormalMode,
                       attrs);
}

static llvm::StringLiteral
getZeroCallUsedRegsKindStr(llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind k) {
  switch (k) {
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::Skip:
    llvm_unreachable("No string value, shouldn't be able to get here");
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedGPRArg:
    return "used-gpr-arg";
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedGPR:
    return "used-gpr";
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedArg:
    return "used-arg";
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::Used:
    return "used";
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllGPRArg:
    return "all-gpr-arg";
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllGPR:
    return "all-gpr";
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllArg:
    return "all-arg";
  case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::All:
    return "all";
  }

  llvm_unreachable("Unknown kind?");
}

/// Add default attributes to a function, which have merge semantics under
/// -mlink-builtin-bitcode and should not simply overwrite any existing
/// attributes in the linked library.
static void addTrivialDefaultFunctionAttributes(
    mlir::MLIRContext *mlirCtx, StringRef name, bool hasOptNoneAttr,
    const CodeGenOptions &codeGenOpts, const LangOptions &langOpts,
    bool attrOnCallSite, mlir::NamedAttrList &attrs) {
  // TODO(cir): Handle optimize attribute flag here.
  // OptimizeNoneAttr takes precedence over -Os or -Oz. No warning needed.
  if (!hasOptNoneAttr) {
    if (codeGenOpts.OptimizeSize)
      attrs.set(cir::CIRDialect::getOptimizeForSizeAttrName(),
                mlir::UnitAttr::get(mlirCtx));
    if (codeGenOpts.OptimizeSize == 2)
      attrs.set(cir::CIRDialect::getMinSizeAttrName(),
                mlir::UnitAttr::get(mlirCtx));
  }

  // TODO(cir): Classic codegen adds 'DisableRedZone', 'indirect-tls-seg-refs'
  // and 'NoImplicitFloat' here.

  if (attrOnCallSite) {
    // Add the 'nobuiltin' tag, which is different from 'no-builtins'.
    if (!codeGenOpts.SimplifyLibCalls || langOpts.isNoBuiltinFunc(name))
      attrs.set(cir::CIRDialect::getNoBuiltinAttrName(),
                mlir::UnitAttr::get(mlirCtx));

    if (!codeGenOpts.TrapFuncName.empty())
      attrs.set(cir::CIRDialect::getTrapFuncNameAttrName(),
                mlir::StringAttr::get(mlirCtx, codeGenOpts.TrapFuncName));
  } else {
    // TODO(cir): Set frame pointer attribute here.
    // TODO(cir): a number of other attribute 1-offs based on codegen/lang opts
    // should be done here: less-recise-fpmad null-pointer-is-valid
    // no-trapping-math
    // various inf/nan/nsz/etc work here.
    //
    // TODO(cir): set stack-protector buffer size attribute (sorted oddly in
    // classic compiler inside of the above region, but should be done on its
    // own).
    // TODO(cir): other attributes here:
    // reciprocal estimates, prefer-vector-width, stackrealign, backchain,
    // split-stack, speculative-load-hardening.

    if (codeGenOpts.getZeroCallUsedRegs() ==
        llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::Skip)
      attrs.erase(cir::CIRDialect::getZeroCallUsedRegsAttrName());
    else
      attrs.set(cir::CIRDialect::getZeroCallUsedRegsAttrName(),
                mlir::StringAttr::get(mlirCtx,
                                      getZeroCallUsedRegsKindStr(
                                          codeGenOpts.getZeroCallUsedRegs())));
  }

  if (langOpts.assumeFunctionsAreConvergent()) {
    // Conservatively, mark all functions and calls in CUDA and OpenCL as
    // convergent (meaning, they may call an intrinsically convergent op, such
    // as __syncthreads() / barrier(), and so can't have certain optimizations
    // applied around them).  LLVM will remove this attribute where it safely
    // can.
    attrs.set(cir::CIRDialect::getConvergentAttrName(),
              mlir::UnitAttr::get(mlirCtx));
  }

  // TODO(cir): Classic codegen adds 'nounwind' here in a bunch of offload
  // targets.

  if (codeGenOpts.SaveRegParams && !attrOnCallSite)
    attrs.set(cir::CIRDialect::getSaveRegParamsAttrName(),
              mlir::UnitAttr::get(mlirCtx));

  // These come in the form of an optional equality sign, so make sure we pass
  // these on correctly. These will eventually just be passed through to
  // LLVM-IR, but we want to put them all in 1 array to simplify the
  // LLVM-MLIR dialect.
  SmallVector<mlir::NamedAttribute> defaultFuncAttrs;
  llvm::transform(
      codeGenOpts.DefaultFunctionAttrs, std::back_inserter(defaultFuncAttrs),
      [mlirCtx](llvm::StringRef arg) {
        auto [var, value] = arg.split('=');
        auto valueAttr =
            value.empty()
                ? cast<mlir::Attribute>(mlir::UnitAttr::get(mlirCtx))
                : cast<mlir::Attribute>(mlir::StringAttr::get(mlirCtx, value));
        return mlir::NamedAttribute(var, valueAttr);
      });

  if (!defaultFuncAttrs.empty())
    attrs.set(cir::CIRDialect::getDefaultFuncAttrsAttrName(),
              mlir::DictionaryAttr::get(mlirCtx, defaultFuncAttrs));

  // TODO(cir): Do branch protection attributes here.
}

/// This function matches the behavior of 'getDefaultFunctionAttributes' from
/// classic codegen, despite the similarity of its name to
/// 'addDefaultFunctionDefinitionAttributes', which is a caller of this
/// function.
void CIRGenModule::addDefaultFunctionAttributes(StringRef name,
                                                bool hasOptNoneAttr,
                                                bool attrOnCallSite,
                                                mlir::NamedAttrList &attrs) {

  addTrivialDefaultFunctionAttributes(&getMLIRContext(), name, hasOptNoneAttr,
                                      codeGenOpts, langOpts, attrOnCallSite,
                                      attrs);

  if (!attrOnCallSite) {
    // TODO(cir): Classic codegen adds pointer-auth attributes here, by calling
    // into TargetCodeGenInfo.  At the moment, we've not looked into this as it
    // is somewhat less used.
    addMergeableDefaultFunctionAttributes(codeGenOpts, attrs);
  }
}

/// Construct the CIR attribute list of a function or call.
void CIRGenModule::constructAttributeList(
    llvm::StringRef name, const CIRGenFunctionInfo &info,
    CIRGenCalleeInfo calleeInfo, mlir::NamedAttrList &attrs,
    mlir::NamedAttrList &retAttrs, cir::CallingConv &callingConv,
    cir::SideEffect &sideEffect, bool attrOnCallSite, bool isThunk) {
  assert(!cir::MissingFeatures::opCallCallConv());
  sideEffect = cir::SideEffect::All;

  auto addUnitAttr = [&](llvm::StringRef name) {
    attrs.set(name, mlir::UnitAttr::get(&getMLIRContext()));
  };

  if (info.isNoReturn())
    addUnitAttr(cir::CIRDialect::getNoReturnAttrName());

  // TODO(cir): Implement/check the CSME Nonsecure call attribute here. This
  // requires being in CSME mode.

  addAttributesFromFunctionProtoType(getBuilder(), attrs,
                                     calleeInfo.getCalleeFunctionProtoType());

  const Decl *targetDecl = calleeInfo.getCalleeDecl().getDecl();

  // TODO(cir): OMP Assume Attributes should be here.

  const NoBuiltinAttr *nba = nullptr;

  // TODO(cir): Some work for arg memory effects can be done here, as it is in
  // classic codegen.

  if (targetDecl) {
    if (targetDecl->hasAttr<NoThrowAttr>())
      addUnitAttr(cir::CIRDialect::getNoThrowAttrName());
    // TODO(cir): This is actually only possible if targetDecl isn't a
    // declarator, which ObjCMethodDecl seems to be the only way to get this to
    // happen.  We're including it here for completeness, but we should add a
    // test for this when we start generating ObjectiveC.
    if (targetDecl->hasAttr<NoReturnAttr>())
      addUnitAttr(cir::CIRDialect::getNoReturnAttrName());
    if (targetDecl->hasAttr<ReturnsTwiceAttr>())
      addUnitAttr(cir::CIRDialect::getReturnsTwiceAttrName());
    if (targetDecl->hasAttr<ColdAttr>())
      addUnitAttr(cir::CIRDialect::getColdAttrName());
    if (targetDecl->hasAttr<HotAttr>())
      addUnitAttr(cir::CIRDialect::getHotAttrName());
    if (targetDecl->hasAttr<NoDuplicateAttr>())
      addUnitAttr(cir::CIRDialect::getNoDuplicatesAttrName());
    if (targetDecl->hasAttr<ConvergentAttr>())
      addUnitAttr(cir::CIRDialect::getConvergentAttrName());

    if (const FunctionDecl *func = dyn_cast<FunctionDecl>(targetDecl)) {
      addAttributesFromFunctionProtoType(
          getBuilder(), attrs, func->getType()->getAs<FunctionProtoType>());

      // TODO(cir): When doing 'return attrs' we need to cover the 'NoAlias' for
      // global allocation functions here.
      assert(!cir::MissingFeatures::opCallAttrs());

      const CXXMethodDecl *md = dyn_cast<CXXMethodDecl>(func);
      bool isVirtualCall = md && md->isVirtual();

      // Don't use [[noreturn]], _Noreturn or [[no_builtin]] for a call to a
      // virtual function. These attributes are not inherited by overloads.
      if (!(attrOnCallSite && isVirtualCall)) {
        if (func->isNoReturn())
          addUnitAttr(cir::CIRDialect::getNoReturnAttrName());
        nba = func->getAttr<NoBuiltinAttr>();
      }
    }

    assert(!cir::MissingFeatures::opCallAttrs());

    // 'const', 'pure' and 'noalias' attributed functions are also nounwind.
    if (targetDecl->hasAttr<ConstAttr>()) {
      // gcc specifies that 'const' functions have greater restrictions than
      // 'pure' functions, so they also cannot have infinite loops.
      sideEffect = cir::SideEffect::Const;
    } else if (targetDecl->hasAttr<PureAttr>()) {
      // gcc specifies that 'pure' functions cannot have infinite loops.
      sideEffect = cir::SideEffect::Pure;
    }

    attrs.set(cir::CIRDialect::getSideEffectAttrName(),
              cir::SideEffectAttr::get(&getMLIRContext(), sideEffect));

    // TODO(cir): When doing 'return attrs' we need to cover the Restrict and
    // ReturnsNonNull attributes here.
    if (targetDecl->hasAttr<AnyX86NoCallerSavedRegistersAttr>())
      addUnitAttr(cir::CIRDialect::getNoCallerSavedRegsAttrName());
    // TODO(cir): Implement 'NoCFCheck' attribute here.  This requires
    // fcf-protection mode.
    if (targetDecl->hasAttr<LeafAttr>())
      addUnitAttr(cir::CIRDialect::getNoCallbackAttrName());
    // TODO(cir): Implement 'BPFFastCall' attribute here.  This requires C, and
    // the BPF target.

    if (auto *allocSizeAttr = targetDecl->getAttr<AllocSizeAttr>()) {
      unsigned size = allocSizeAttr->getElemSizeParam().getLLVMIndex();

      if (allocSizeAttr->getNumElemsParam().isValid()) {
        unsigned numElts = allocSizeAttr->getNumElemsParam().getLLVMIndex();
        attrs.set(cir::CIRDialect::getAllocSizeAttrName(),
                  builder.getDenseI32ArrayAttr(
                      {static_cast<int>(size), static_cast<int>(numElts)}));
      } else {
        attrs.set(cir::CIRDialect::getAllocSizeAttrName(),
                  builder.getDenseI32ArrayAttr({static_cast<int>(size)}));
      }
    }

    // TODO(cir): Quite a few CUDA and OpenCL attributes are added here, like
    // uniform-work-group-size.

    // TODO(cir): we should also do 'aarch64_pstate_sm_body' here.

    if (auto *modularFormat = targetDecl->getAttr<ModularFormatAttr>()) {
      FormatAttr *format = targetDecl->getAttr<FormatAttr>();
      StringRef type = format->getType()->getName();
      std::string formatIdx = std::to_string(format->getFormatIdx());
      std::string firstArg = std::to_string(format->getFirstArg());
      SmallVector<StringRef> args = {
          type, formatIdx, firstArg,
          modularFormat->getModularImplFn()->getName(),
          modularFormat->getImplName()};
      llvm::append_range(args, modularFormat->aspects());
      attrs.set(cir::CIRDialect::getModularFormatAttrName(),
                builder.getStringAttr(llvm::join(args, ",")));
    }
  }

  addNoBuiltinAttributes(getMLIRContext(), attrs, getLangOpts(), nba);

  bool hasOptNoneAttr = targetDecl && targetDecl->hasAttr<OptimizeNoneAttr>();
  addDefaultFunctionAttributes(name, hasOptNoneAttr, attrOnCallSite, attrs);
  if (targetDecl) {
    // TODO(cir): There is another region of `if (targetDecl)` that handles
    // removing some attributes that are necessary modifications of the
    // default-function attrs. Including:
    // NoSpeculativeLoadHardening
    // SpeculativeLoadHardening
    // NoSplitStack
    // Non-lazy-bind
    // 'sample-profile-suffix-elision-policy'.

    if (targetDecl->hasAttr<ZeroCallUsedRegsAttr>()) {
      // A function "__attribute__((...))" overrides the command-line flag.
      auto kind =
          targetDecl->getAttr<ZeroCallUsedRegsAttr>()->getZeroCallUsedRegs();
      attrs.set(
          cir::CIRDialect::getZeroCallUsedRegsAttrName(),
          mlir::StringAttr::get(
              &getMLIRContext(),
              ZeroCallUsedRegsAttr::ConvertZeroCallUsedRegsKindToStr(kind)));
    }

    if (targetDecl->hasAttr<NoConvergentAttr>())
      attrs.erase(cir::CIRDialect::getConvergentAttrName());
  }

  // TODO(cir): A bunch of non-call-site function IR attributes from
  // declaration-specific information, including tail calls,
  // cmse_nonsecure_entry, additional/automatic 'returns-twice' functions,
  // CPU-features/overrides, and hotpatch support.

  // TODO(cir): Add loader-replaceable attribute here.

  constructFunctionReturnAttributes(info, targetDecl, isThunk, retAttrs);
  constructFunctionArgumentAttributes();

  // TODO(cir): Arg attrs.

  assert(!cir::MissingFeatures::opCallAttrs());
}

bool CIRGenModule::hasStrictReturn(QualType retTy, const Decl *targetDecl) {
  // As-is msan can not tolerate noundef mismatch between caller and
  // implementation. Mismatch is possible for e.g. indirect calls from C-caller
  // into C++. Such mismatches lead to confusing false reports. To avoid
  // expensive workaround on msan we enforce initialization event in uncommon
  // cases where it's allowed.
  if (getLangOpts().Sanitize.has(SanitizerKind::Memory))
    return true;
  // C++ explicitly makes returning undefined values UB. C's rule only applies
  // to used values, so we never mark them noundef for now.
  if (!getLangOpts().CPlusPlus)
    return false;
  if (targetDecl) {
    if (const FunctionDecl *func = dyn_cast<FunctionDecl>(targetDecl)) {
      if (func->isExternC())
        return false;
    } else if (const VarDecl *var = dyn_cast<VarDecl>(targetDecl)) {
      // Function pointer.
      if (var->isExternC())
        return false;
    }
  }

  // We don't want to be too aggressive with the return checking, unless
  // it's explicit in the code opts or we're using an appropriate sanitizer.
  // Try to respect what the programmer intended.
  return getCodeGenOpts().StrictReturn ||
         !mayDropFunctionReturn(getASTContext(), retTy) ||
         getLangOpts().Sanitize.has(SanitizerKind::Return);
}

bool CIRGenModule::mayDropFunctionReturn(const ASTContext &context,
                                         QualType retTy) {
  // We can't just discard the return value for a record type with a
  // complex destructor or a non-trivially copyable type.
  if (const RecordType *recTy =
          retTy.getCanonicalType()->getAsCanonical<RecordType>()) {
    if (const auto *record = dyn_cast<CXXRecordDecl>(recTy->getDecl()))
      return record->hasTrivialDestructor();
  }
  return retTy.isTriviallyCopyableType(context);
}

static bool determineNoUndef(QualType clangTy, CIRGenTypes &types,
                             const cir::CIRDataLayout &layout,
                             const cir::ABIArgInfo &argInfo) {
  mlir::Type ty = types.convertTypeForMem(clangTy);
  assert(!cir::MissingFeatures::abiArgInfo());
  if (argInfo.isIndirect() || argInfo.isIndirectAliased())
    return true;
  if (argInfo.isExtend() && !argInfo.isNoExt())
    return true;

  if (cir::isSized(ty) && !layout.typeSizeEqualsStoreSize(ty))
    // TODO: This will result in a modest amount of values not marked noundef
    // when they could be. We care about values that *invisibly* contain undef
    // bits from the perspective of LLVM IR.
    return false;

  assert(!cir::MissingFeatures::opCallCallConv());
  // TODO(cir): The calling convention code needs to figure if the
  // coerced-to-type is larger than the actual type, and remove the noundef
  // attribute. Classic compiler did it here.
  if (clangTy->isBitIntType())
    return true;
  if (clangTy->isReferenceType())
    return true;
  if (clangTy->isNullPtrType())
    return false;
  if (clangTy->isMemberPointerType())
    // TODO: Some member pointers are `noundef`, but it depends on the ABI. For
    // now, never mark them.
    return false;
  if (clangTy->isScalarType()) {
    if (const ComplexType *Complex = dyn_cast<ComplexType>(clangTy))
      return determineNoUndef(Complex->getElementType(), types, layout,
                              argInfo);
    return true;
  }
  if (const VectorType *Vector = dyn_cast<VectorType>(clangTy))
    return determineNoUndef(Vector->getElementType(), types, layout, argInfo);
  if (const MatrixType *Matrix = dyn_cast<MatrixType>(clangTy))
    return determineNoUndef(Matrix->getElementType(), types, layout, argInfo);
  if (const ArrayType *Array = dyn_cast<ArrayType>(clangTy))
    return determineNoUndef(Array->getElementType(), types, layout, argInfo);

  // TODO: Some structs may be `noundef`, in specific situations.
  return false;
}

void CIRGenModule::constructFunctionReturnAttributes(
    const CIRGenFunctionInfo &info, const Decl *targetDecl, bool isThunk,
    mlir::NamedAttrList &retAttrs) {
  // Collect attributes from arguments and return values.
  QualType retTy = info.getReturnType();
  const cir::ABIArgInfo retInfo = info.getReturnInfo();
  const cir::CIRDataLayout &layout = getDataLayout();

  if (codeGenOpts.EnableNoundefAttrs && hasStrictReturn(retTy, targetDecl) &&
      !retTy->isVoidType() &&
      determineNoUndef(retTy, getTypes(), layout, retInfo))
    retAttrs.set(mlir::LLVM::LLVMDialect::getNoUndefAttrName(),
                 mlir::UnitAttr::get(&getMLIRContext()));

  // TODO(cir): classic codegen adds a bunch of attributes based on
  // calling-convention lowering results.  However, since calling conventions
  // haven't happened yet, this work likely has to happen there.

  if (!isThunk) {
    // TODO(cir): following comment taken from classic codegen, so if anything
    // happens there, we should reflect it here.
    // FIXME: fix this properly, https://reviews.llvm.org/D100388
    if (const auto *refTy = retTy->getAs<ReferenceType>()) {
      QualType pointeeTy = refTy->getPointeeType();
      if (!pointeeTy->isIncompleteType() && pointeeTy->isConstantSizeType())
        retAttrs.set(mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
                     builder.getI64IntegerAttr(
                         getMinimumObjectSize(pointeeTy).getQuantity()));

      if (getTypes().getTargetAddressSpace(pointeeTy) == 0 &&
          !codeGenOpts.NullPointerIsValid)
        retAttrs.set(mlir::LLVM::LLVMDialect::getNonNullAttrName(),
                     mlir::UnitAttr::get(&getMLIRContext()));

      if (pointeeTy->isObjectType())
        retAttrs.set(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                     builder.getI64IntegerAttr(
                         getNaturalTypeAlignment(pointeeTy).getQuantity()));
    }
  }
}

void CIRGenModule::constructFunctionArgumentAttributes() {
  assert(!cir::MissingFeatures::functionArgumentAttrs());
  // TODO(cir): This needs implementation.
}

/// Returns the canonical formal type of the given C++ method.
static CanQual<FunctionProtoType> getFormalType(const CXXMethodDecl *md) {
  return md->getType()
      ->getCanonicalTypeUnqualified()
      .getAs<FunctionProtoType>();
}

/// Adds the formal parameters in FPT to the given prefix. If any parameter in
/// FPT has pass_object_size_attrs, then we'll add parameters for those, too.
/// TODO(cir): this should be shared with LLVM codegen
static void appendParameterTypes(const CIRGenTypes &cgt,
                                 SmallVectorImpl<CanQualType> &prefix,
                                 CanQual<FunctionProtoType> fpt) {
  assert(!cir::MissingFeatures::opCallExtParameterInfo());
  // Fast path: don't touch param info if we don't need to.
  if (!fpt->hasExtParameterInfos()) {
    prefix.append(fpt->param_type_begin(), fpt->param_type_end());
    return;
  }

  cgt.getCGModule().errorNYI("appendParameterTypes: hasExtParameterInfos");
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXStructorDeclaration(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());

  llvm::SmallVector<CanQualType, 16> argTypes;
  argTypes.push_back(deriveThisType(md->getParent(), md));

  bool passParams = true;

  if (auto *cd = dyn_cast<CXXConstructorDecl>(md)) {
    // A base class inheriting constructor doesn't get forwarded arguments
    // needed to construct a virtual base (or base class thereof)
    if (cd->getInheritedConstructor())
      cgm.errorNYI(cd->getSourceRange(),
                   "arrangeCXXStructorDeclaration: inheriting constructor");
  }

  CanQual<FunctionProtoType> fpt = getFormalType(md);

  if (passParams)
    appendParameterTypes(*this, argTypes, fpt);

  // The structor signature may include implicit parameters.
  [[maybe_unused]] CIRGenCXXABI::AddedStructorArgCounts addedArgs =
      theCXXABI.buildStructorSignature(gd, argTypes);
  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoExtParamInfo());

  RequiredArgs required =
      (passParams && md->isVariadic() ? RequiredArgs(argTypes.size())
                                      : RequiredArgs::All);

  CanQualType resultType = theCXXABI.hasThisReturn(gd) ? argTypes.front()
                           : theCXXABI.hasMostDerivedReturn(gd)
                               ? astContext.VoidPtrTy
                               : astContext.VoidTy;

  assert(!theCXXABI.hasThisReturn(gd) &&
         "Please send PR with a test and remove this");

  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoExtParamInfo());
  assert(!cir::MissingFeatures::opCallFnInfoOpts());

  return arrangeCIRFunctionInfo(resultType, argTypes, fpt->getExtInfo(),
                                required);
}

/// Derives the 'this' type for CIRGen purposes, i.e. ignoring method CVR
/// qualification. Either or both of `rd` and `md` may be null. A null `rd`
/// indicates that there is no meaningful 'this' type, and a null `md` can occur
/// when calling a method pointer.
CanQualType CIRGenTypes::deriveThisType(const CXXRecordDecl *rd,
                                        const CXXMethodDecl *md) {
  CanQualType recTy;
  if (rd) {
    recTy = getASTContext().getCanonicalTagType(rd);
  } else {
    // This can happen with the MS ABI. It shouldn't need anything more than
    // setting recTy to VoidTy here, but we're flagging it for now because we
    // don't have the full handling implemented.
    cgm.errorNYI("deriveThisType: no record decl");
    recTy = getASTContext().VoidTy;
  }

  if (md)
    recTy = CanQualType::CreateUnsafe(getASTContext().getAddrSpaceQualType(
        recTy, md->getMethodQualifiers().getAddressSpace()));
  return getASTContext().getPointerType(recTy);
}

/// Arrange the CIR function layout for a value of the given function type, on
/// top of any implicit parameters already stored.
static const CIRGenFunctionInfo &
arrangeCIRFunctionInfo(CIRGenTypes &cgt, SmallVectorImpl<CanQualType> &prefix,
                       CanQual<FunctionProtoType> fpt) {
  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  RequiredArgs required =
      RequiredArgs::getFromProtoWithExtraSlots(fpt, prefix.size());
  assert(!cir::MissingFeatures::opCallExtParameterInfo());
  appendParameterTypes(cgt, prefix, fpt);
  CanQualType resultType = fpt->getReturnType().getUnqualifiedType();
  return cgt.arrangeCIRFunctionInfo(resultType, prefix, fpt->getExtInfo(),
                                    required);
}

void CIRGenFunction::emitDelegateCallArg(CallArgList &args,
                                         const VarDecl *param,
                                         SourceLocation loc) {
  // StartFunction converted the ABI-lowered parameter(s) into a local alloca.
  // We need to turn that into an r-value suitable for emitCall
  Address local = getAddrOfLocalVar(param);

  QualType type = param->getType();

  if (type->getAsCXXRecordDecl()) {
    cgm.errorNYI(param->getSourceRange(),
                 "emitDelegateCallArg: record argument");
    return;
  }

  // GetAddrOfLocalVar returns a pointer-to-pointer for references, but the
  // argument needs to be the original pointer.
  if (type->isReferenceType()) {
    args.add(
        RValue::get(builder.createLoad(getLoc(param->getSourceRange()), local)),
        type);
  } else if (getLangOpts().ObjCAutoRefCount) {
    cgm.errorNYI(param->getSourceRange(),
                 "emitDelegateCallArg: ObjCAutoRefCount");
    // For the most part, we just need to load the alloca, except that aggregate
    // r-values are actually pointers to temporaries.
  } else {
    args.add(convertTempToRValue(local, type, loc), type);
  }

  // Deactivate the cleanup for the callee-destructed param that was pushed.
  assert(!cir::MissingFeatures::thunks());
  if (type->isRecordType() &&
      type->castAsRecordDecl()->isParamDestroyedInCallee() &&
      param->needsDestruction(getContext())) {
    cgm.errorNYI(param->getSourceRange(),
                 "emitDelegateCallArg: callee-destructed param");
  }
}

static const CIRGenFunctionInfo &
arrangeFreeFunctionLikeCall(CIRGenTypes &cgt, CIRGenModule &cgm,
                            const CallArgList &args,
                            const FunctionType *fnType) {

  RequiredArgs required = RequiredArgs::All;

  if (const auto *proto = dyn_cast<FunctionProtoType>(fnType)) {
    if (proto->isVariadic())
      required = RequiredArgs::getFromProtoWithExtraSlots(proto, 0);
    if (proto->hasExtParameterInfos())
      cgm.errorNYI("call to functions with extra parameter info");
  }

  SmallVector<CanQualType, 16> argTypes;
  argTypes.reserve(args.size());
  for (const CallArg &arg : args)
    argTypes.push_back(cgt.getASTContext().getCanonicalParamType(arg.ty));
  CanQualType retType = fnType->getReturnType()->getCanonicalTypeUnqualified();

  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return cgt.arrangeCIRFunctionInfo(retType, argTypes, fnType->getExtInfo(),
                                    required);
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// extraPrefixArgs is the number of ABI-specific args passed after the `this`
/// parameter.
/// passProtoArgs indicates whether `args` has args for the parameters in the
/// given CXXConstructorDecl.
const CIRGenFunctionInfo &CIRGenTypes::arrangeCXXConstructorCall(
    const CallArgList &args, const CXXConstructorDecl *d, CXXCtorType ctorKind,
    unsigned extraPrefixArgs, unsigned extraSuffixArgs, bool passProtoArgs) {

  // FIXME: Kill copy.
  llvm::SmallVector<CanQualType, 16> argTypes;
  for (const auto &arg : args)
    argTypes.push_back(astContext.getCanonicalParamType(arg.ty));

  // +1 for implicit this, which should always be args[0]
  unsigned totalPrefixArgs = 1 + extraPrefixArgs;

  CanQual<FunctionProtoType> fpt = getFormalType(d);
  RequiredArgs required = passProtoArgs
                              ? RequiredArgs::getFromProtoWithExtraSlots(
                                    fpt, totalPrefixArgs + extraSuffixArgs)
                              : RequiredArgs::All;

  GlobalDecl gd(d, ctorKind);
  if (theCXXABI.hasThisReturn(gd))
    cgm.errorNYI(d->getSourceRange(),
                 "arrangeCXXConstructorCall: hasThisReturn");
  if (theCXXABI.hasMostDerivedReturn(gd))
    cgm.errorNYI(d->getSourceRange(),
                 "arrangeCXXConstructorCall: hasMostDerivedReturn");
  CanQualType resultType = astContext.VoidTy;

  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoExtParamInfo());

  return arrangeCIRFunctionInfo(resultType, argTypes, fpt->getExtInfo(),
                                required);
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// numPrefixArgs is the number of the ABI-specific prefix arguments we have. It
/// does not count `this`.
const CIRGenFunctionInfo &CIRGenTypes::arrangeCXXMethodCall(
    const CallArgList &args, const FunctionProtoType *proto,
    RequiredArgs required, unsigned numPrefixArgs) {
  assert(!cir::MissingFeatures::opCallExtParameterInfo());
  assert(numPrefixArgs + 1 <= args.size() &&
         "Emitting a call with less args than the required prefix?");

  // FIXME: Kill copy.
  llvm::SmallVector<CanQualType, 16> argTypes;
  for (const CallArg &arg : args)
    argTypes.push_back(astContext.getCanonicalParamType(arg.ty));

  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return arrangeCIRFunctionInfo(
      proto->getReturnType()->getCanonicalTypeUnqualified(), argTypes,
      proto->getExtInfo(), required);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionCall(const CallArgList &args,
                                     const FunctionType *fnType) {
  return arrangeFreeFunctionLikeCall(*this, cgm, args, fnType);
}

/// Arrange the argument and result information for a declaration or definition
/// of the given C++ non-static member function. The member function must be an
/// ordinary function, i.e. not a constructor or destructor.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXMethodDeclaration(const CXXMethodDecl *md) {
  assert(!isa<CXXConstructorDecl>(md) && "wrong method for constructors!");
  assert(!isa<CXXDestructorDecl>(md) && "wrong method for destructors!");

  auto prototype =
      md->getType()->getCanonicalTypeUnqualified().getAs<FunctionProtoType>();
  assert(!cir::MissingFeatures::cudaSupport());

  if (md->isInstance()) {
    // The abstract case is perfectly fine.
    auto *thisType = theCXXABI.getThisArgumentTypeForMethod(md);
    return arrangeCXXMethodType(thisType, prototype.getTypePtr(), md);
  }

  return arrangeFreeFunctionType(prototype);
}

/// Arrange the argument and result information for a call to an unknown C++
/// non-static member function of the given abstract type. (A null RD means we
/// don't have any meaningful "this" argument type, so fall back to a generic
/// pointer type). The member fucntion must be an ordinary function, i.e. not a
/// constructor or destructor.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXMethodType(const CXXRecordDecl *rd,
                                  const FunctionProtoType *fpt,
                                  const CXXMethodDecl *md) {
  llvm::SmallVector<CanQualType, 16> argTypes;

  // Add the 'this' pointer.
  argTypes.push_back(deriveThisType(rd, md));

  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return ::arrangeCIRFunctionInfo(
      *this, argTypes,
      fpt->getCanonicalTypeUnqualified().getAs<FunctionProtoType>());
}

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeFunctionDeclaration(const FunctionDecl *fd) {
  if (const auto *md = dyn_cast<CXXMethodDecl>(fd))
    if (md->isInstance())
      return arrangeCXXMethodDeclaration(md);

  CanQualType funcTy = fd->getType()->getCanonicalTypeUnqualified();

  assert(isa<FunctionType>(funcTy));
  // TODO: setCUDAKernelCallingConvention
  assert(!cir::MissingFeatures::cudaSupport());

  // Handle C89/gnu89 no-prototype functions (FunctionNoProtoType).
  //
  // In C89, a function declared without a prototype does not type-check
  // arguments and may legally be called with additional arguments.
  // If we model such functions as non-variadic, the first observed call-site
  // would freeze the function signature to that arity, causing later calls
  // with different argument counts to fail CIR verification.
  //
  // When the target ABI permits variadic no-proto calls, model the function
  // as variadic with zero fixed parameters to preserve C semantics and keep
  // the CIR function type stable across call sites.
  if (CanQual<FunctionNoProtoType> noProto =
          funcTy.getAs<FunctionNoProtoType>()) {
    assert(!cir::MissingFeatures::opCallCIRGenFuncInfoExtParamInfo());
    assert(!cir::MissingFeatures::opCallFnInfoOpts());

    if (cgm.getTargetCIRGenInfo().isNoProtoCallVariadic(noProto.getTypePtr())) {
      return arrangeCIRFunctionInfo(noProto->getReturnType(), {},
                                    noProto->getExtInfo(), RequiredArgs(0));
    }
    return arrangeCIRFunctionInfo(noProto->getReturnType(), {},
                                  noProto->getExtInfo(), RequiredArgs::All);
  }

  return arrangeFreeFunctionType(funcTy.castAs<FunctionProtoType>());
}

static cir::CIRCallOpInterface emitCallLikeOp(
    CIRGenFunction &cgf, mlir::Location callLoc, cir::FuncType indirectFuncTy,
    mlir::Value indirectFuncVal, cir::FuncOp directFuncOp,
    const SmallVectorImpl<mlir::Value> &cirCallArgs, bool isInvoke,
    const mlir::NamedAttrList &attrs, const mlir::NamedAttrList &retAttrs) {
  CIRGenBuilderTy &builder = cgf.getBuilder();

  assert(!cir::MissingFeatures::opCallSurroundingTry());

  assert(builder.getInsertionBlock() && "expected valid basic block");

  cir::CallOp op;
  if (indirectFuncTy) {
    // TODO(cir): Set calling convention for indirect calls.
    assert(!cir::MissingFeatures::opCallCallConv());
    op = builder.createIndirectCallOp(callLoc, indirectFuncVal, indirectFuncTy,
                                      cirCallArgs, attrs, retAttrs);
  } else {
    op = builder.createCallOp(callLoc, directFuncOp, cirCallArgs, attrs,
                              retAttrs);
  }

  return op;
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionProtoType> fpt) {
  SmallVector<CanQualType, 16> argTypes;
  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return ::arrangeCIRFunctionInfo(*this, argTypes, fpt);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionNoProtoType> fnpt) {
  CanQualType resultType = fnpt->getReturnType().getUnqualifiedType();
  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return arrangeCIRFunctionInfo(resultType, {}, fnpt->getExtInfo(),
                                RequiredArgs(0));
}

RValue CIRGenFunction::emitCall(const CIRGenFunctionInfo &funcInfo,
                                const CIRGenCallee &callee,
                                ReturnValueSlot returnValue,
                                const CallArgList &args,
                                cir::CIRCallOpInterface *callOp,
                                mlir::Location loc) {
  QualType retTy = funcInfo.getReturnType();
  cir::FuncType cirFuncTy = getTypes().getFunctionType(funcInfo);

  SmallVector<mlir::Value, 16> cirCallArgs(args.size());

  assert(!cir::MissingFeatures::emitLifetimeMarkers());

  // Translate all of the arguments as necessary to match the CIR lowering.
  for (auto [argNo, arg, canQualArgType] :
       llvm::enumerate(args, funcInfo.argTypes())) {

    // Insert a padding argument to ensure proper alignment.
    assert(!cir::MissingFeatures::opCallPaddingArgs());

    mlir::Type argType = convertType(canQualArgType);
    if (!mlir::isa<cir::RecordType>(argType) &&
        !mlir::isa<cir::ComplexType>(argType)) {
      mlir::Value v;
      if (arg.isAggregate())
        cgm.errorNYI(loc, "emitCall: aggregate call argument");
      v = arg.getKnownRValue().getValue();

      // We might have to widen integers, but we should never truncate.
      if (argType != v.getType() && mlir::isa<cir::IntType>(v.getType()))
        cgm.errorNYI(loc, "emitCall: widening integer call argument");

      // If the argument doesn't match, perform a bitcast to coerce it. This
      // can happen due to trivial type mismatches.
      // TODO(cir): When getFunctionType is added, assert that this isn't
      // needed.
      assert(!cir::MissingFeatures::opCallBitcastArg());
      cirCallArgs[argNo] = v;
    } else {
      Address src = Address::invalid();
      if (!arg.isAggregate()) {
        src = createMemTemp(arg.ty, loc, "coerce");
        arg.copyInto(*this, src, loc);
      } else {
        src = arg.hasLValue() ? arg.getKnownLValue().getAddress()
                              : arg.getKnownRValue().getAggregateAddress();
      }

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      mlir::Type srcTy = src.getElementType();
      // FIXME(cir): get proper location for each argument.
      mlir::Location argLoc = loc;

      // If the source type is smaller than the destination type of the
      // coerce-to logic, copy the source value into a temp alloca the size
      // of the destination type to allow loading all of it. The bits past
      // the source value are left undef.
      // FIXME(cir): add data layout info and compare sizes instead of
      // matching the types.
      //
      // uint64_t SrcSize = CGM.getDataLayout().getTypeAllocSize(SrcTy);
      // uint64_t DstSize = CGM.getDataLayout().getTypeAllocSize(STy);
      // if (SrcSize < DstSize) {
      assert(!cir::MissingFeatures::dataLayoutTypeAllocSize());
      if (srcTy != argType) {
        cgm.errorNYI(loc, "emitCall: source type does not match argument type");
      } else {
        // FIXME(cir): this currently only runs when the types are exactly the
        // same, but should be when alloc sizes are the same, fix this as soon
        // as datalayout gets introduced.
        assert(!cir::MissingFeatures::dataLayoutTypeAllocSize());
      }

      // assert(NumCIRArgs == STy.getMembers().size());
      // In LLVMGen: Still only pass the struct without any gaps but mark it
      // as such somehow.
      //
      // In CIRGen: Emit a load from the "whole" struct,
      // which shall be broken later by some lowering step into multiple
      // loads.
      assert(!cir::MissingFeatures::lowerAggregateLoadStore());
      cirCallArgs[argNo] = builder.createLoad(argLoc, src);
    }
  }

  const CIRGenCallee &concreteCallee = callee.prepareConcreteCallee(*this);
  mlir::Operation *calleePtr = concreteCallee.getFunctionPointer();

  assert(!cir::MissingFeatures::opCallInAlloca());

  mlir::NamedAttrList attrs;
  mlir::NamedAttrList retAttrs;
  StringRef funcName;
  if (auto calleeFuncOp = dyn_cast<cir::FuncOp>(calleePtr))
    funcName = calleeFuncOp.getName();

  assert(!cir::MissingFeatures::opCallCallConv());
  assert(!cir::MissingFeatures::opCallAttrs());
  cir::CallingConv callingConv;
  cir::SideEffect sideEffect;
  cgm.constructAttributeList(funcName, funcInfo, callee.getAbstractInfo(),
                             attrs, retAttrs, callingConv, sideEffect,
                             /*attrOnCallSite=*/true, /*isThunk=*/false);

  cir::FuncType indirectFuncTy;
  mlir::Value indirectFuncVal;
  cir::FuncOp directFuncOp;
  if (auto fnOp = dyn_cast<cir::FuncOp>(calleePtr)) {
    directFuncOp = fnOp;
  } else if (auto getGlobalOp = mlir::dyn_cast<cir::GetGlobalOp>(calleePtr)) {
    // FIXME(cir): This peephole optimization avoids indirect calls for
    // builtins. This should be fixed in the builtin declaration instead by
    // not emitting an unecessary get_global in the first place.
    // However, this is also used for no-prototype functions.
    mlir::Operation *globalOp = cgm.getGlobalValue(getGlobalOp.getName());
    assert(globalOp && "undefined global function");
    directFuncOp = mlir::cast<cir::FuncOp>(globalOp);
  } else {
    [[maybe_unused]] mlir::ValueTypeRange<mlir::ResultRange> resultTypes =
        calleePtr->getResultTypes();
    [[maybe_unused]] auto funcPtrTy =
        mlir::dyn_cast<cir::PointerType>(resultTypes.front());
    assert(funcPtrTy && mlir::isa<cir::FuncType>(funcPtrTy.getPointee()) &&
           "expected pointer to function");

    indirectFuncTy = cirFuncTy;
    indirectFuncVal = calleePtr->getResult(0);
  }

  assert(!cir::MissingFeatures::msvcCXXPersonality());
  assert(!cir::MissingFeatures::functionUsesSEHTry());
  assert(!cir::MissingFeatures::nothrowAttr());

  bool cannotThrow = attrs.getNamed("nothrow").has_value();
  bool isInvoke = !cannotThrow && isCatchOrCleanupRequired();

  mlir::Location callLoc = loc;
  cir::CIRCallOpInterface theCall =
      emitCallLikeOp(*this, loc, indirectFuncTy, indirectFuncVal, directFuncOp,
                     cirCallArgs, isInvoke, attrs, retAttrs);

  if (callOp)
    *callOp = theCall;

  assert(!cir::MissingFeatures::opCallMustTail());
  assert(!cir::MissingFeatures::opCallReturn());

  mlir::Type retCIRTy = convertType(retTy);
  if (isa<cir::VoidType>(retCIRTy))
    return getUndefRValue(retTy);
  switch (getEvaluationKind(retTy)) {
  case cir::TEK_Aggregate: {
    Address destPtr = returnValue.getValue();

    if (!destPtr.isValid())
      destPtr = createMemTemp(retTy, callLoc, getCounterAggTmpAsString());

    mlir::ResultRange results = theCall->getOpResults();
    assert(results.size() <= 1 && "multiple returns from a call");

    SourceLocRAIIObject loc{*this, callLoc};
    emitAggregateStore(results[0], destPtr);
    return RValue::getAggregate(destPtr);
  }
  case cir::TEK_Scalar: {
    mlir::ResultRange results = theCall->getOpResults();
    assert(results.size() == 1 && "unexpected number of returns");

    // If the argument doesn't match, perform a bitcast to coerce it. This
    // can happen due to trivial type mismatches.
    if (results[0].getType() != retCIRTy)
      cgm.errorNYI(loc, "bitcast on function return value");

    mlir::Region *region = builder.getBlock()->getParent();
    if (region != theCall->getParentRegion())
      cgm.errorNYI(loc, "function calls with cleanup");

    return RValue::get(results[0]);
  }
  case cir::TEK_Complex: {
    mlir::ResultRange results = theCall->getOpResults();
    assert(!results.empty() &&
           "Expected at least one result for complex rvalue");
    return RValue::getComplex(results[0]);
  }
  }
  llvm_unreachable("Invalid evaluation kind");
}

void CallArg::copyInto(CIRGenFunction &cgf, Address addr,
                       mlir::Location loc) const {
  LValue dst = cgf.makeAddrLValue(addr, ty);
  if (!hasLV && rv.isScalar())
    cgf.cgm.errorNYI(loc, "copyInto scalar value");
  else if (!hasLV && rv.isComplex())
    cgf.emitStoreOfComplex(loc, rv.getComplexValue(), dst, /*isInit=*/true);
  else
    cgf.cgm.errorNYI(loc, "copyInto hasLV");
  isUsed = true;
}

mlir::Value CIRGenFunction::emitRuntimeCall(mlir::Location loc,
                                            cir::FuncOp callee,
                                            ArrayRef<mlir::Value> args) {
  // TODO(cir): set the calling convention to this runtime call.
  assert(!cir::MissingFeatures::opFuncCallingConv());

  cir::CallOp call = builder.createCallOp(loc, callee, args);
  assert(call->getNumResults() <= 1 &&
         "runtime functions have at most 1 result");

  if (call->getNumResults() == 0)
    return nullptr;

  return call->getResult(0);
}

void CIRGenFunction::emitCallArg(CallArgList &args, const clang::Expr *e,
                                 clang::QualType argType) {
  assert(argType->isReferenceType() == e->isGLValue() &&
         "reference binding to unmaterialized r-value!");

  if (e->isGLValue()) {
    assert(e->getObjectKind() == OK_Ordinary);
    return args.add(emitReferenceBindingToExpr(e), argType);
  }

  bool hasAggregateEvalKind = hasAggregateEvaluationKind(argType);

  // In the Microsoft C++ ABI, aggregate arguments are destructed by the
  // callee. However, we still have to push an EH-only cleanup in case we
  // unwind before we make it to the call.
  if (argType->isRecordType() &&
      argType->castAsRecordDecl()->isParamDestroyedInCallee()) {
    assert(!cir::MissingFeatures::msabi());
    cgm.errorNYI(e->getSourceRange(), "emitCallArg: msabi is NYI");
  }

  if (hasAggregateEvalKind && isa<ImplicitCastExpr>(e) &&
      cast<CastExpr>(e)->getCastKind() == CK_LValueToRValue) {
    LValue lv = emitLValue(cast<CastExpr>(e)->getSubExpr());
    assert(lv.isSimple());
    args.addUncopiedAggregate(lv, argType);
    return;
  }

  args.add(emitAnyExprToTemp(e), argType);
}

QualType CIRGenFunction::getVarArgType(const Expr *arg) {
  // System headers on Windows define NULL to 0 instead of 0LL on Win64. MSVC
  // implicitly widens null pointer constants that are arguments to varargs
  // functions to pointer-sized ints.
  if (!getTarget().getTriple().isOSWindows())
    return arg->getType();

  assert(!cir::MissingFeatures::msabi());
  cgm.errorNYI(arg->getSourceRange(), "getVarArgType: NYI for Windows target");
  return arg->getType();
}

/// Similar to emitAnyExpr(), however, the result will always be accessible
/// even if no aggregate location is provided.
RValue CIRGenFunction::emitAnyExprToTemp(const Expr *e) {
  AggValueSlot aggSlot = AggValueSlot::ignored();

  if (hasAggregateEvaluationKind(e->getType()))
    aggSlot = createAggTemp(e->getType(), getLoc(e->getSourceRange()),
                            getCounterAggTmpAsString());

  return emitAnyExpr(e, aggSlot);
}

void CIRGenFunction::emitCallArgs(
    CallArgList &args, PrototypeWrapper prototype,
    llvm::iterator_range<clang::CallExpr::const_arg_iterator> argRange,
    AbstractCallee callee, unsigned paramsToSkip) {
  llvm::SmallVector<QualType, 16> argTypes;

  assert(!cir::MissingFeatures::opCallCallConv());

  // First, if a prototype was provided, use those argument types.
  bool isVariadic = false;
  if (prototype.p) {
    assert(!cir::MissingFeatures::opCallObjCMethod());

    const auto *fpt = cast<const FunctionProtoType *>(prototype.p);
    isVariadic = fpt->isVariadic();
    assert(!cir::MissingFeatures::opCallCallConv());
    argTypes.assign(fpt->param_type_begin() + paramsToSkip,
                    fpt->param_type_end());
  } else {
    // No prototype (e.g. implicit/old-style): allow extra args.
    // Treat as variadic so we use promoted vararg types for all arguments.
    isVariadic = true;
  }

  // If we still have any arguments, emit them using the type of the argument.
  for (const clang::Expr *a : llvm::drop_begin(argRange, argTypes.size()))
    argTypes.push_back(isVariadic ? getVarArgType(a) : a->getType());
  assert(argTypes.size() == (size_t)(argRange.end() - argRange.begin()));

  // We must evaluate arguments from right to left in the MS C++ ABI, because
  // arguments are destroyed left to right in the callee. As a special case,
  // there are certain language constructs taht require left-to-right
  // evaluation, and in those cases we consider the evaluation order
  // requirement to trump the "destruction order is reverse construction
  // order" guarantee.
  auto leftToRight = true;
  assert(!cir::MissingFeatures::msabi());

  auto maybeEmitImplicitObjectSize = [&](size_t i, const Expr *arg,
                                         RValue emittedArg) {
    if (!callee.hasFunctionDecl() || i >= callee.getNumParams())
      return;
    auto *ps = callee.getParamDecl(i)->getAttr<PassObjectSizeAttr>();
    if (!ps)
      return;

    assert(!cir::MissingFeatures::opCallImplicitObjectSizeArgs());
    cgm.errorNYI("emit implicit object size for call arg");
  };

  // Evaluate each argument in the appropriate order.
  size_t callArgsStart = args.size();
  for (size_t i = 0; i != argTypes.size(); ++i) {
    size_t idx = leftToRight ? i : argTypes.size() - i - 1;
    CallExpr::const_arg_iterator currentArg = argRange.begin() + idx;
    size_t initialArgSize = args.size();

    emitCallArg(args, *currentArg, argTypes[idx]);

    // In particular, we depend on it being the last arg in Args, and the
    // objectsize bits depend on there only being one arg if !LeftToRight.
    assert(initialArgSize + 1 == args.size() &&
           "The code below depends on only adding one arg per emitCallArg");
    (void)initialArgSize;

    // Since pointer argument are never emitted as LValue, it is safe to emit
    // non-null argument check for r-value only.
    if (!args.back().hasLValue()) {
      RValue rvArg = args.back().getKnownRValue();
      assert(!cir::MissingFeatures::sanitizers());
      maybeEmitImplicitObjectSize(idx, *currentArg, rvArg);
    }

    if (!leftToRight)
      std::reverse(args.begin() + callArgsStart, args.end());
  }
}
