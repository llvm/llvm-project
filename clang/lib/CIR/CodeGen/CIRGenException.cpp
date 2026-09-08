//===--- CIRGenException.cpp - Emit CIR Code for C++ exceptions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ exception related code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Location.h"

#include "clang/Basic/DiagnosticSema.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;
using namespace clang::CIRGen;

const EHPersonality EHPersonality::GNU_C = {"__gcc_personality_v0", nullptr};
const EHPersonality EHPersonality::GNU_C_SJLJ = {"__gcc_personality_sj0",
                                                 nullptr};
const EHPersonality EHPersonality::GNU_C_SEH = {"__gcc_personality_seh0",
                                                nullptr};
const EHPersonality EHPersonality::NeXT_ObjC = {"__objc_personality_v0",
                                                nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus = {"__gxx_personality_v0",
                                                    nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus_SJLJ = {
    "__gxx_personality_sj0", nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus_SEH = {
    "__gxx_personality_seh0", nullptr};
const EHPersonality EHPersonality::GNU_ObjC = {"__gnu_objc_personality_v0",
                                               "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjC_SJLJ = {
    "__gnu_objc_personality_sj0", "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjC_SEH = {
    "__gnu_objc_personality_seh0", "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjCXX = {
    "__gnustep_objcxx_personality_v0", nullptr};
const EHPersonality EHPersonality::GNUstep_ObjC = {
    "__gnustep_objc_personality_v0", nullptr};
const EHPersonality EHPersonality::MSVC_except_handler = {"_except_handler3",
                                                          nullptr};
const EHPersonality EHPersonality::MSVC_C_specific_handler = {
    "__C_specific_handler", nullptr};
const EHPersonality EHPersonality::MSVC_CxxFrameHandler3 = {
    "__CxxFrameHandler3", nullptr};
const EHPersonality EHPersonality::GNU_Wasm_CPlusPlus = {
    "__gxx_wasm_personality_v0", nullptr};
const EHPersonality EHPersonality::XL_CPlusPlus = {"__xlcxx_personality_v1",
                                                   nullptr};
const EHPersonality EHPersonality::ZOS_CPlusPlus = {"__zos_cxx_personality_v2",
                                                    nullptr};

static const EHPersonality &getCPersonality(const TargetInfo &target,
                                            const CodeGenOptions &cgOpts) {
  const llvm::Triple &triple = target.getTriple();
  if (triple.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (cgOpts.hasSjLjExceptions())
    return EHPersonality::GNU_C_SJLJ;
  if (cgOpts.hasDWARFExceptions())
    return EHPersonality::GNU_C;
  if (cgOpts.hasSEHExceptions())
    return EHPersonality::GNU_C_SEH;
  return EHPersonality::GNU_C;
}

static const EHPersonality &getObjCPersonality(const TargetInfo &target,
                                               const LangOptions &langOpts,
                                               const CodeGenOptions &cgOpts) {
  const llvm::Triple &triple = target.getTriple();
  if (triple.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;

  switch (langOpts.ObjCRuntime.getKind()) {
  case ObjCRuntime::FragileMacOSX:
    return getCPersonality(target, cgOpts);
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return EHPersonality::NeXT_ObjC;
  case ObjCRuntime::GNUstep:
    if (langOpts.ObjCRuntime.getVersion() >= VersionTuple(1, 7))
      return EHPersonality::GNUstep_ObjC;
    [[fallthrough]];
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    if (cgOpts.hasSjLjExceptions())
      return EHPersonality::GNU_ObjC_SJLJ;
    if (cgOpts.hasSEHExceptions())
      return EHPersonality::GNU_ObjC_SEH;
    return EHPersonality::GNU_ObjC;
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getCXXPersonality(const TargetInfo &target,
                                              const CodeGenOptions &cgOpts) {
  const llvm::Triple &triple = target.getTriple();
  if (triple.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (triple.isOSAIX())
    return EHPersonality::XL_CPlusPlus;
  if (cgOpts.hasSjLjExceptions())
    return EHPersonality::GNU_CPlusPlus_SJLJ;
  if (cgOpts.hasDWARFExceptions())
    return EHPersonality::GNU_CPlusPlus;
  if (cgOpts.hasSEHExceptions())
    return EHPersonality::GNU_CPlusPlus_SEH;
  if (cgOpts.hasWasmExceptions())
    return EHPersonality::GNU_Wasm_CPlusPlus;
  return EHPersonality::GNU_CPlusPlus;
}

/// Determines the personality function to use when both C++
/// and Objective-C exceptions are being caught.
static const EHPersonality &getObjCXXPersonality(const TargetInfo &target,
                                                 const LangOptions &langOpts,
                                                 const CodeGenOptions &cgOpts) {
  if (target.getTriple().isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;

  switch (langOpts.ObjCRuntime.getKind()) {
  // In the fragile ABI, just use C++ exception handling and hope
  // they're not doing crazy exception mixing.
  case ObjCRuntime::FragileMacOSX:
    return getCXXPersonality(target, cgOpts);

  // The ObjC personality defers to the C++ personality for non-ObjC
  // handlers.  Unlike the C++ case, we use the same personality
  // function on targets using (backend-driven) SJLJ EH.
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return getObjCPersonality(target, langOpts, cgOpts);

  case ObjCRuntime::GNUstep:
    return EHPersonality::GNU_ObjCXX;

  // The GCC runtime's personality function inherently doesn't support
  // mixed EH.  Use the ObjC personality just to avoid returning null.
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    return getObjCPersonality(target, langOpts, cgOpts);
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getSEHPersonalityMSVC(const llvm::Triple &triple) {
  return triple.getArch() == llvm::Triple::x86
             ? EHPersonality::MSVC_except_handler
             : EHPersonality::MSVC_C_specific_handler;
}

const EHPersonality &EHPersonality::get(CIRGenModule &cgm,
                                        const FunctionDecl *fd) {
  const llvm::Triple &triple = cgm.getTarget().getTriple();
  const LangOptions &langOpts = cgm.getLangOpts();
  const CodeGenOptions &cgOpts = cgm.getCodeGenOpts();
  const TargetInfo &target = cgm.getTarget();

  // Functions using SEH get an SEH personality.
  if (fd && fd->usesSEHTry())
    return getSEHPersonalityMSVC(triple);

  if (langOpts.ObjC) {
    return langOpts.CPlusPlus ? getObjCXXPersonality(target, langOpts, cgOpts)
                              : getObjCPersonality(target, langOpts, cgOpts);
  }
  return langOpts.CPlusPlus ? getCXXPersonality(target, cgOpts)
                            : getCPersonality(target, cgOpts);
}

const EHPersonality &EHPersonality::get(CIRGenFunction &cgf) {
  const auto *fg = cgf.curCodeDecl;
  // For outlined finallys and filters, use the SEH personality in case they
  // contain more SEH. This mostly only affects finallys. Filters could
  // hypothetically use gnu statement expressions to sneak in nested SEH.
  fg = fg ? fg : cgf.curSEHParent.getDecl();
  return get(cgf.cgm, dyn_cast_or_null<FunctionDecl>(fg));
}

static llvm::StringRef getPersonalityFn(CIRGenModule &cgm,
                                        const EHPersonality &personality) {
  // Create the personality function type: i32 (...)
  mlir::Type i32Ty = cgm.getBuilder().getSInt32Ty();
  auto funcTy = cir::FuncType::get({}, i32Ty, /*isVarArg=*/true);

  cir::FuncOp personalityFn = cgm.createRuntimeFunction(
      funcTy, personality.personalityFn, {}, /*isLocal=*/true);

  return personalityFn.getSymName();
}

void CIRGenFunction::emitCXXThrowExpr(const CXXThrowExpr *e) {
  // Herbception `throw throws`: route the error value to the nearest active
  // `try { } catch throws` handler, or take the {T, i1} error return path of
  // the enclosing throws function. This is not a traditional throw.
  if (e->isHerbception()) {
    mlir::Location loc = getLoc(e->getSourceRange());

    if (!herbceptionCatchScopes.empty()) {
      const HerbceptionCatchScope &scope = herbceptionCatchScopes.back();

      RunCleanupsScope throwScope(*this);
      if (const Expr *subExpr = e->getSubExpr())
        emitAnyExprToMem(subExpr, scope.errorSlot, Qualifiers(),
                         /*isInitializer=*/false);
      throwScope.forceCleanup();

      cir::GotoOp::create(builder, loc, scope.handlerLabel);
      // A throw marks the end of the block; create a new one for codegen
      // after the throw statement.
      builder.createBlock(builder.getBlock()->getParent());
      return;
    }

    // Error return path: the enclosing function must carry a herbception
    // spec; store {error, true} and return.
    if (!curFnInfo || !curFnInfo->hasThrowsReturn()) {
      cgm.getDiags().Report(e->getExprLoc(),
                            diag::err_throw_throws_no_catch_handler);
      return;
    }
    const Expr *subExpr = e->getSubExpr();
    if (!subExpr) {
      // Bare `throw throws` (rethrow) in a throws function: the error value is
      // already in the return slot from a previous catch. Wrap it with the
      // discriminant set to true and return.
      if (!fnRetAlloca) {
        // Void throws function: no return slot. The error type is the payload
        // type; create a temp, wrap with disc=true and return.
        auto fn = cast<cir::FuncOp>(curFn);
        auto shapedTy =
            cast<cir::RecordType>(fn.getFunctionType().getReturnType());
        mlir::Type errTy = shapedTy.getMembers()[0];
        CharUnits errAlign =
            CharUnits::fromQuantity(cgm.getDataLayout().getABITypeAlign(errTy));
        Address errSlot = createTempAlloca(errTy, errAlign, loc, "herb.error");
        mlir::Value payload = builder.createLoad(loc, errSlot);
        mlir::Value wrapped = wrapHerbceptionReturnValue(loc, payload,
                                                         /*disc=*/true);
        cir::ReturnOp::create(builder, loc, wrapped);
        builder.createBlock(builder.getBlock()->getParent());
        return;
      }
      mlir::Type retTy =
          cast<cir::FuncOp>(curFn).getFunctionType().getReturnType();
      mlir::Value value =
          cir::LoadOp::create(builder, loc, retTy, *fnRetAlloca);
      value = wrapHerbceptionReturnValue(loc, value, /*disc=*/true);
      cir::ReturnOp::create(builder, loc, value);
      builder.createBlock(builder.getBlock()->getParent());
      return;
    }

    auto fn = cast<cir::FuncOp>(curFn);
    auto shapedTy = cast<cir::RecordType>(fn.getFunctionType().getReturnType());
    CharUnits align =
        CharUnits::fromQuantity(cgm.getDataLayout().getABITypeAlign(shapedTy));
    Address tmp = createTempAlloca(shapedTy, align, loc, "__herb.ret");
    llvm::SmallVector<mlir::Type> members(shapedTy.getMembers().begin(),
                                          shapedTy.getMembers().end());
    {
      RunCleanupsScope throwScope(*this);
      mlir::Value payloadPtr = builder.createGetMember(
          loc, builder.getPointerTo(members[0]), tmp.getBasePointer(), "", 0);
      // The fabricated value has the named std::error/E record type while
      // the payload slot may be the anonymous {void *, size_t} shape; coerce
      // through a pointer bitcast (same layout, same size).
      mlir::Type errTy = convertTypeForMem(subExpr->getType());
      mlir::Value errPtr = payloadPtr;
      if (errTy != members[0])
        errPtr = builder.createBitcast(payloadPtr, builder.getPointerTo(errTy));
      emitAnyExprToMem(subExpr, Address(errPtr, errTy, align), Qualifiers(),
                       /*isInitializer=*/false);
      mlir::Value discPtr = builder.createGetMember(
          loc, builder.getPointerTo(members[1]), tmp.getBasePointer(), "", 1);
      builder.createStore(loc, builder.getBool(true, loc),
                          Address(discPtr, members[1], align));
      throwScope.forceCleanup();
    }
    mlir::Value result = builder.createLoad(loc, tmp);
    cir::ReturnOp::create(builder, loc, result);
    // A throw marks the end of the block; create a new one for codegen after
    // the throw statement (the ops there are unreachable).
    builder.createBlock(builder.getBlock()->getParent());
    return;
  }

  const llvm::Triple &triple = getTarget().getTriple();
  if (cgm.getLangOpts().OpenMPIsTargetDevice &&
      (triple.isNVPTX() || triple.isAMDGCN())) {
    cgm.errorNYI("emitCXXThrowExpr OpenMP with NVPTX or AMDGCN Triples");
    return;
  }

  if (const Expr *subExpr = e->getSubExpr()) {
    QualType throwType = subExpr->getType();
    if (throwType->isObjCObjectPointerType()) {
      cgm.errorNYI("emitCXXThrowExpr ObjCObjectPointerType");
      return;
    }

    cgm.getCXXABI().emitThrow(*this, e);
    return;
  }

  cgm.getCXXABI().emitRethrow(*this, /*isNoReturn=*/true);
}

void CIRGenFunction::emitAnyExprToExn(const Expr *e, Address addr) {
  // Make sure the exception object is cleaned up if there's an
  // exception during initialization.
  assert(!cir::MissingFeatures::ehCleanupScope());

  // __cxa_allocate_exception returns a void*;  we need to cast this
  // to the appropriate type for the object.
  mlir::Type ty = convertTypeForMem(e->getType());
  Address typedAddr = addr.withElementType(builder, ty);

  // From LLVM's codegen:
  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  emitAnyExprToMem(e, typedAddr, e->getType().getQualifiers(),
                   /*isInitializer=*/true);

  // Deactivate the cleanup block.
  assert(!cir::MissingFeatures::ehCleanupScope());
}

void CIRGenFunction::addCatchHandlerAttr(
    const CXXCatchStmt *catchStmt, SmallVector<mlir::Attribute> &handlerAttrs) {
  mlir::Location catchLoc = getLoc(catchStmt->getBeginLoc());

  if (catchStmt->getExceptionDecl()) {
    // FIXME: Dropping the reference type on the type into makes it
    // impossible to correctly implement catch-by-reference
    // semantics for pointers.  Unfortunately, this is what all
    // existing compilers do, and it's not clear that the standard
    // personality routine is capable of doing this right.  See C++ DR 388:
    //   http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#388
    Qualifiers caughtTypeQuals;
    QualType caughtType = cgm.getASTContext().getUnqualifiedArrayType(
        catchStmt->getCaughtType().getNonReferenceType(), caughtTypeQuals);
    if (caughtType->isObjCObjectPointerType()) {
      cgm.errorNYI("addCatchHandlerAttr: caughtType ObjCObjectPointerType");
      return;
    }

    CatchTypeInfo typeInfo = cgm.getCXXABI().getAddrOfCXXCatchHandlerType(
        catchLoc, caughtType, catchStmt->getCaughtType());
    handlerAttrs.push_back(typeInfo.rtti);
  } else {
    // No exception decl indicates '...', a catch-all.
    handlerAttrs.push_back(cir::CatchAllAttr::get(&getMLIRContext()));
  }
}

namespace {
struct CallEndCatch final : EHScopeStack::Cleanup {
  CallEndCatch(mlir::Value catchToken) : catchToken(catchToken) {}
  mlir::Value catchToken;

  void emit(CIRGenFunction &cgf, Flags flags) override {
    cir::EndCatchOp::create(cgf.getBuilder(), *cgf.currSrcLoc, catchToken);
    cir::YieldOp::create(cgf.getBuilder(), *cgf.currSrcLoc);
  }
};
} // namespace

static mlir::Value callBeginCatch(CIRGenFunction &cgf, mlir::Value ehToken,
                                  mlir::Type exnPtrTy) {
  auto catchTokenTy = cir::CatchTokenType::get(cgf.getBuilder().getContext());
  auto beginCatch = cir::BeginCatchOp::create(cgf.getBuilder(),
                                              cgf.getBuilder().getUnknownLoc(),
                                              catchTokenTy, exnPtrTy, ehToken);

  cgf.ehStack.pushCleanup<CallEndCatch>(NormalAndEHCleanup,
                                        beginCatch.getCatchToken());

  return beginCatch.getExnPtr();
}

/// Get or create the catch-init copy thunk for \p catchParam.
///
/// The copy thunk has signature `void(T*, T*)` (where `T` is the catch
/// parameter type) and contains the normal aggregate emission of the catch
/// parameter's init expression.
///
/// The thunk name is keyed off the catch parameter's canonical type mangled
/// name, so a single translation unit emits at most one thunk per catch type.
static cir::FuncOp getOrCreateCopyThunk(CIRGenFunction &cgf,
                                        const VarDecl &catchParam,
                                        cir::PointerType paramAddrType,
                                        mlir::Location loc) {
  CIRGenModule &cgm = cgf.cgm;
  CIRGenBuilderTy &builder = cgm.getBuilder();
  mlir::ModuleOp mod = cgm.getModule();

  const Expr *copyExpr = catchParam.getInit();
  assert(copyExpr && "non-trivial copy expects a copy expression");

  llvm::SmallString<128> thunkName;
  llvm::raw_svector_ostream thunkNameStream(thunkName);
  thunkNameStream << "__clang_cir_catch_copy_";
  cgm.getCXXABI().getMangleContext().mangleCanonicalTypeName(
      catchParam.getType(), thunkNameStream);

  if (cir::FuncOp existing = cgm.lookupFuncOp(thunkName))
    return existing;

  mlir::Type voidTy = cir::VoidType::get(builder.getContext());
  auto thunkTy = cir::FuncType::get({paramAddrType, paramAddrType}, voidTy,
                                    /*isVarArg=*/false);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mod.getBody());
  cir::FuncOp thunk = cir::FuncOp::create(builder, loc, thunkName, thunkTy);
  cgm.insertGlobalSymbol(thunk);
  thunk.setLinkage(cir::GlobalLinkageKind::LinkOnceODRLinkage);
  thunk.setGlobalVisibility(cir::VisibilityKind::Hidden);
  thunk->setAttr(cir::CIRDialect::getCatchCopyThunkAttrName(),
                 builder.getUnitAttr());

  mlir::Block *entry = thunk.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  // Use a fresh CIRGenFunction to drive the body emission. We need just enough
  // state for emitAggExpr / emitCXXConstructorCall to compute the call-site
  // argument attributes; the helper has no AST decl, no exception scopes, and
  // no return value, so we bypass the full startFunction/finishFunction
  // machinery.
  CIRGenFunction subCgf(cgm, builder);
  subCgf.curFn = thunk;

  // Some emission paths (e.g. materializing temporaries for default args via
  // emitAnyExprToTemp) need both a current source location and a lexical
  // scope to anchor allocas. Since we bypass startFunction, install both
  // explicitly for the lifetime of the thunk's body emission.
  CIRGenFunction::SourceLocRAIIObject thunkLoc(subCgf, loc);
  CIRGenFunction::LexicalScope thunkScope(subCgf, loc, entry);

  // Bind the OpaqueValueExpr at the source position of the catch parameter's
  // copy expression to an LValue at the thunk's `src` block argument.
  LValue srcLV = subCgf.makeNaturalAlignAddrLValue(entry->getArgument(1),
                                                   catchParam.getType());
  CIRGenFunction::OpaqueValueMapping opaqueValue(
      subCgf, OpaqueValueExpr::findInCopyConstruct(copyExpr), srcLV);

  // Drive the construction into the helper's `dest` block argument via the
  // normal aggregate-emission machinery so that `ExprWithCleanups`,
  // converting/inheriting constructors, and any future copy-construction
  // shapes flow through unchanged.
  Address destAddr = subCgf.makeNaturalAddressForPointer(
      entry->getArgument(0), catchParam.getType(), clang::CharUnits::Zero());
  subCgf.emitAggExpr(
      copyExpr, AggValueSlot::forAddr(
                    destAddr, Qualifiers(), AggValueSlot::IsNotDestructed,
                    AggValueSlot::IsNotAliased, AggValueSlot::DoesNotOverlap));

  cir::ReturnOp::create(builder, loc);
  return thunk;
}

/// A "special initializer" callback for initializing a catch
/// parameter during catch initialization.
static void initCatchParam(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                           mlir::Value ehToken, const VarDecl &catchParam,
                           SourceLocation loc) {
  CanQualType catchType =
      cgf.cgm.getASTContext().getCanonicalType(catchParam.getType());
  cir::InitCatchKind kind;
  bool shouldInitFromExnDirectly = false;

  // If we're catching by reference, we can just cast the object
  // pointer to the appropriate pointer.
  if (isa<ReferenceType>(catchType)) {
    QualType caughtType = cast<ReferenceType>(catchType)->getPointeeType();
    if (const PointerType *ptr = dyn_cast<PointerType>(caughtType)) {
      shouldInitFromExnDirectly = !ptr->getPointeeType()->isRecordType();
    }
    kind = cir::InitCatchKind::Reference;
  } else {
    cir::TypeEvaluationKind tek = cgf.getEvaluationKind(catchType);
    if (tek == cir::TEK_Aggregate) {
      assert(isa<RecordType>(catchType) && "unexpected catch type!");
      const Expr *copyExpr = catchParam.getInit();
      kind = !copyExpr ? cir::InitCatchKind::TrivialCopy
                       : cir::InitCatchKind::NonTrivialCopy;
    } else {
      // Scalars and complexes.
      if (catchType->hasPointerRepresentation()) {
        switch (catchType.getQualifiers().getObjCLifetime()) {
        case Qualifiers::OCL_Weak:
        case Qualifiers::OCL_Strong:
          kind = cir::InitCatchKind::Objc;
          break;

        case Qualifiers::OCL_ExplicitNone:
        case Qualifiers::OCL_Autoreleasing:
        case Qualifiers::OCL_None:
          kind = cir::InitCatchKind::Pointer;
          break;
        }
      } else {
        kind = cir::InitCatchKind::Scalar;
      }
    }
  }

  CIRGenFunction::AutoVarEmission var = cgf.emitAutoVarAlloca(catchParam);
  Address paramAddr = var.getAllocatedAddress();
  mlir::Location mloc = cgf.getLoc(loc);

  if (kind == cir::InitCatchKind::NonTrivialCopy ||
      (kind == cir::InitCatchKind::Reference && shouldInitFromExnDirectly)) {
    // Sanitizer-checked construction (UBSan vptr/derived-class checks, etc.)
    // would require additional adornments that cir.construct_catch_param does
    // not yet carry.
    assert(!cir::MissingFeatures::sanitizers());

    mlir::FlatSymbolRefAttr copyFun{};
    if (kind == cir::InitCatchKind::NonTrivialCopy) {
      auto paramAddrType =
          mlir::cast<cir::PointerType>(paramAddr.getPointer().getType());
      cir::FuncOp thunk =
          getOrCreateCopyThunk(cgf, catchParam, paramAddrType, mloc);
      copyFun = mlir::FlatSymbolRefAttr::get(thunk.getSymNameAttr());
    }

    cir::ConstructCatchParamOp::create(builder, mloc, ehToken,
                                       paramAddr.getPointer(), kind, copyFun);
  }

  mlir::Value exnPtr = callBeginCatch(cgf, ehToken, builder.getVoidPtrTy());
  cir::InitCatchParamOp::create(builder, mloc, exnPtr, paramAddr.getPointer(),
                                kind);
  cgf.emitAutoVarCleanups(var);
}

/// Begins a catch statement by initializing the catch variable and
/// calling __cxa_begin_catch.
void CIRGenFunction::emitBeginCatch(const CXXCatchStmt *catchStmt,
                                    mlir::Value ehToken) {
  // We have to be very careful with the ordering of cleanups here:
  //   C++ [except.throw]p4:
  //     The destruction [of the exception temporary] occurs
  //     immediately after the destruction of the object declared in
  //     the exception-declaration in the handler.
  //
  // So the precise ordering is:
  //   1.  Construct catch variable.
  //   2.  begin_catch
  //   3.  Enter CallEndCatch cleanup
  //   4.  Enter dtor cleanup
  //
  VarDecl *catchParam = catchStmt->getExceptionDecl();
  if (!catchParam) {
    callBeginCatch(*this, ehToken, builder.getVoidPtrTy());
    return;
  }

  // Emit the local. Make sure the alloca's superseed the current scope, since
  // these are going to be consumed by `cir.catch`, which is not within the
  // current scope.
  initCatchParam(*this, builder, ehToken, *catchParam,
                 catchStmt->getBeginLoc());
}

mlir::LogicalResult
CIRGenFunction::emitCXXTryStmt(const CXXTryStmt &s,
                               cxxTryBodyEmitter &bodyCallback) {
  // Herbception `try { } catch throws(E e) { }` handlers are dispatched on
  // the normal return path (no unwinder), so they bypass the EH machinery
  // entirely. Route them before any personality function is installed.
  bool hasHerbceptionHandler = false;
  for (unsigned i = 0, n = s.getNumHandlers(); i != n; ++i)
    if (isa<CXXCatchThrowsStmt>(s.getHandler(i)))
      hasHerbceptionHandler = true;
  if (hasHerbceptionHandler)
    return emitHerbceptionCatchTry(s);

  mlir::Location loc = getLoc(s.getSourceRange());

  // Create a scope to hold try local storage for catch params.
  mlir::OpBuilder::InsertPoint scopeIP;
  cir::ScopeOp::create(
      builder, loc,
      /*scopeBuilder=*/[&](mlir::OpBuilder &b, mlir::Location loc) {
        scopeIP = builder.saveInsertionPoint();
      });

  // Set personality function if not already set
  auto funcOp = mlir::cast<cir::FuncOp>(curFn);
  if (!funcOp.getPersonality())
    funcOp.setPersonality(getPersonalityFn(cgm, EHPersonality::get(*this)));

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.restoreInsertionPoint(scopeIP);

  const llvm::Triple &t = getTarget().getTriple();
  // If we encounter a try statement on in an OpenMP target region offloaded
  // to a GPU, we treat it as a basic block.
  const bool isTargetDevice =
      (cgm.getLangOpts().OpenMPIsTargetDevice && (t.isNVPTX() || t.isAMDGCN()));
  if (isTargetDevice) {
    cgm.errorNYI("emitCXXTryStmt: OpenMP target region offloaded to GPU");
    return mlir::success();
  }

  mlir::Location tryLoc = getLoc(s.getBeginLoc());
  SmallVector<mlir::Attribute> handlerAttrs;

  CIRGenFunction::LexicalScope tryBodyScope{*this, tryLoc,
                                            builder.getInsertionBlock()};

  if (getLangOpts().EHAsynch) {
    cgm.errorNYI("enterCXXTryStmt: EHAsynch");
    return mlir::failure();
  }

  // Create the try operation.
  mlir::LogicalResult tryRes = mlir::success();
  auto tryOp = cir::TryOp::create(
      builder, tryLoc,
      /*tryBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Create a RunCleanupsScope that allows us to apply any cleanups that
        // are created for statements within the try body before exiting the
        // try body.
        RunCleanupsScope tryBodyCleanups(*this);
        if (bodyCallback(*this).failed())
          tryRes = mlir::failure();
        tryBodyCleanups.forceCleanup();
        if (!builder.getBlock()->mightHaveTerminator() ||
            !builder.getBlock()->getTerminator())
          cir::YieldOp::create(builder, loc);
      },
      /*handlersBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::OperationState &result) {
        mlir::OpBuilder::InsertionGuard guard(b);
        bool hasCatchAll = false;
        unsigned numHandlers = s.getNumHandlers();
        mlir::Type ehTokenTy = cir::EhTokenType::get(&getMLIRContext());
        for (unsigned i = 0; i != numHandlers; ++i) {
          const auto *catchStmt = cast<CXXCatchStmt>(s.getHandler(i));
          if (!catchStmt->getExceptionDecl())
            hasCatchAll = true;
          mlir::Region *region = result.addRegion();
          builder.createBlock(region, /*insertPt=*/{}, {ehTokenTy}, {loc});
          addCatchHandlerAttr(catchStmt, handlerAttrs);
        }
        if (!hasCatchAll) {
          // Create unwind region.
          mlir::Region *region = result.addRegion();
          mlir::Block *unwindBlock =
              builder.createBlock(region, /*insertPt=*/{}, {ehTokenTy}, {loc});
          cir::ResumeOp::create(builder, loc, unwindBlock->getArgument(0));
          handlerAttrs.push_back(cir::UnwindAttr::get(&getMLIRContext()));
        }
      });

  if (tryRes.failed())
    return mlir::failure();

  // Add final array of clauses into TryOp.
  tryOp.setHandlerTypesAttr(
      mlir::ArrayAttr::get(&getMLIRContext(), handlerAttrs));

  // Emit the catch handler bodies. This has to be done after the try op is
  // created and in place so that we can find the insertion point for the
  // catch parameter alloca.
  unsigned numHandlers = s.getNumHandlers();
  for (unsigned i = 0; i != numHandlers; ++i) {
    const auto *catchStmt = cast<CXXCatchStmt>(s.getHandler(i));
    mlir::Region *handler = &tryOp.getHandlerRegions()[i];
    mlir::Location handlerLoc = getLoc(catchStmt->getCatchLoc());

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&handler->front());

    // Get the !cir.eh_token block argument from the handler region.
    mlir::Value ehToken = handler->front().getArgument(0);

    // Enter a cleanup scope, including the catch variable and the
    // end-catch.
    RunCleanupsScope handlerScope(*this);

    // Initialize the catch variable.
    // TODO(cir): Move this out of CXXABI.
    assert(!cir::MissingFeatures::currentFuncletPad());
    emitBeginCatch(catchStmt, ehToken);

    // Emit the PGO counter increment.
    assert(!cir::MissingFeatures::incrementProfileCounter());

    // Perform the body of the catch.
    [[maybe_unused]] mlir::LogicalResult emitResult =
        emitStmt(catchStmt->getHandlerBlock(), /*useCurrentScope=*/true);
    assert(emitResult.succeeded() && "failed to emit catch handler block");

    // [except.handle]p11:
    //   The currently handled exception is rethrown if control
    //   reaches the end of a handler of the function-try-block of a
    //   constructor or destructor.

    // TODO(cir): Handle implicit rethrow?

    // Fall out through the catch cleanups.
    handlerScope.forceCleanup();

    mlir::Block *block = &handler->getBlocks().back();
    if (block->empty() ||
        !block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(block);
      builder.createYield(handlerLoc);
    }
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitCXXTryStmt(const CXXTryStmt &s) {
  if (s.getTryBlock()->body_empty())
    return mlir::LogicalResult::success();

  struct simpleTryBodyEmitter final : cxxTryBodyEmitter {
    const clang::CXXTryStmt &s;
    simpleTryBodyEmitter(const clang::CXXTryStmt &s) : s(s) {}

    mlir::LogicalResult operator()(CIRGenFunction &cgf) override {
      return cgf.emitStmt(s.getTryBlock(), /*useCurrentScope=*/true);
    }
    ~simpleTryBodyEmitter() override = default;
  };

  simpleTryBodyEmitter emitter{s};

  return emitCXXTryStmt(s, emitter);
}

mlir::LogicalResult
CIRGenFunction::emitHerbceptionCatchTry(const CXXTryStmt &s) {
  // Traditional and herbception clauses may interleave in classic codegen
  // (the two channels dispatch independently). CIR does not support the
  // mixed form yet.
  bool hasTraditional = false;
  for (unsigned i = 0, n = s.getNumHandlers(); i != n; ++i)
    if (!isa<CXXCatchThrowsStmt>(s.getHandler(i)))
      hasTraditional = true;
  if (hasTraditional) {
    cgm.errorNYI(s.getSourceRange(),
                 "emitCXXTryStmt: interleaved traditional and herbception "
                 "catch handlers");
    return mlir::failure();
  }

  // A `catch throws(std::error)` handler carrying a legacy conversion
  // expression additionally catches traditional C++ exceptions through a
  // catch-all EH scope; record the first such handler.
  const CXXCatchThrowsStmt *legacyHandlerStmt = nullptr;
  for (unsigned i = 0, n = s.getNumHandlers(); i != n; ++i) {
    const auto *ct = cast<CXXCatchThrowsStmt>(s.getHandler(i));
    if (ct->getLegacyExceptionErrorValue() && !legacyHandlerStmt)
      legacyHandlerStmt = ct;
  }

  const std::string contLabel = "herb.try.cont." + llvm::Twine(
      herbceptionTryCounter++).str();

  // Allocate the error slot of each handler and make the scopes active while
  // the try body is emitted, so herbception failure paths route to them.
  struct HandlerInfo {
    const CXXCatchThrowsStmt *stmt;
    std::string label;
    Address slot;
  };
  llvm::SmallVector<HandlerInfo, 4> handlers;
  unsigned savedScopes = herbceptionCatchScopes.size();
  for (unsigned i = 0, n = s.getNumHandlers(); i != n; ++i) {
    const auto *ct = cast<CXXCatchThrowsStmt>(s.getHandler(i));
    VarDecl *vd = ct->getExceptionDecl();
    mlir::Type errTy =
        convertTypeForMem(vd ? vd->getType() : getContext().VoidPtrTy);
    Address slot = createDefaultAlignTempAlloca(errTy, getLoc(ct->getCatchLoc()),
                                                "herb.error");
    std::string label =
        "catch.throws." + llvm::Twine(herbceptionTryCounter++).str();
    herbceptionCatchScopes.push_back({label, slot});
    handlers.push_back({ct, label, slot});
  }

  // Emit the try body. A `throw throws` inside it stores the error value into
  // the innermost handler's slot and branches to its label; the cleanups of
  // this scope run on that exit.
  if (!legacyHandlerStmt) {
    RunCleanupsScope bodyCleanups(*this);
    if (emitStmt(s.getTryBlock(), /*useCurrentScope=*/true).failed()) {
      herbceptionCatchScopes.truncate(savedScopes);
      return mlir::failure();
    }
  } else {
    // Wrap the try block in a catch-all EH scope so calls to noexcept(false)
    // functions inside it become invokes; the caught legacy exception is
    // converted to a fabricated std::error and routed to its handler.
    mlir::Location tryLoc = getLoc(s.getBeginLoc());
    llvm::SmallVector<mlir::Attribute> handlerAttrs;
    mlir::LogicalResult tryBodyRes = mlir::success();
    auto tryOp = cir::TryOp::create(
        builder, tryLoc,
        /*tryBuilder=*/[&](mlir::OpBuilder &, mlir::Location loc) {
          RunCleanupsScope bodyCleanups(*this);
          if (emitStmt(s.getTryBlock(), /*useCurrentScope=*/true).failed())
            tryBodyRes = mlir::failure();
          if (!builder.getBlock()->mightHaveTerminator() ||
              !builder.getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>())
            cir::YieldOp::create(builder, loc);
        },
        /*handlersBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc,
            mlir::OperationState &result) {
          mlir::Type ehTokenTy = cir::EhTokenType::get(&getMLIRContext());
          // The catch-all claims every exception, so no unwind region is
          // added alongside it.
          mlir::Region *handlerRegion = result.addRegion();
          b.createBlock(handlerRegion, /*insertPt=*/{}, {ehTokenTy}, {loc});
          handlerAttrs.push_back(cir::CatchAllAttr::get(&getMLIRContext()));
        });
    tryOp.setHandlerTypesAttr(
        mlir::ArrayAttr::get(&getMLIRContext(), handlerAttrs));

    // Fill the catch-all region: begin the catch, fabricate the std::error
    // from the thrown object pointer, store it into the handler's slot and
    // route to it. The exception is deliberately never ended: it is consumed
    // by the conversion, mirroring classic codegen.
    mlir::Region &catchAll = tryOp.getHandlerRegions().front();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&catchAll.front());
    mlir::Value ehToken = catchAll.front().getArgument(0);

    RunCleanupsScope convertScope(*this);
    auto beginCatch = cir::BeginCatchOp::create(
        builder, getLoc(legacyHandlerStmt->getBeginLoc()),
        cir::CatchTokenType::get(builder.getContext()), builder.getVoidPtrTy(),
        ehToken);
    mlir::Value exnPtr = beginCatch.getExnPtr();
    const HandlerInfo *target = nullptr;
    for (const HandlerInfo &h : handlers)
      if (h.stmt == legacyHandlerStmt) {
        target = &h;
        break;
      }
    {
      llvm::SaveAndRestore<mlir::Value> restoreExnPtr(curHerbceptionExnPtr,
                                                      exnPtr);
      emitAnyExprToMem(legacyHandlerStmt->getLegacyExceptionErrorValue(),
                       target->slot, Qualifiers(), /*isInitializer=*/false);
    }
    convertScope.forceCleanup();

    cir::GotoOp::create(builder, tryLoc, target->label);
    if (tryBodyRes.failed()) {
      herbceptionCatchScopes.truncate(savedScopes);
      return mlir::failure();
    }
  }

  // The herbception handlers are emitted with their catch scopes no longer
  // active: a `throw throws` inside a handler routes past this try statement
  // (to an enclosing handler), never to one of its siblings.
  herbceptionCatchScopes.truncate(savedScopes);

  // Fall through the try body into the continuation.
  if (!builder.getBlock()->mightHaveTerminator() ||
      !builder.getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>())
    cir::GotoOp::create(builder, getLoc(s.getEndLoc()), contLabel);

  // Emit the handlers. They are reached only through the routing above, so
  // each one starts a fresh block behind a cir.label.
  for (const HandlerInfo &h : handlers) {
    builder.createBlock(builder.getBlock()->getParent());
    cir::LabelOp::create(builder, getLoc(h.stmt->getCatchLoc()), h.label);

    RunCleanupsScope handlerScope(*this);
    if (VarDecl *vd = h.stmt->getExceptionDecl()) {
      // Bind the exception variable directly from the error payload slot. The
      // error value is compiler-fabricated (its constructors are deleted), so
      // it is not default/copy-constructed; the variable's destructor is
      // registered so ~std::error runs do_cleanup exactly once when the
      // handler exits.
      AutoVarEmission var = emitAutoVarAlloca(*vd);
      Address addr = var.getObjectAddress(*this);
      assert(addr.getElementType() == h.slot.getElementType() &&
             "herbception error slot type mismatch");
      mlir::Value errVal =
          builder.createLoad(getLoc(h.stmt->getCatchLoc()), h.slot);
      builder.createStore(getLoc(h.stmt->getCatchLoc()), errVal, addr);
      emitAutoVarCleanups(var);
    }

    if (emitStmt(h.stmt->getHandlerBlock(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    handlerScope.forceCleanup();

    mlir::Block *block = builder.getBlock();
    if (block->empty() ||
        !block->back().hasTrait<mlir::OpTrait::IsTerminator>())
      cir::GotoOp::create(builder, getLoc(h.stmt->getEndLoc()), contLabel);
  }

  // The continuation point, reached from the end of the try body or the end
  // of any handler.
  builder.createBlock(builder.getBlock()->getParent());
  cir::LabelOp::create(builder, getLoc(s.getEndLoc()), contLabel);

  return mlir::success();
}

// in classic codegen this function is mapping to `isInvokeDest` previously
// and currently it's mapping to the conditions that performs early returns in
// `getInvokeDestImpl`, in CIR we need the condition to know if the EH scope
// may throw exception or now.
bool CIRGenFunction::isCatchOrCleanupRequired() {
  // If exceptions are disabled/ignored and SEH is not in use, then there is
  // no invoke destination. SEH "works" even if exceptions are off. In
  // practice, this means that C++ destructors and other EH cleanups don't
  // run, which is consistent with MSVC's behavior, except in the presence of
  // -EHa
  const LangOptions &lo = cgm.getLangOpts();
  if (!lo.Exceptions || lo.IgnoreExceptions) {
    if (!lo.Borland && !lo.MicrosoftExt)
      return false;
    cgm.errorNYI("isInvokeDest: no exceptions or ignore exception");
    return false;
  }

  // CUDA device code doesn't have exceptions.
  if (lo.CUDA && lo.CUDAIsDevice)
    return false;

  return ehStack.requiresCatchOrCleanup();
}

RValue CIRGenFunction::emitHerbceptionTry(const CXXTryExpr *E) {
  assert(curFnInfo && curFnInfo->hasThrowsReturn() &&
         "herbception try outside a throws function");

  const Expr *sub = E->getSubExpr()->IgnoreParenImpCasts();
  const CallExpr *call = cast<CallExpr>(sub);
  QualType callTy = call->getType();

  // Emit the call. It returns {T, i1}; suppress routing to an enclosing
  // herbception catch scope: this expression handles the discriminant itself.
  // The result is an aggregate RValue holding the {T, i1} shaped struct.
  RValue callRValue;
  {
    llvm::SaveAndRestore<bool> saveInOperand(inHerbceptionOperand, true);
    callRValue = emitCallExpr(call);
  }
  assert(callRValue.isAggregate() &&
         "throws call must return aggregate {T, i1}");

  // Copy the {T, i1} result to a temp so we can extract members.
  Address callAddr = callRValue.getAggregateAddress();
  auto shapedTy = cast<cir::RecordType>(callAddr.getElementType());
  assert(shapedTy.getMembers().size() == 2 &&
         "throws call must return {T, i1}");
  CharUnits align =
      CharUnits::fromQuantity(cgm.getDataLayout().getABITypeAlign(shapedTy));
  Address tmp =
      createTempAlloca(shapedTy, align, getLoc(E->getExprLoc()), "__herb.ret");
  builder.createStore(getLoc(E->getExprLoc()),
                      callRValue.getAggregateAddress().getPointer(), tmp);

  llvm::SmallVector<mlir::Type> members(shapedTy.getMembers().begin(),
                                        shapedTy.getMembers().end());
  auto memberAddr = [&](unsigned idx) {
    mlir::Value memberPtr = builder.createGetMember(
        getLoc(E->getExprLoc()), builder.getPointerTo(members[idx]),
        tmp.getPointer(), "", idx);
    return Address(memberPtr, members[idx], align);
  };

  mlir::Location loc = getLoc(E->getExprLoc());
  mlir::Value disc = builder.createLoad(loc, memberAddr(1));
  Address successAddr = memberAddr(0);

  // Error path: if an enclosing catch-throws handler is active, route to it;
  // otherwise auto-propagate through the enclosing throws function.
  cir::IfOp routeIf = cir::IfOp::create(
      builder, loc, disc,
      /*withElseRegion=*/false, [&](mlir::OpBuilder &, mlir::Location) {
        // Error path
        if (!herbceptionCatchScopes.empty()) {
          const HerbceptionCatchScope &scope = herbceptionCatchScopes.back();
          mlir::Value payload = builder.createLoad(loc, successAddr);
          mlir::Type slotTy = scope.errorSlot.getElementType();
          if (slotTy != payload.getType()) {
            Address ptmp =
                createTempAlloca(payload.getType(), align, loc, "herb.payload");
            builder.createStore(loc, payload, ptmp);
            mlir::Value casted = builder.createBitcast(
                ptmp.getBasePointer(), builder.getPointerTo(slotTy));
            payload = builder.createLoad(loc, Address(casted, slotTy, align));
          }
          builder.createStore(loc, payload, scope.errorSlot);
          cir::GotoOp::create(builder, loc, scope.handlerLabel);
        } else {
          // Auto-propagate: wrap the error payload with disc=true and return.
          mlir::Value payload = builder.createLoad(loc, successAddr);
          mlir::Type callerPayloadTy = members[0];
          if (payload.getType() != callerPayloadTy) {
            Address ptmp =
                createTempAlloca(payload.getType(), align, loc, "herb.payload");
            builder.createStore(loc, payload, ptmp);
            mlir::Value casted = builder.createBitcast(
                ptmp.getBasePointer(), builder.getPointerTo(callerPayloadTy));
            payload = builder.createLoad(
                loc, Address(casted, callerPayloadTy, align));
          }
          mlir::Value wrapped =
              wrapHerbceptionReturnValue(loc, payload, /*disc=*/true);
          cir::ReturnOp::create(builder, loc, wrapped);
        }
      });

  // Success path: the try expression's value is the success value.
  builder.setInsertionPointAfter(routeIf);

  if (getEvaluationKind(callTy) == cir::TEK_Scalar) {
    mlir::Value v = builder.createLoad(loc, successAddr);
    return RValue::get(v);
  }
  if (getEvaluationKind(callTy) == cir::TEK_Aggregate) {
    Address payloadAddr = createMemTemp(callTy, loc, "try.payload");
    mlir::Value payload = builder.createLoad(loc, successAddr);
    emitAggregateStore(payload, payloadAddr);
    return RValue::getAggregate(payloadAddr);
  }
  cgm.errorNYI(E->getSourceRange(), "emitHerbceptionTry: complex result");
  return getUndefRValue(callTy);
}

RValue CIRGenFunction::emitHerbceptionCatchReturnFailure(const CXXCatchReturnFailureExpr *E) {
  const Expr *sub = E->getSubExpr()->IgnoreParenImpCasts();
  const CallExpr *call = cast<CallExpr>(sub);

  // Emit the call. It returns {T, i1}; suppress routing to an enclosing
  // herbception catch scope. The result is an aggregate RValue holding the
  // {T, i1} shaped struct.
  RValue callRValue;
  {
    llvm::SaveAndRestore<bool> saveInOperand(inHerbceptionOperand, true);
    callRValue = emitCallExpr(call);
  }
  assert(callRValue.isAggregate() &&
         "throws call must return aggregate {T, i1}");

  // Copy the {T, i1} result to a temp so we can extract members.
  Address callAddr = callRValue.getAggregateAddress();
  auto shapedTy = cast<cir::RecordType>(callAddr.getElementType());
  assert(shapedTy.getMembers().size() == 2 &&
         "throws call must return {T, i1}");
  CharUnits align =
      CharUnits::fromQuantity(cgm.getDataLayout().getABITypeAlign(shapedTy));
  Address tmp =
      createTempAlloca(shapedTy, align, getLoc(E->getExprLoc()), "__herb.ret");
  builder.createStore(getLoc(E->getExprLoc()),
                      callRValue.getAggregateAddress().getPointer(), tmp);

  llvm::SmallVector<mlir::Type> members(shapedTy.getMembers().begin(),
                                        shapedTy.getMembers().end());

  mlir::Location loc = getLoc(E->getExprLoc());
  auto memberAddr = [&](unsigned idx) {
    mlir::Value memberPtr = builder.createGetMember(
        loc, builder.getPointerTo(members[idx]), tmp.getPointer(), "", idx);
    return Address(memberPtr, members[idx], align);
  };

  mlir::Value disc = builder.createLoad(loc, memberAddr(1));
  mlir::Value payload = builder.createLoad(loc, memberAddr(0));

  // Build the catch-fails value: struct { union { T value; E error; }; bool
  // failed; }
  QualType eitherTy = E->getType();
  const RecordDecl *RD = eitherTy->getAsRecordDecl();
  assert(RD && "either type must be a record");

  Address addr = createMemTempWithoutCast(eitherTy, loc, "herb.catch");

  // Find field by name, descending through anonymous union if present.
  auto findField =
      [&](StringRef Name) -> std::pair<const FieldDecl *, const FieldDecl *> {
    for (const FieldDecl *F : RD->fields()) {
      if (F->getName() == Name)
        return {F, F};
      if (F->isAnonymousStructOrUnion()) {
        if (const auto *FR = F->getType()->getAsRecordDecl())
          for (const FieldDecl *Sub : FR->fields())
            if (Sub->getName() == Name)
              return {F, Sub};
      }
    }
    return {nullptr, nullptr};
  };

  auto storeField = [&](StringRef Name, mlir::Value V) {
    auto [outer, inner] = findField(Name);
    if (!inner)
      return;
    // For fields inside an anonymous union, we need to:
    // 1. Get the address of the union (a direct field of the struct)
    // 2. Get the address of the field inside the union
    const RecordDecl *fieldParent = inner->getParent();
    if (fieldParent->isUnion()) {
      // Field is inside a union - get union address first, then field address
      const CIRGenRecordLayout &layout =
          cgm.getTypes().getCIRGenRecordLayout(RD);
      unsigned unionIndex = layout.getCIRFieldNo(outer);
      mlir::Type unionTy = layout.getCIRType().getMembers()[unionIndex];
      mlir::Value unionPtr = builder.createGetMember(
          loc, builder.getPointerTo(unionTy), addr.getPointer(),
          outer->getName(), unionIndex);
      // For unions, all fields map to index 0
      mlir::Type fieldType = convertType(inner->getType());
      mlir::Value fieldPtr = builder.createGetMember(
          loc, builder.getPointerTo(fieldType), unionPtr, inner->getName(), 0);
      Address fieldAddr(fieldPtr, fieldType, align);
      builder.createStore(loc, V, fieldAddr);
    } else {
      // Direct field of the record - use the record layout
      const CIRGenRecordLayout &layout =
          cgm.getTypes().getCIRGenRecordLayout(fieldParent);
      unsigned fieldIndex = layout.getCIRFieldNo(inner);
      Address fieldAddr = emitAddrOfFieldStorage(addr, inner, Name, fieldIndex);
      if (!fieldAddr.isValid())
        return;
      mlir::Type fieldTy = fieldAddr.getElementType();
      mlir::Value toStore = V;
      if (fieldTy != V.getType()) {
        Address ptmp =
            createTempAlloca(V.getType(), align, loc, "herb.payload");
        builder.createStore(loc, V, ptmp);
        mlir::Value casted = builder.createBitcast(
            ptmp.getBasePointer(), builder.getPointerTo(fieldTy));
        toStore = builder.createLoad(loc, Address(casted, fieldTy, align));
      }
      builder.createStore(loc, toStore, fieldAddr);
    }
  };

  storeField("failed", disc);
  storeField("value", payload);
  storeField("error", payload);

  return RValue::getAggregate(addr);
}
