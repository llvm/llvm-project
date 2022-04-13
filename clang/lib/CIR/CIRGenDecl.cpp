//===--- CIRGenDecl.cpp - Emit CIR Code for declarations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

#include "clang/AST/Decl.h"

using namespace cir;
using namespace clang;

CIRGenFunction::AutoVarEmission
CIRGenFunction::buildAutoVarAlloca(const VarDecl &D) {
  QualType Ty = D.getType();
  // TODO: (|| Ty.getAddressSpace() == LangAS::opencl_private &&
  //        getLangOpts().OpenCL))
  assert(Ty.getAddressSpace() == LangAS::Default);

  assert(!D.isEscapingByref() && "not implemented");
  assert(!Ty->isVariablyModifiedType() && "not implemented");
  assert(!getContext()
              .getLangOpts()
              .OpenMP && // !CGF.getLangOpts().OpenMPIRBuilder
         "not implemented");
  bool NRVO =
      getContext().getLangOpts().ElideConstructors && D.isNRVOVariable();
  assert(!NRVO && "not implemented");
  assert(Ty->isConstantSizeType() && "not implemented");
  assert(!D.hasAttr<AnnotateAttr>() && "not implemented");

  AutoVarEmission emission(D);
  CharUnits alignment = getContext().getDeclAlign(&D);
  // TODO: debug info
  // TODO: use CXXABI

  // If this value is an array or struct with a statically determinable
  // constant initializer, there are optimizations we can do.
  //
  // TODO: We should constant-evaluate the initializer of any variable,
  // as long as it is initialized by a constant expression. Currently,
  // isConstantInitializer produces wrong answers for structs with
  // reference or bitfield members, and a few other cases, and checking
  // for POD-ness protects us from some of these.
  if (D.getInit() && (Ty->isArrayType() || Ty->isRecordType()) &&
      (D.isConstexpr() ||
       ((Ty.isPODType(getContext()) ||
         getContext().getBaseElementType(Ty)->isObjCObjectPointerType()) &&
        D.getInit()->isConstantInitializer(getContext(), false)))) {

    // If the variable's a const type, and it's neither an NRVO
    // candidate nor a __block variable and has no mutable members,
    // emit it as a global instead.
    // Exception is if a variable is located in non-constant address space
    // in OpenCL.
    // TODO: deal with CGM.getCodeGenOpts().MergeAllConstants
    // TODO: perhaps we don't need this at all at CIR since this can
    // be done as part of lowering down to LLVM.
    if ((!getContext().getLangOpts().OpenCL ||
         Ty.getAddressSpace() == LangAS::opencl_constant) &&
        (!NRVO && !D.isEscapingByref() && CGM.isTypeConstant(Ty, true)))
      assert(0 && "not implemented");

    // Otherwise, tell the initialization code that we're in this case.
    emission.IsConstantAggregate = true;
  }

  // TODO: track source location range...
  mlir::Value addr;
  if (failed(declare(&D, Ty, getLoc(D.getSourceRange()), alignment, addr))) {
    CGM.emitError("Cannot declare variable");
    return emission;
  }

  // TODO: what about emitting lifetime markers for MSVC catch parameters?
  // TODO: something like @llvm.lifetime.start/end here? revisit this later.
  emission.Addr = Address{addr, alignment};
  return emission;
}

/// Determine whether the given initializer is trivial in the sense
/// that it requires no code to be generated.
bool CIRGenFunction::isTrivialInitializer(const Expr *Init) {
  if (!Init)
    return true;

  if (const CXXConstructExpr *Construct = dyn_cast<CXXConstructExpr>(Init))
    if (CXXConstructorDecl *Constructor = Construct->getConstructor())
      if (Constructor->isTrivial() && Constructor->isDefaultConstructor() &&
          !Construct->requiresZeroInitialization())
        return true;

  return false;
}
void CIRGenFunction::buildAutoVarInit(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  const VarDecl &D = *emission.Variable;
  QualType type = D.getType();

  // If this local has an initializer, emit it now.
  const Expr *Init = D.getInit();

  // TODO: in LLVM codegen if we are at an unreachable point, the initializer
  // isn't emitted unless it contains a label. What we want for CIR?
  assert(builder.getInsertionBlock());

  // Initialize the variable here if it doesn't have a initializer and it is a
  // C struct that is non-trivial to initialize or an array containing such a
  // struct.
  if (!Init && type.isNonTrivialToPrimitiveDefaultInitialize() ==
                   QualType::PDIK_Struct) {
    assert(0 && "not implemented");
    return;
  }

  const Address Loc = emission.Addr;

  // Note: constexpr already initializes everything correctly.
  LangOptions::TrivialAutoVarInitKind trivialAutoVarInit =
      (D.isConstexpr()
           ? LangOptions::TrivialAutoVarInitKind::Uninitialized
           : (D.getAttr<UninitializedAttr>()
                  ? LangOptions::TrivialAutoVarInitKind::Uninitialized
                  : getContext().getLangOpts().getTrivialAutoVarInit()));

  auto initializeWhatIsTechnicallyUninitialized = [&](Address Loc) {
    if (trivialAutoVarInit ==
        LangOptions::TrivialAutoVarInitKind::Uninitialized)
      return;

    assert(0 && "unimplemented");
  };

  if (isTrivialInitializer(Init))
    return initializeWhatIsTechnicallyUninitialized(Loc);

  if (emission.IsConstantAggregate ||
      D.mightBeUsableInConstantExpressions(getContext())) {
    assert(0 && "not implemented");
  }

  initializeWhatIsTechnicallyUninitialized(Loc);
  LValue lv = LValue::makeAddr(Loc, type, AlignmentSource::Decl);
  return buildExprAsInit(Init, &D, lv);
}

void CIRGenFunction::buildAutoVarCleanups(const AutoVarEmission &emission) {
  assert(emission.Variable && "emission was not valid!");

  // TODO: in LLVM codegen if we are at an unreachable point codgen
  // is ignored. What we want for CIR?
  assert(builder.getInsertionBlock());
  const VarDecl &D = *emission.Variable;

  // Check the type for a cleanup.
  // TODO: something like emitAutoVarTypeCleanup
  if (QualType::DestructionKind dtorKind = D.needsDestruction(getContext()))
    assert(0 && "not implemented");

  // In GC mode, honor objc_precise_lifetime.
  if (getContext().getLangOpts().getGC() != LangOptions::NonGC &&
      D.hasAttr<ObjCPreciseLifetimeAttr>())
    assert(0 && "not implemented");

  // Handle the cleanup attribute.
  if (const CleanupAttr *CA = D.getAttr<CleanupAttr>())
    assert(0 && "not implemented");

  // TODO: handle block variable
}

/// Emit code and set up symbol table for a variable declaration with auto,
/// register, or no storage class specifier. These turn into simple stack
/// objects, globals depending on target.
void CIRGenFunction::buildAutoVarDecl(const VarDecl &D) {
  AutoVarEmission emission = buildAutoVarAlloca(D);
  buildAutoVarInit(emission);
  buildAutoVarCleanups(emission);
}

void CIRGenFunction::buildVarDecl(const VarDecl &D) {
  if (D.hasExternalStorage()) {
    assert(0 && "should we just returns is there something to track?");
    // Don't emit it now, allow it to be emitted lazily on its first use.
    return;
  }

  // Some function-scope variable does not have static storage but still
  // needs to be emitted like a static variable, e.g. a function-scope
  // variable in constant address space in OpenCL.
  if (D.getStorageDuration() != SD_Automatic)
    assert(0 && "not implemented");

  if (D.getType().getAddressSpace() == LangAS::opencl_local)
    assert(0 && "not implemented");

  assert(D.hasLocalStorage());
  return buildAutoVarDecl(D);
}

void CIRGenFunction::buildScalarInit(const Expr *init, const ValueDecl *D,
                                     LValue lvalue) {
  // TODO: this is where a lot of ObjC lifetime stuff would be done.
  mlir::Value value = buildScalarExpr(init);
  SourceLocRAIIObject Loc{*this, getLoc(D->getSourceRange())};
  buldStoreThroughLValue(RValue::get(value), lvalue, D);
  return;
}

void CIRGenFunction::buildExprAsInit(const Expr *init, const ValueDecl *D,
                                     LValue lvalue) {
  QualType type = D->getType();

  if (type->isReferenceType()) {
    assert(0 && "not implemented");
    return;
  }
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case TEK_Scalar:
    buildScalarInit(init, D, lvalue);
    return;
  case TEK_Complex: {
    assert(0 && "not implemented");
    return;
  }
  case TEK_Aggregate:
    assert(0 && "not implemented");
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

void CIRGenFunction::buildDecl(const Decl &D) {
  switch (D.getKind()) {
  case Decl::ImplicitConceptSpecialization:
  case Decl::HLSLBuffer:
  case Decl::UnnamedGlobalConstant:
  case Decl::TopLevelStmt:
    llvm_unreachable("NYI");
  case Decl::BuiltinTemplate:
  case Decl::TranslationUnit:
  case Decl::ExternCContext:
  case Decl::Namespace:
  case Decl::UnresolvedUsingTypename:
  case Decl::ClassTemplateSpecialization:
  case Decl::ClassTemplatePartialSpecialization:
  case Decl::VarTemplateSpecialization:
  case Decl::VarTemplatePartialSpecialization:
  case Decl::TemplateTypeParm:
  case Decl::UnresolvedUsingValue:
  case Decl::NonTypeTemplateParm:
  case Decl::CXXDeductionGuide:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor:
  case Decl::CXXConversion:
  case Decl::Field:
  case Decl::MSProperty:
  case Decl::IndirectField:
  case Decl::ObjCIvar:
  case Decl::ObjCAtDefsField:
  case Decl::ParmVar:
  case Decl::ImplicitParam:
  case Decl::ClassTemplate:
  case Decl::VarTemplate:
  case Decl::FunctionTemplate:
  case Decl::TypeAliasTemplate:
  case Decl::TemplateTemplateParm:
  case Decl::ObjCMethod:
  case Decl::ObjCCategory:
  case Decl::ObjCProtocol:
  case Decl::ObjCInterface:
  case Decl::ObjCCategoryImpl:
  case Decl::ObjCImplementation:
  case Decl::ObjCProperty:
  case Decl::ObjCCompatibleAlias:
  case Decl::PragmaComment:
  case Decl::PragmaDetectMismatch:
  case Decl::AccessSpec:
  case Decl::LinkageSpec:
  case Decl::Export:
  case Decl::ObjCPropertyImpl:
  case Decl::FileScopeAsm:
  case Decl::Friend:
  case Decl::FriendTemplate:
  case Decl::Block:
  case Decl::Captured:
  case Decl::UsingShadow:
  case Decl::ConstructorUsingShadow:
  case Decl::ObjCTypeParam:
  case Decl::Binding:
  case Decl::UnresolvedUsingIfExists:
    llvm_unreachable("Declaration should not be in declstmts!");
  case Decl::Record:    // struct/union/class X;
  case Decl::CXXRecord: // struct/union/class X; [C++]
    assert(0 && "Not implemented");
    return;
  case Decl::Enum: // enum X;
    assert(0 && "Not implemented");
    return;
  case Decl::Function:     // void X();
  case Decl::EnumConstant: // enum ? { X = ? }
  case Decl::StaticAssert: // static_assert(X, ""); [C++0x]
  case Decl::Label:        // __label__ x;
  case Decl::Import:
  case Decl::MSGuid: // __declspec(uuid("..."))
  case Decl::TemplateParamObject:
  case Decl::OMPThreadPrivate:
  case Decl::OMPAllocate:
  case Decl::OMPCapturedExpr:
  case Decl::OMPRequires:
  case Decl::Empty:
  case Decl::Concept:
  case Decl::LifetimeExtendedTemporary:
  case Decl::RequiresExprBody:
    // None of these decls require codegen support.
    return;

  case Decl::NamespaceAlias:
    assert(0 && "Not implemented");
    return;
  case Decl::Using: // using X; [C++]
    assert(0 && "Not implemented");
    return;
  case Decl::UsingEnum: // using enum X; [C++]
    assert(0 && "Not implemented");
    return;
  case Decl::UsingPack:
    assert(0 && "Not implemented");
    return;
  case Decl::UsingDirective: // using namespace X; [C++]
    assert(0 && "Not implemented");
    return;
  case Decl::Var:
  case Decl::Decomposition: {
    const VarDecl &VD = cast<VarDecl>(D);
    assert(VD.isLocalVarDecl() &&
           "Should not see file-scope variables inside a function!");
    buildVarDecl(VD);
    if (auto *DD = dyn_cast<DecompositionDecl>(&VD))
      assert(0 && "Not implemented");

    // FIXME: add this
    // if (auto *DD = dyn_cast<DecompositionDecl>(&VD))
    //   for (auto *B : DD->bindings())
    //     if (auto *HD = B->getHoldingVar())
    //       EmitVarDecl(*HD);
    return;
  }

  case Decl::OMPDeclareReduction:
  case Decl::OMPDeclareMapper:
    assert(0 && "Not implemented");

  case Decl::Typedef:     // typedef int X;
  case Decl::TypeAlias: { // using X = int; [C++0x]
    assert(0 && "Not implemented");
  }
  }
}
