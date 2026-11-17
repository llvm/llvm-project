//===--- Program.cpp - Bytecode for the constexpr VM ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Program.h"
#include "Context.h"
#include "Function.h"
#include "PrimType.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

#define DEBUG_TYPE "exprconstant"
using namespace clang;
using namespace clang::interp;

Pointer Program::getPtrGlobal(unsigned Idx) const {
  assert(Idx < Globals.size());

  Block *B = Globals[Idx];

  // Force de-serialization of a redeclaration that might initialize this
  // global.
  if (B->isGlobal() && B->getBlockDesc<GlobalInlineDescriptor>().InitState !=
                           GlobalInitState::Initialized) {
    if (const VarDecl *VD = B->getDescriptor()->asVarDecl()) {
      const VarDecl *MD = VD->getMostRecentDecl();
      if (MD != VD && MD->hasInit() && !MD->getInit()->isValueDependent()) {
        MD->evaluateValue();
        // Note that we need to get Globals[Idx] here again since the code block
        // above might've actually changed what global Idx points to.
        return Pointer(Globals[Idx]);
      }
    }
  }

  return Pointer(B);
}

UnsignedOrNone Program::getGlobal(const ValueDecl *VD) {
  if (auto It = GlobalIndices.find(VD); It != GlobalIndices.end())
    return It->second;

  // Find any previous declarations which were already evaluated.
  std::optional<unsigned> Index;
  for (const Decl *P = VD->getPreviousDecl(); P; P = P->getPreviousDecl()) {
    if (auto It = GlobalIndices.find(P); It != GlobalIndices.end()) {
      Index = It->second;
      break;
    }
  }

  // Map the decl to the existing index.
  if (Index)
    GlobalIndices[VD] = *Index;

  return std::nullopt;
}

UnsignedOrNone Program::getGlobal(const Expr *E) {
  if (auto It = GlobalIndices.find(E); It != GlobalIndices.end())
    return It->second;
  return std::nullopt;
}

UnsignedOrNone Program::getOrCreateGlobal(const ValueDecl *VD,
                                          const Expr *Init) {
  if (auto Idx = getGlobal(VD))
    return Idx;

  if (auto Idx = createGlobal(VD, Init)) {
    GlobalIndices[VD] = *Idx;
    return Idx;
  }
  return std::nullopt;
}

UnsignedOrNone Program::createGlobal(const ValueDecl *VD, const Expr *Init,
                                     bool IsConstexprUnknown,
                                     bool CanBeExtern) {
  bool IsExtern;
  bool IsWeak = VD->isWeak();
  if (const auto *Var = dyn_cast<VarDecl>(VD)) {
    IsExtern = CanBeExtern && !Init && Var->hasExternalStorage();
  } else {
    assert((isa<UnnamedGlobalConstantDecl, MSGuidDecl, TemplateParamObjectDecl>(
        VD)));
    IsExtern = false;
  }

  // Register all previous declarations as well. For extern blocks, just replace
  // the index with the new variable.
  UnsignedOrNone Idx = createGlobal(VD, VD->getType(), IsExtern, IsWeak,
                                    IsConstexprUnknown, Init);
  if (!Idx)
    return std::nullopt;

  Block *NewGlobal = Globals[*Idx];
  GlobalIndices[VD] = *Idx;

  for (const Decl *Redecl = VD->getPreviousDecl(); Redecl;
       Redecl = Redecl->getPreviousDecl()) {
    // If the redeclaration hasn't been registered yet at all, we just set its
    // global index to Idx. If it has been registered yet, it might have
    // pointers pointing to it and we need to transfer those pointers to the new
    // block.
    auto [Iter, Inserted] = GlobalIndices.try_emplace(Redecl);
    if (Inserted) {
      Iter->second = *Idx;
      continue;
    }

    Block *RedeclBlock = Globals[Iter->second];
    // All pointers pointing to the previous extern decl now point to the
    // new decl.
    // A previous iteration might've already fixed up the pointers for this
    // global.
    if (RedeclBlock != NewGlobal) {
      RedeclBlock->movePointersTo(NewGlobal);
      RedeclBlock->invokeDtor();
    }

    Globals[Iter->second] = NewGlobal;
    Iter->second = *Idx;
  }

  return *Idx;
}

UnsignedOrNone Program::createGlobal(const Expr *E, QualType ExprType) {
  if (auto Idx = getGlobal(E))
    return Idx;
  if (auto Idx = createGlobal(E, ExprType,
                              /*IsExtern=*/false, /*IsWeak=*/false,
                              /*IsConstexprUnknown=*/false)) {
    GlobalIndices[E] = *Idx;
    return *Idx;
  }
  return std::nullopt;
}

UnsignedOrNone Program::createGlobal(DeclOrExpr D, QualType Ty, bool IsExtern,
                                     bool IsWeak, bool IsConstexprUnknown,
                                     const Expr *Init) {
  // Since this global variable is constexpr-unknown and a reference, register
  // the pointee type instead. When referencing the variable, the pointer will
  // then be of the pointee type instead of just PT_Ptr.
  if (Ty->isReferenceType() && IsConstexprUnknown)
    Ty = Ty->getPointeeType();

  // Create a descriptor for the global.
  Descriptor *Desc;
  const bool IsConst = Ty.isConstQualified();
  const bool IsTemporary = D.isExpr();
  const bool IsVolatile = Ty.isVolatileQualified();
  if (OptPrimType T = Ctx.classify(Ty))
    Desc = createDescriptor(D, *T, nullptr, IsConst, IsTemporary,
                            /*IsMutable=*/false, IsVolatile);
  else
    Desc = createDescriptor(D, Ty.getTypePtr(), IsConst, IsTemporary,
                            /*IsMutable=*/false, IsVolatile, Init);

  if (!Desc)
    return std::nullopt;

  // Allocate a block for storage.
  unsigned I = Globals.size();

  BlockFlags Flags =
      BlockFlags().withExtern(IsExtern).withWeak(IsWeak).withConstexprUnknown(
          IsConstexprUnknown);
  auto *G =
      new (Allocate(Block::allocSize(Desc, Block::Global), alignof(Block)))
          Block(Ctx.getEvalID(), getCurrentDecl(), Desc, Block::Global, Flags);
  G->invokeCtor();

  // Initialize GlobalInlineDescriptor fields.
  auto *GD =
      new (&G->getBlockDesc<GlobalInlineDescriptor>()) GlobalInlineDescriptor();
  if (!Init)
    GD->InitState = GlobalInitState::NoInitializer;
  Globals.push_back(G);

  return I;
}

Function *Program::getFunction(const FunctionDecl *F) {
  F = F->getFirstDecl();
  assert(F);
  auto It = Funcs.find(F);
  return It == Funcs.end() ? nullptr : It->second;
}

Record *Program::getOrCreateRecord(const RecordDecl *RD) {
  // Use the actual definition as a key.
  RD = RD->getDefinition();
  if (!RD)
    return nullptr;

  if (!RD->isCompleteDefinition())
    return nullptr;

  // Return an existing record if available.  Otherwise, we insert nullptr now
  // and replace that later, so recursive calls to this function with the same
  // RecordDecl don't run into infinite recursion.
  auto [It, Inserted] = Records.try_emplace(RD);
  if (!Inserted)
    return It->second;

  // Number of bytes required by fields and base classes.
  unsigned BaseSize = 0;
  // Number of bytes required by virtual base.
  unsigned VirtSize = 0;

  // Helper to get a base descriptor.
  auto GetBaseDesc = [this](const RecordDecl *BD,
                            const Record *BR) -> const Descriptor * {
    if (!BR)
      return nullptr;
    return allocateDescriptor(BD, BR, /*IsConst=*/false, /*IsTemporary=*/false,
                              /*IsMutable=*/false, /*IsVolatile=*/false);
  };

  bool HasPtrField = false;
  // Reserve space for base classes.
  unsigned NumBases = 0;
  Record::Base *Bases = nullptr;
  unsigned NumVBases = 0;
  Record::Base *VBases = nullptr;
  if (const auto *CD = dyn_cast<CXXRecordDecl>(RD)) {
    NumBases = CD->getNumBases();
    // NB: This overallocates by all explicitly specified virtual bases.
    if (NumBases != 0)
      Bases = Allocate<Record::Base>(NumBases);

    unsigned I = 0;
    for (const CXXBaseSpecifier &Spec : CD->bases()) {
      assert(I <= NumBases);
      if (Spec.isVirtual())
        continue;

      // In error cases, the base might not be a RecordType.
      const auto *BD = Spec.getType()->getAsCXXRecordDecl();
      if (!BD)
        return nullptr;
      const Record *BR = getOrCreateRecord(BD);

      const Descriptor *Desc = GetBaseDesc(BD, BR);
      if (!Desc)
        return nullptr;

      BaseSize += align(sizeof(InlineDescriptor));
      new (&Bases[I]) Record::Base(BD, Desc, BR, BaseSize);
      BaseSize += align(BR->getSize());
      HasPtrField |= BR->hasPtrField();
      ++I;
    }
    // Make sure we don't include the virtual base specifiers we skipped above.
    NumBases = I;

    I = 0;
    NumVBases = CD->getNumVBases();
    if (NumVBases != 0)
      VBases = Allocate<Record::Base>(NumVBases);
    for (const CXXBaseSpecifier &Spec : CD->vbases()) {
      assert(I <= NumVBases);
      const auto *BD = Spec.getType()->castAsCXXRecordDecl();
      const Record *BR = getOrCreateRecord(BD);

      const Descriptor *Desc = GetBaseDesc(BD, BR);
      if (!Desc)
        return nullptr;

      VirtSize += align(sizeof(InlineDescriptor));
      new (&VBases[I]) Record::Base(BD, Desc, BR, VirtSize);
      VirtSize += align(BR->getSize());
      HasPtrField |= BR->hasPtrField();
      ++I;
    }
    assert(I == NumVBases);
  }

  // Reserve space for fields.
  unsigned NumFields = RD->getNumFields();
  Record::Field *Fields = nullptr;
  if (NumFields != 0)
    Fields = Allocate<Record::Field>(NumFields);
  unsigned I = 0;
  for (const FieldDecl *FD : RD->fields()) {
    FD = FD->getFirstDecl();
    // Note that we DO create fields and descriptors
    // for unnamed bitfields here, even though we later ignore
    // them everywhere. That's so the FieldDecl's getFieldIndex() matches.

    // Reserve space for the field's descriptor and the offset.
    BaseSize += align(sizeof(InlineDescriptor));

    // Classify the field and add its metadata.
    QualType FT = FD->getType();
    const bool IsConst = FT.isConstQualified();
    const bool IsMutable = FD->isMutable();
    const bool IsVolatile = FT.isVolatileQualified();
    const Descriptor *Desc;
    OptPrimType T = Ctx.classify(FT);
    if (T) {
      Desc = createDescriptor(FD, *T, nullptr, IsConst,
                              /*IsTemporary=*/false, IsMutable, IsVolatile);
      HasPtrField = HasPtrField || (T == PT_Ptr);
    } else if ((Desc = createDescriptor(FD, FT.getTypePtr(), IsConst,
                                        /*IsTemporary=*/false, IsMutable,
                                        IsVolatile))) {
      HasPtrField =
          HasPtrField ||
          (Desc->isPrimitiveArray() && Desc->getPrimType() == PT_Ptr) ||
          (Desc->ElemRecord && Desc->ElemRecord->hasPtrField());
    } else {
      Desc = allocateDescriptor(FD);
    }
    assert(Desc);
    new (&Fields[I]) Record::Field(FD, Desc, BaseSize, T);
    BaseSize += align(Desc->getAllocSize());
    ++I;
  }

  // Adjust virtual base offsets to account for base size.
  for (unsigned I = 0; I != NumVBases; ++I)
    VBases[I].Offset += BaseSize;

  Record *R = new (Allocator)
      Record(RD, {Bases, NumBases}, {Fields, NumFields}, {VBases, NumVBases},
             VirtSize, BaseSize, HasPtrField);
  Records[RD] = R;
  return R;
}

// The expression we use for the filler can only work if the computed value for
// it is independent of its position in the array. We cannot really *know* if an
// expression has this propery, so we're conservative here.
//
// To summarize, we only allow ImplicitValueInitExprs, or InitListExprs
// consisting of such.
static bool maybeDependentArrayFiller(const Expr *Filler) {
  assert(Filler);

  if (isa<ImplicitValueInitExpr>(Filler))
    return false;
  if (const auto *ILE = dyn_cast<InitListExpr>(Filler)) {
    for (unsigned I = 0, N = ILE->getNumInits(); I != N; ++I) {
      if (maybeDependentArrayFiller(ILE->getInit(I)))
        return true;
    }

    if (const Expr *F = ILE->getArrayFiller())
      return maybeDependentArrayFiller(F);
    return false;
  }
  return true;
}

/// Returns how many of the \p NumElems elements of an array initialized by
/// \p Init need storage. Elements initialized by a zero filler share one
/// element, e.g. `int A[100] = {1, 2};` stores 3 elements.
///
/// The returned value includes the filler element.
static size_t getNumStoredElems(const VarDecl *VD, size_t NumElems,
                                const Expr *Init) {
  assert(Init);
  // Only global variables.
  if (!Context::shouldBeGloballyIndexed(VD))
    return NumElems;

  // char foo[1000] = "abc"
  if (const auto *SL = dyn_cast<StringLiteral>(Init))
    return std::min<size_t>(SL->getLength() + 1, NumElems);

  const auto *ILE = dyn_cast<InitListExpr>(Init);
  if (!ILE)
    return NumElems;

  // char foo[1000] = {"abc"}
  if (ILE->isStringLiteralInit()) {
    // ObjCEncodeExpr can also appear here.
    if (const auto *SL = dyn_cast<StringLiteral>(ILE->getInit(0)))
      return std::min<size_t>(SL->getLength() + 1, NumElems);
    return NumElems;
  }

  // An EmbedExpr initializes more than one element.
  if (llvm::any_of(ILE->inits(), [](const Expr *E) {
        return isa<EmbedExpr>(E->IgnoreParenImpCasts());
      }))
    return NumElems;

  assert(ILE->getType()->isArrayType());
  const Expr *Filler = ILE->getArrayFiller();
  if (!Filler)
    return NumElems;

  if (maybeDependentArrayFiller(Filler))
    return NumElems;

  return std::min<size_t>(ILE->getNumInits() + 1, NumElems);
}

Descriptor *Program::createDescriptor(DeclOrExpr D, const Type *Ty,
                                      bool IsConst, bool IsTemporary,
                                      bool IsMutable, bool IsVolatile,
                                      const Expr *Init) {
  // Classes and structures.
  if (const auto *RD = Ty->getAsRecordDecl()) {
    if (const auto *Record = getOrCreateRecord(RD))
      return allocateDescriptor(D, Record, IsConst, IsTemporary, IsMutable,
                                IsVolatile);
    return allocateDescriptor(D);
  }

  // Arrays.
  if (const auto *ArrayType = Ty->getAsArrayTypeUnsafe()) {
    QualType ElemTy = ArrayType->getElementType();
    // Array of well-known bounds.
    if (const auto *CAT = dyn_cast<ConstantArrayType>(ArrayType)) {
      size_t NumElems = CAT->getZExtSize();
      // The actual number of elements we will allocate storage for.
      size_t NumStoredElems = NumElems;
      if (Init && D.isVarDecl())
        NumStoredElems = getNumStoredElems(D.asVarDecl(), NumElems, Init);

      // This only relevant for tests/SemaCXX/large-array-init.cpp.
      LLVM_DEBUG(llvm::dbgs() << "The number of elements to initialize: "
                              << (NumStoredElems - 1) << ".\n");

      if (OptPrimType T = Ctx.classify(ElemTy)) {
        // Arrays of primitives.
        unsigned ElemSize = primSize(*T);
        if ((Descriptor::MaxArrayElemBytes / ElemSize) < NumElems) {
          return nullptr;
        }
        return allocateDescriptor(D, CAT, *T, NumElems, NumStoredElems, IsConst,
                                  IsTemporary, IsMutable, IsVolatile);
      }
        // Arrays of composites. In this case, the array is a list of pointers,
        // followed by the actual elements.
      const Descriptor *ElemDesc =
          createDescriptor(D, ElemTy.getTypePtr(), IsConst, IsTemporary);
      if (!ElemDesc)
        return nullptr;
      unsigned ElemSize = ElemDesc->getAllocSize() + sizeof(InlineDescriptor);
      if (std::numeric_limits<unsigned>::max() / ElemSize <= NumElems)
        return nullptr;
      return allocateDescriptor(D, Ty, ElemDesc, NumElems, NumStoredElems,
                                IsConst, IsTemporary, IsMutable);
    }

    // Array of unknown bounds - cannot be accessed and pointer arithmetic
    // is forbidden on pointers to such objects.
    if (isa<IncompleteArrayType>(ArrayType) ||
        isa<VariableArrayType>(ArrayType)) {
      if (OptPrimType T = Ctx.classify(ElemTy)) {
        return allocateDescriptor(D, *T, IsConst, IsTemporary,
                                  Descriptor::UnknownSize{});
      }
      const Descriptor *Desc =
          createDescriptor(D, ElemTy.getTypePtr(), IsConst, IsTemporary);
      if (!Desc)
        return nullptr;
      return allocateDescriptor(D, Desc, IsTemporary,
                                Descriptor::UnknownSize{});
    }
  }

  // Atomic types.
  if (const auto *AT = Ty->getAs<AtomicType>()) {
    const Type *InnerTy = AT->getValueType().getTypePtr();
    return createDescriptor(D, InnerTy, IsConst, IsTemporary, IsMutable);
  }

  // Complex types - represented as arrays of elements.
  if (const auto *CT = Ty->getAs<ComplexType>()) {
    OptPrimType ElemTy = Ctx.classify(CT->getElementType());
    if (!ElemTy)
      return nullptr;

    return allocateDescriptor(D, CT, *ElemTy, 2, 2, IsConst, IsTemporary,
                              IsMutable, IsVolatile);
  }

  // Same with vector types.
  if (const auto *VT = Ty->getAs<VectorType>()) {
    OptPrimType ElemTy = Ctx.classify(VT->getElementType());
    if (!ElemTy)
      return nullptr;

    unsigned NumElems = VT->getNumElements();
    return allocateDescriptor(D, VT, *ElemTy, NumElems, NumElems, IsConst,
                              IsTemporary, IsMutable, IsVolatile);
  }

  // Same with constant matrix types.
  if (const auto *MT = Ty->getAs<ConstantMatrixType>()) {
    OptPrimType ElemTy = Ctx.classify(MT->getElementType());
    if (!ElemTy)
      return nullptr;

    unsigned NumElems = MT->getNumElementsFlattened();
    return allocateDescriptor(D, MT, *ElemTy, NumElems, NumElems, IsConst,
                              IsTemporary, IsMutable, IsVolatile);
  }

  return nullptr;
}
