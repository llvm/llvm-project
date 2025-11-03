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
#include "Integral.h"
#include "PrimType.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"

using namespace clang;
using namespace clang::interp;

unsigned Program::getOrCreateNativePointer(const void *Ptr) {
  auto [It, Inserted] =
      NativePointerIndices.try_emplace(Ptr, NativePointers.size());
  if (Inserted)
    NativePointers.push_back(Ptr);

  return It->second;
}

const void *Program::getNativePointer(unsigned Idx) {
  return NativePointers[Idx];
}

unsigned Program::createGlobalString(const StringLiteral *S, const Expr *Base) {
  const size_t CharWidth = S->getCharByteWidth();
  const size_t BitWidth = CharWidth * Ctx.getCharBit();
  unsigned StringLength = S->getLength();

  PrimType CharType;
  switch (CharWidth) {
  case 1:
    CharType = PT_Sint8;
    break;
  case 2:
    CharType = PT_Uint16;
    break;
  case 4:
    CharType = PT_Uint32;
    break;
  default:
    llvm_unreachable("unsupported character width");
  }

  if (!Base)
    Base = S;

  // Create a descriptor for the string.
  Descriptor *Desc =
      allocateDescriptor(Base, CharType, Descriptor::GlobalMD, StringLength + 1,
                         /*isConst=*/true,
                         /*isTemporary=*/false,
                         /*isMutable=*/false);

  // Allocate storage for the string.
  // The byte length does not include the null terminator.
  unsigned GlobalIndex = Globals.size();
  unsigned Sz = Desc->getAllocSize();
  auto *G = new (Allocator, Sz) Global(Ctx.getEvalID(), Desc, /*isStatic=*/true,
                                       /*isExtern=*/false);
  G->block()->invokeCtor();

  new (G->block()->rawData())
      GlobalInlineDescriptor{GlobalInitState::Initialized};
  Globals.push_back(G);

  const Pointer Ptr(G->block());
  if (CharWidth == 1) {
    std::memcpy(&Ptr.elem<char>(0), S->getString().data(), StringLength);
  } else {
    // Construct the string in storage.
    for (unsigned I = 0; I <= StringLength; ++I) {
      const uint32_t CodePoint = I == StringLength ? 0 : S->getCodeUnit(I);
      switch (CharType) {
      case PT_Sint8: {
        using T = PrimConv<PT_Sint8>::T;
        Ptr.elem<T>(I) = T::from(CodePoint, BitWidth);
        break;
      }
      case PT_Uint16: {
        using T = PrimConv<PT_Uint16>::T;
        Ptr.elem<T>(I) = T::from(CodePoint, BitWidth);
        break;
      }
      case PT_Uint32: {
        using T = PrimConv<PT_Uint32>::T;
        Ptr.elem<T>(I) = T::from(CodePoint, BitWidth);
        break;
      }
      default:
        llvm_unreachable("unsupported character type");
      }
    }
  }
  Ptr.initializeAllElements();

  return GlobalIndex;
}

Pointer Program::getPtrGlobal(unsigned Idx) const {
  assert(Idx < Globals.size());
  return Pointer(Globals[Idx]->block());
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

unsigned Program::getOrCreateDummy(const DeclTy &D) {
  assert(D);
  // Dedup blocks since they are immutable and pointers cannot be compared.
  if (auto It = DummyVariables.find(D.getOpaqueValue());
      It != DummyVariables.end())
    return It->second;

  QualType QT;
  bool IsWeak = false;
  if (const auto *E = dyn_cast<const Expr *>(D)) {
    QT = E->getType();
  } else {
    const auto *VD = cast<ValueDecl>(cast<const Decl *>(D));
    IsWeak = VD->isWeak();
    QT = VD->getType();
    if (QT->isPointerOrReferenceType())
      QT = QT->getPointeeType();
  }
  assert(!QT.isNull());

  Descriptor *Desc;
  if (OptPrimType T = Ctx.classify(QT))
    Desc = createDescriptor(D, *T, /*SourceTy=*/nullptr, std::nullopt,
                            /*IsConst=*/QT.isConstQualified());
  else
    Desc = createDescriptor(D, QT.getTypePtr(), std::nullopt,
                            /*IsConst=*/QT.isConstQualified());
  if (!Desc)
    Desc = allocateDescriptor(D);

  assert(Desc);

  // Allocate a block for storage.
  unsigned I = Globals.size();

  auto *G = new (Allocator, Desc->getAllocSize())
      Global(Ctx.getEvalID(), getCurrentDecl(), Desc, /*IsStatic=*/true,
             /*IsExtern=*/false, IsWeak, /*IsDummy=*/true);
  G->block()->invokeCtor();
  assert(G->block()->isDummy());

  Globals.push_back(G);
  DummyVariables[D.getOpaqueValue()] = I;
  return I;
}

UnsignedOrNone Program::createGlobal(const ValueDecl *VD, const Expr *Init) {
  bool IsStatic, IsExtern;
  bool IsWeak = VD->isWeak();
  if (const auto *Var = dyn_cast<VarDecl>(VD)) {
    IsStatic = Context::shouldBeGloballyIndexed(VD);
    IsExtern = Var->hasExternalStorage();
  } else if (isa<UnnamedGlobalConstantDecl, MSGuidDecl,
                 TemplateParamObjectDecl>(VD)) {
    IsStatic = true;
    IsExtern = false;
  } else {
    IsStatic = false;
    IsExtern = true;
  }

  // Register all previous declarations as well. For extern blocks, just replace
  // the index with the new variable.
  UnsignedOrNone Idx =
      createGlobal(VD, VD->getType(), IsStatic, IsExtern, IsWeak, Init);
  if (!Idx)
    return std::nullopt;

  Global *NewGlobal = Globals[*Idx];
  for (const Decl *Redecl : VD->redecls()) {
    unsigned &PIdx = GlobalIndices[Redecl];
    if (Redecl != VD) {
      if (Block *RedeclBlock = Globals[PIdx]->block();
          RedeclBlock->isExtern()) {
        Globals[PIdx] = NewGlobal;
        // All pointers pointing to the previous extern decl now point to the
        // new decl.
        // A previous iteration might've already fixed up the pointers for this
        // global.
        if (RedeclBlock != NewGlobal->block())
          RedeclBlock->movePointersTo(NewGlobal->block());
      }
    }
    PIdx = *Idx;
  }

  return *Idx;
}

UnsignedOrNone Program::createGlobal(const Expr *E) {
  if (auto Idx = getGlobal(E))
    return Idx;
  if (auto Idx = createGlobal(E, E->getType(), /*isStatic=*/true,
                              /*isExtern=*/false, /*IsWeak=*/false)) {
    GlobalIndices[E] = *Idx;
    return *Idx;
  }
  return std::nullopt;
}

UnsignedOrNone Program::createGlobal(const DeclTy &D, QualType Ty,
                                     bool IsStatic, bool IsExtern, bool IsWeak,
                                     const Expr *Init) {
  // Create a descriptor for the global.
  Descriptor *Desc;
  const bool IsConst = Ty.isConstQualified();
  const bool IsTemporary = D.dyn_cast<const Expr *>();
  const bool IsVolatile = Ty.isVolatileQualified();
  if (OptPrimType T = Ctx.classify(Ty))
    Desc = createDescriptor(D, *T, nullptr, Descriptor::GlobalMD, IsConst,
                            IsTemporary, /*IsMutable=*/false, IsVolatile);
  else
    Desc = createDescriptor(D, Ty.getTypePtr(), Descriptor::GlobalMD, IsConst,
                            IsTemporary, /*IsMutable=*/false, IsVolatile);

  if (!Desc)
    return std::nullopt;

  // Allocate a block for storage.
  unsigned I = Globals.size();

  auto *G = new (Allocator, Desc->getAllocSize()) Global(
      Ctx.getEvalID(), getCurrentDecl(), Desc, IsStatic, IsExtern, IsWeak);
  G->block()->invokeCtor();

  // Initialize GlobalInlineDescriptor fields.
  auto *GD = new (G->block()->rawData()) GlobalInlineDescriptor();
  if (!Init)
    GD->InitState = GlobalInitState::NoInitializer;
  Globals.push_back(G);

  return I;
}

Function *Program::getFunction(const FunctionDecl *F) {
  F = F->getCanonicalDecl();
  assert(F);
  auto It = Funcs.find(F);
  return It == Funcs.end() ? nullptr : It->second.get();
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
    return allocateDescriptor(BD, BR, std::nullopt, /*isConst=*/false,
                              /*isTemporary=*/false,
                              /*isMutable=*/false, /*IsVolatile=*/false);
  };

  // Reserve space for base classes.
  Record::BaseList Bases;
  Record::VirtualBaseList VirtBases;
  if (const auto *CD = dyn_cast<CXXRecordDecl>(RD)) {
    for (const CXXBaseSpecifier &Spec : CD->bases()) {
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
      Bases.push_back({BD, BaseSize, Desc, BR});
      BaseSize += align(BR->getSize());
    }

    for (const CXXBaseSpecifier &Spec : CD->vbases()) {
      const auto *BD = Spec.getType()->castAsCXXRecordDecl();
      const Record *BR = getOrCreateRecord(BD);

      const Descriptor *Desc = GetBaseDesc(BD, BR);
      if (!Desc)
        return nullptr;

      VirtSize += align(sizeof(InlineDescriptor));
      VirtBases.push_back({BD, VirtSize, Desc, BR});
      VirtSize += align(BR->getSize());
    }
  }

  // Reserve space for fields.
  Record::FieldList Fields;
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
    if (OptPrimType T = Ctx.classify(FT)) {
      Desc = createDescriptor(FD, *T, nullptr, std::nullopt, IsConst,
                              /*isTemporary=*/false, IsMutable, IsVolatile);
    } else {
      Desc = createDescriptor(FD, FT.getTypePtr(), std::nullopt, IsConst,
                              /*isTemporary=*/false, IsMutable, IsVolatile);
    }
    if (!Desc)
      return nullptr;
    Fields.push_back({FD, BaseSize, Desc});
    BaseSize += align(Desc->getAllocSize());
  }

  Record *R = new (Allocator) Record(RD, std::move(Bases), std::move(Fields),
                                     std::move(VirtBases), VirtSize, BaseSize);
  Records[RD] = R;
  return R;
}

Descriptor *Program::createDescriptor(const DeclTy &D, const Type *Ty,
                                      Descriptor::MetadataSize MDSize,
                                      bool IsConst, bool IsTemporary,
                                      bool IsMutable, bool IsVolatile,
                                      const Expr *Init) {

  // Classes and structures.
  if (const auto *RD = Ty->getAsRecordDecl()) {
    if (const auto *Record = getOrCreateRecord(RD))
      return allocateDescriptor(D, Record, MDSize, IsConst, IsTemporary,
                                IsMutable, IsVolatile);
    return allocateDescriptor(D, MDSize);
  }

  // Arrays.
  if (const auto *ArrayType = Ty->getAsArrayTypeUnsafe()) {
    QualType ElemTy = ArrayType->getElementType();
    // Array of well-known bounds.
    if (const auto *CAT = dyn_cast<ConstantArrayType>(ArrayType)) {
      size_t NumElems = CAT->getZExtSize();
      if (OptPrimType T = Ctx.classify(ElemTy)) {
        // Arrays of primitives.
        unsigned ElemSize = primSize(*T);
        if (std::numeric_limits<unsigned>::max() / ElemSize <= NumElems) {
          return {};
        }
        return allocateDescriptor(D, *T, MDSize, NumElems, IsConst, IsTemporary,
                                  IsMutable);
      }
        // Arrays of composites. In this case, the array is a list of pointers,
        // followed by the actual elements.
        const Descriptor *ElemDesc = createDescriptor(
            D, ElemTy.getTypePtr(), std::nullopt, IsConst, IsTemporary);
        if (!ElemDesc)
          return nullptr;
        unsigned ElemSize = ElemDesc->getAllocSize() + sizeof(InlineDescriptor);
        if (std::numeric_limits<unsigned>::max() / ElemSize <= NumElems)
          return {};
        return allocateDescriptor(D, Ty, ElemDesc, MDSize, NumElems, IsConst,
                                  IsTemporary, IsMutable);
    }

    // Array of unknown bounds - cannot be accessed and pointer arithmetic
    // is forbidden on pointers to such objects.
    if (isa<IncompleteArrayType>(ArrayType) ||
        isa<VariableArrayType>(ArrayType)) {
      if (OptPrimType T = Ctx.classify(ElemTy)) {
        return allocateDescriptor(D, *T, MDSize, IsConst, IsTemporary,
                                  Descriptor::UnknownSize{});
      }
        const Descriptor *Desc = createDescriptor(
            D, ElemTy.getTypePtr(), std::nullopt, IsConst, IsTemporary);
        if (!Desc)
          return nullptr;
        return allocateDescriptor(D, Desc, MDSize, IsTemporary,
                                  Descriptor::UnknownSize{});
    }
  }

  // Atomic types.
  if (const auto *AT = Ty->getAs<AtomicType>()) {
    const Type *InnerTy = AT->getValueType().getTypePtr();
    return createDescriptor(D, InnerTy, MDSize, IsConst, IsTemporary,
                            IsMutable);
  }

  // Complex types - represented as arrays of elements.
  if (const auto *CT = Ty->getAs<ComplexType>()) {
    OptPrimType ElemTy = Ctx.classify(CT->getElementType());
    if (!ElemTy)
      return nullptr;

    return allocateDescriptor(D, *ElemTy, MDSize, 2, IsConst, IsTemporary,
                              IsMutable);
  }

  // Same with vector types.
  if (const auto *VT = Ty->getAs<VectorType>()) {
    OptPrimType ElemTy = Ctx.classify(VT->getElementType());
    if (!ElemTy)
      return nullptr;

    return allocateDescriptor(D, *ElemTy, MDSize, VT->getNumElements(), IsConst,
                              IsTemporary, IsMutable);
  }

  return nullptr;
}
