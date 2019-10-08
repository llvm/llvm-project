//===--- Program.cpp - Bytecode for the constexpr VM ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Program.h"
#include "ByteCodeStmtGen.h"
#include "Context.h"
#include "Function.h"
#include "Opcode.h"
#include "PrimType.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::interp;

unsigned Program::createGlobalString(const StringLiteral *S) {
  const size_t CharWidth = S->getCharByteWidth();
  const size_t BitWidth = CharWidth * Ctx.getCharBit();

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

  // Create a descriptor for the string.
  Descriptor *Desc = allocateDescriptor(S, CharType, S->getLength() + 1,
                                        /*isConst=*/true,
                                        /*isTemporary=*/false,
                                        /*isMutable=*/false);

  // Allocate storage for the string.
  // The byte length does not include the null terminator.
  unsigned I = Globals.size();
  unsigned Sz = Desc->getAllocSize();
  auto *G = new (Allocator, Sz) Global(Desc, /*isStatic=*/true,
                                       /*isExtern=*/false);
  Globals.push_back(G);

  // Construct the string in storage.
  const Pointer Ptr(G->block());
  for (unsigned I = 0, N = S->getLength(); I <= N; ++I) {
    Pointer Field = Ptr.atIndex(I).narrow();
    const uint32_t CodePoint = I == N ? 0 : S->getCodeUnit(I);
    switch (CharType) {
      case PT_Sint8: {
        using T = PrimConv<PT_Sint8>::T;
        Field.deref<T>() = T::from(CodePoint, BitWidth);
        break;
      }
      case PT_Uint16: {
        using T = PrimConv<PT_Uint16>::T;
        Field.deref<T>() = T::from(CodePoint, BitWidth);
        break;
      }
      case PT_Uint32: {
        using T = PrimConv<PT_Uint32>::T;
        Field.deref<T>() = T::from(CodePoint, BitWidth);
        break;
      }
      default:
        llvm_unreachable("unsupported character type");
    }
  }
  return I;
}

Pointer Program::getPtrGlobal(unsigned Idx) {
  assert(Idx < Globals.size());
  return Pointer(Globals[Idx]->block());
}

llvm::Optional<unsigned> Program::getGlobal(const ValueDecl *VD) {
  auto It = GlobalIndices.find(VD);
  if (It != GlobalIndices.end())
    return It->second;

  // Find any previous declarations which were aleady evaluated.
  llvm::Optional<unsigned> Index;
  for (const Decl *P = VD; P; P = P->getPreviousDecl()) {
    auto It = GlobalIndices.find(P);
    if (It != GlobalIndices.end()) {
      Index = It->second;
      break;
    }
  }

  // Map the decl to the existing index.
  if (Index) {
    GlobalIndices[VD] = *Index;
    return {};
  }

  return Index;
}

llvm::Optional<unsigned> Program::getOrCreateGlobal(const ValueDecl *VD) {
  if (auto Idx = getGlobal(VD))
    return Idx;

  if (auto Idx = createGlobal(VD)) {
    GlobalIndices[VD] = *Idx;
    return Idx;
  }
  return {};
}

llvm::Optional<unsigned> Program::getOrCreateDummy(const ParmVarDecl *PD) {
  auto &ASTCtx = Ctx.getASTContext();

  // Create a pointer to an incomplete array of the specified elements.
  QualType ElemTy = PD->getType()->castAs<PointerType>()->getPointeeType();
  QualType Ty = ASTCtx.getIncompleteArrayType(ElemTy, ArrayType::Normal, 0);

  // Dedup blocks since they are immutable and pointers cannot be compared.
  auto It = DummyParams.find(PD);
  if (It != DummyParams.end())
    return It->second;

  if (auto Idx = createGlobal(PD, Ty, /*isStatic=*/true, /*isExtern=*/true)) {
    DummyParams[PD] = *Idx;
    return Idx;
  }
  return {};
}

llvm::Optional<unsigned> Program::createGlobal(const ValueDecl *VD) {
  bool IsStatic, IsExtern;
  if (auto *Var = dyn_cast<VarDecl>(VD)) {
    IsStatic = !Var->hasLocalStorage();
    IsExtern = !Var->getAnyInitializer();
  } else {
    IsStatic = false;
    IsExtern = true;
  }
  if (auto Idx = createGlobal(VD, VD->getType(), IsStatic, IsExtern)) {
    for (const Decl *P = VD; P; P = P->getPreviousDecl())
      GlobalIndices[P] = *Idx;
    return *Idx;
  }
  return {};
}

llvm::Optional<unsigned> Program::createGlobal(const Expr *E) {
  return createGlobal(E, E->getType(), /*isStatic=*/true, /*isExtern=*/false);
}

llvm::Optional<unsigned> Program::createGlobal(const DeclTy &D, QualType Ty,
                                               bool IsStatic, bool IsExtern) {
  // Create a descriptor for the global.
  Descriptor *Desc;
  const bool IsConst = Ty.isConstQualified();
  const bool IsTemporary = D.dyn_cast<const Expr *>();
  if (auto T = Ctx.classify(Ty)) {
    Desc = createDescriptor(D, *T, IsConst, IsTemporary);
  } else {
    Desc = createDescriptor(D, Ty.getTypePtr(), IsConst, IsTemporary);
  }
  if (!Desc)
    return {};

  // Allocate a block for storage.
  unsigned I = Globals.size();

  auto *G = new (Allocator, Desc->getAllocSize())
      Global(getCurrentDecl(), Desc, IsStatic, IsExtern);
  G->block()->invokeCtor();

  Globals.push_back(G);

  return I;
}

Function *Program::getFunction(const FunctionDecl *F) {
  F = F->getDefinition();
  auto It = Funcs.find(F);
  return It == Funcs.end() ? nullptr : It->second.get();
}

llvm::Expected<Function *> Program::getOrCreateFunction(const FunctionDecl *F) {
  if (Function *Func = getFunction(F)) {
    return Func;
  }

  // Try to compile the function if it wasn't compiled yet.
  if (const FunctionDecl *FD = F->getDefinition())
    return ByteCodeStmtGen<ByteCodeEmitter>(Ctx, *this).compileFunc(FD);

  // A relocation which traps if not resolved.
  return nullptr;
}

Record *Program::getOrCreateRecord(const RecordDecl *RD) {
  // Use the actual definition as a key.
  RD = RD->getDefinition();
  if (!RD)
    return nullptr;

  // Deduplicate records.
  auto It = Records.find(RD);
  if (It != Records.end()) {
    return It->second;
  }

  // Number of bytes required by fields and base classes.
  unsigned Size = 0;
  // Number of bytes required by virtual base.
  unsigned VirtSize = 0;

  // Helper to get a base descriptor.
  auto GetBaseDesc = [this](const RecordDecl *BD, Record *BR) -> Descriptor * {
    if (!BR)
      return nullptr;
    return allocateDescriptor(BD, BR, /*isConst=*/false,
                              /*isTemporary=*/false,
                              /*isMutable=*/false);
  };

  // Reserve space for base classes.
  Record::BaseList Bases;
  Record::VirtualBaseList VirtBases;
  if (auto *CD = dyn_cast<CXXRecordDecl>(RD)) {
    for (const CXXBaseSpecifier &Spec : CD->bases()) {
      if (Spec.isVirtual())
        continue;

      const RecordDecl *BD = Spec.getType()->castAs<RecordType>()->getDecl();
      Record *BR = getOrCreateRecord(BD);
      if (Descriptor *Desc = GetBaseDesc(BD, BR)) {
        Size += align(sizeof(InlineDescriptor));
        Bases.push_back({BD, Size, Desc, BR});
        Size += align(BR->getSize());
        continue;
      }
      return nullptr;
    }

    for (const CXXBaseSpecifier &Spec : CD->vbases()) {
      const RecordDecl *BD = Spec.getType()->castAs<RecordType>()->getDecl();
      Record *BR = getOrCreateRecord(BD);

      if (Descriptor *Desc = GetBaseDesc(BD, BR)) {
        VirtSize += align(sizeof(InlineDescriptor));
        VirtBases.push_back({BD, VirtSize, Desc, BR});
        VirtSize += align(BR->getSize());
        continue;
      }
      return nullptr;
    }
  }

  // Reserve space for fields.
  Record::FieldList Fields;
  for (const FieldDecl *FD : RD->fields()) {
    // Reserve space for the field's descriptor and the offset.
    Size += align(sizeof(InlineDescriptor));

    // Classify the field and add its metadata.
    QualType FT = FD->getType();
    const bool IsConst = FT.isConstQualified();
    const bool IsMutable = FD->isMutable();
    Descriptor *Desc;
    if (llvm::Optional<PrimType> T = Ctx.classify(FT)) {
      Desc = createDescriptor(FD, *T, IsConst, /*isTemporary=*/false,
                              IsMutable);
    } else {
      Desc = createDescriptor(FD, FT.getTypePtr(), IsConst,
                              /*isTemporary=*/false, IsMutable);
    }
    if (!Desc)
      return nullptr;
    Fields.push_back({FD, Size, Desc});
    Size += align(Desc->getAllocSize());
  }

  Record *R = new (Allocator) Record(RD, std::move(Bases), std::move(Fields),
                                     std::move(VirtBases), VirtSize, Size);
  Records.insert({RD, R});
  return R;
}

Descriptor *Program::createDescriptor(const DeclTy &D, const Type *Ty,
                                      bool IsConst, bool IsTemporary,
                                      bool IsMutable) {
  // Classes and structures.
  if (auto *RT = Ty->getAs<RecordType>()) {
    if (auto *Record = getOrCreateRecord(RT->getDecl()))
      return allocateDescriptor(D, Record, IsConst, IsTemporary, IsMutable);
  }

  // Arrays.
  if (auto ArrayType = Ty->getAsArrayTypeUnsafe()) {
    QualType ElemTy = ArrayType->getElementType();
    // Array of well-known bounds.
    if (auto CAT = dyn_cast<ConstantArrayType>(ArrayType)) {
      size_t NumElems = CAT->getSize().getZExtValue();
      if (llvm::Optional<PrimType> T = Ctx.classify(ElemTy)) {
        // Arrays of primitives.
        unsigned ElemSize = primSize(*T);
        if (std::numeric_limits<unsigned>::max() / ElemSize <= NumElems) {
          return {};
        }
        return allocateDescriptor(D, *T, NumElems, IsConst, IsTemporary,
                                  IsMutable);
      } else {
        // Arrays of composites. In this case, the array is a list of pointers,
        // followed by the actual elements.
        Descriptor *Desc =
            createDescriptor(D, ElemTy.getTypePtr(), IsConst, IsTemporary);
        if (!Desc)
          return nullptr;
        InterpSize ElemSize = Desc->getAllocSize() + sizeof(InlineDescriptor);
        if (std::numeric_limits<unsigned>::max() / ElemSize <= NumElems)
          return {};
        return allocateDescriptor(D, Desc, NumElems, IsConst, IsTemporary,
                                  IsMutable);
      }
    }

    // Array of unknown bounds - cannot be accessed and pointer arithmetic
    // is forbidden on pointers to such objects.
    if (isa<IncompleteArrayType>(ArrayType)) {
      if (llvm::Optional<PrimType> T = Ctx.classify(ElemTy)) {
        return allocateDescriptor(D, *T, IsTemporary,
                                  Descriptor::UnknownSize{});
      } else {
        Descriptor *Desc =
            createDescriptor(D, ElemTy.getTypePtr(), IsConst, IsTemporary);
        if (!Desc)
          return nullptr;
        return allocateDescriptor(D, Desc, IsTemporary,
                                  Descriptor::UnknownSize{});
      }
    }
  }

  // Atomic types.
  if (auto *AT = Ty->getAs<AtomicType>()) {
    const Type *InnerTy = AT->getValueType().getTypePtr();
    return createDescriptor(D, InnerTy, IsConst, IsTemporary, IsMutable);
  }

  // Complex types - represented as arrays of elements.
  if (auto *CT = Ty->getAs<ComplexType>()) {
    PrimType ElemTy = *Ctx.classify(CT->getElementType());
    return allocateDescriptor(D, ElemTy, 2, IsConst, IsTemporary, IsMutable);
  }

  return nullptr;
}
