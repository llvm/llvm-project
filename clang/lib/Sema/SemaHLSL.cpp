//===- SemaHLSL.cpp - Semantic Analysis for HLSL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for HLSL constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaHLSL.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Frontend/HLSL/HLSLBinding.h"
#include "llvm/Frontend/HLSL/RootSignatureValidations.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DXILABI.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/Triple.h"
#include <cmath>
#include <cstddef>
#include <iterator>
#include <utility>

using namespace clang;
using RegisterType = HLSLResourceBindingAttr::RegisterType;

static CXXRecordDecl *createHostLayoutStruct(Sema &S,
                                             CXXRecordDecl *StructDecl);

static RegisterType getRegisterType(ResourceClass RC) {
  switch (RC) {
  case ResourceClass::SRV:
    return RegisterType::SRV;
  case ResourceClass::UAV:
    return RegisterType::UAV;
  case ResourceClass::CBuffer:
    return RegisterType::CBuffer;
  case ResourceClass::Sampler:
    return RegisterType::Sampler;
  }
  llvm_unreachable("unexpected ResourceClass value");
}

static RegisterType getRegisterType(const HLSLAttributedResourceType *ResTy) {
  return getRegisterType(ResTy->getAttrs().ResourceClass);
}

// Converts the first letter of string Slot to RegisterType.
// Returns false if the letter does not correspond to a valid register type.
static bool convertToRegisterType(StringRef Slot, RegisterType *RT) {
  assert(RT != nullptr);
  switch (Slot[0]) {
  case 't':
  case 'T':
    *RT = RegisterType::SRV;
    return true;
  case 'u':
  case 'U':
    *RT = RegisterType::UAV;
    return true;
  case 'b':
  case 'B':
    *RT = RegisterType::CBuffer;
    return true;
  case 's':
  case 'S':
    *RT = RegisterType::Sampler;
    return true;
  case 'c':
  case 'C':
    *RT = RegisterType::C;
    return true;
  case 'i':
  case 'I':
    *RT = RegisterType::I;
    return true;
  default:
    return false;
  }
}

static ResourceClass getResourceClass(RegisterType RT) {
  switch (RT) {
  case RegisterType::SRV:
    return ResourceClass::SRV;
  case RegisterType::UAV:
    return ResourceClass::UAV;
  case RegisterType::CBuffer:
    return ResourceClass::CBuffer;
  case RegisterType::Sampler:
    return ResourceClass::Sampler;
  case RegisterType::C:
  case RegisterType::I:
    // Deliberately falling through to the unreachable below.
    break;
  }
  llvm_unreachable("unexpected RegisterType value");
}

static Builtin::ID getSpecConstBuiltinId(const Type *Type) {
  const auto *BT = dyn_cast<BuiltinType>(Type);
  if (!BT) {
    if (!Type->isEnumeralType())
      return Builtin::NotBuiltin;
    return Builtin::BI__builtin_get_spirv_spec_constant_int;
  }

  switch (BT->getKind()) {
  case BuiltinType::Bool:
    return Builtin::BI__builtin_get_spirv_spec_constant_bool;
  case BuiltinType::Short:
    return Builtin::BI__builtin_get_spirv_spec_constant_short;
  case BuiltinType::Int:
    return Builtin::BI__builtin_get_spirv_spec_constant_int;
  case BuiltinType::LongLong:
    return Builtin::BI__builtin_get_spirv_spec_constant_longlong;
  case BuiltinType::UShort:
    return Builtin::BI__builtin_get_spirv_spec_constant_ushort;
  case BuiltinType::UInt:
    return Builtin::BI__builtin_get_spirv_spec_constant_uint;
  case BuiltinType::ULongLong:
    return Builtin::BI__builtin_get_spirv_spec_constant_ulonglong;
  case BuiltinType::Half:
    return Builtin::BI__builtin_get_spirv_spec_constant_half;
  case BuiltinType::Float:
    return Builtin::BI__builtin_get_spirv_spec_constant_float;
  case BuiltinType::Double:
    return Builtin::BI__builtin_get_spirv_spec_constant_double;
  default:
    return Builtin::NotBuiltin;
  }
}

DeclBindingInfo *ResourceBindings::addDeclBindingInfo(const VarDecl *VD,
                                                      ResourceClass ResClass) {
  assert(getDeclBindingInfo(VD, ResClass) == nullptr &&
         "DeclBindingInfo already added");
  assert(!hasBindingInfoForDecl(VD) || BindingsList.back().Decl == VD);
  // VarDecl may have multiple entries for different resource classes.
  // DeclToBindingListIndex stores the index of the first binding we saw
  // for this decl. If there are any additional ones then that index
  // shouldn't be updated.
  DeclToBindingListIndex.try_emplace(VD, BindingsList.size());
  return &BindingsList.emplace_back(VD, ResClass);
}

DeclBindingInfo *ResourceBindings::getDeclBindingInfo(const VarDecl *VD,
                                                      ResourceClass ResClass) {
  auto Entry = DeclToBindingListIndex.find(VD);
  if (Entry != DeclToBindingListIndex.end()) {
    for (unsigned Index = Entry->getSecond();
         Index < BindingsList.size() && BindingsList[Index].Decl == VD;
         ++Index) {
      if (BindingsList[Index].ResClass == ResClass)
        return &BindingsList[Index];
    }
  }
  return nullptr;
}

bool ResourceBindings::hasBindingInfoForDecl(const VarDecl *VD) const {
  return DeclToBindingListIndex.contains(VD);
}

SemaHLSL::SemaHLSL(Sema &S) : SemaBase(S) {}

Decl *SemaHLSL::ActOnStartBuffer(Scope *BufferScope, bool CBuffer,
                                 SourceLocation KwLoc, IdentifierInfo *Ident,
                                 SourceLocation IdentLoc,
                                 SourceLocation LBrace) {
  // For anonymous namespace, take the location of the left brace.
  DeclContext *LexicalParent = SemaRef.getCurLexicalContext();
  HLSLBufferDecl *Result = HLSLBufferDecl::Create(
      getASTContext(), LexicalParent, CBuffer, KwLoc, Ident, IdentLoc, LBrace);

  // if CBuffer is false, then it's a TBuffer
  auto RC = CBuffer ? llvm::hlsl::ResourceClass::CBuffer
                    : llvm::hlsl::ResourceClass::SRV;
  Result->addAttr(HLSLResourceClassAttr::CreateImplicit(getASTContext(), RC));

  SemaRef.PushOnScopeChains(Result, BufferScope);
  SemaRef.PushDeclContext(BufferScope, Result);

  return Result;
}

static unsigned calculateLegacyCbufferFieldAlign(const ASTContext &Context,
                                                 QualType T) {
  // Arrays and Structs are always aligned to new buffer rows
  if (T->isArrayType() || T->isStructureType())
    return 16;

  // Vectors are aligned to the type they contain
  if (const VectorType *VT = T->getAs<VectorType>())
    return calculateLegacyCbufferFieldAlign(Context, VT->getElementType());

  assert(Context.getTypeSize(T) <= 64 &&
         "Scalar bit widths larger than 64 not supported");

  // Scalar types are aligned to their byte width
  return Context.getTypeSize(T) / 8;
}

// Calculate the size of a legacy cbuffer type in bytes based on
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules
static unsigned calculateLegacyCbufferSize(const ASTContext &Context,
                                           QualType T) {
  constexpr unsigned CBufferAlign = 16;
  if (const RecordType *RT = T->getAs<RecordType>()) {
    unsigned Size = 0;
    const RecordDecl *RD = RT->getOriginalDecl()->getDefinitionOrSelf();
    for (const FieldDecl *Field : RD->fields()) {
      QualType Ty = Field->getType();
      unsigned FieldSize = calculateLegacyCbufferSize(Context, Ty);
      unsigned FieldAlign = calculateLegacyCbufferFieldAlign(Context, Ty);

      // If the field crosses the row boundary after alignment it drops to the
      // next row
      unsigned AlignSize = llvm::alignTo(Size, FieldAlign);
      if ((AlignSize % CBufferAlign) + FieldSize > CBufferAlign) {
        FieldAlign = CBufferAlign;
      }

      Size = llvm::alignTo(Size, FieldAlign);
      Size += FieldSize;
    }
    return Size;
  }

  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    unsigned ElementCount = AT->getSize().getZExtValue();
    if (ElementCount == 0)
      return 0;

    unsigned ElementSize =
        calculateLegacyCbufferSize(Context, AT->getElementType());
    unsigned AlignedElementSize = llvm::alignTo(ElementSize, CBufferAlign);
    return AlignedElementSize * (ElementCount - 1) + ElementSize;
  }

  if (const VectorType *VT = T->getAs<VectorType>()) {
    unsigned ElementCount = VT->getNumElements();
    unsigned ElementSize =
        calculateLegacyCbufferSize(Context, VT->getElementType());
    return ElementSize * ElementCount;
  }

  return Context.getTypeSize(T) / 8;
}

// Validate packoffset:
// - if packoffset it used it must be set on all declarations inside the buffer
// - packoffset ranges must not overlap
static void validatePackoffset(Sema &S, HLSLBufferDecl *BufDecl) {
  llvm::SmallVector<std::pair<VarDecl *, HLSLPackOffsetAttr *>> PackOffsetVec;

  // Make sure the packoffset annotations are either on all declarations
  // or on none.
  bool HasPackOffset = false;
  bool HasNonPackOffset = false;
  for (auto *Field : BufDecl->buffer_decls()) {
    VarDecl *Var = dyn_cast<VarDecl>(Field);
    if (!Var)
      continue;
    if (Field->hasAttr<HLSLPackOffsetAttr>()) {
      PackOffsetVec.emplace_back(Var, Field->getAttr<HLSLPackOffsetAttr>());
      HasPackOffset = true;
    } else {
      HasNonPackOffset = true;
    }
  }

  if (!HasPackOffset)
    return;

  if (HasNonPackOffset)
    S.Diag(BufDecl->getLocation(), diag::warn_hlsl_packoffset_mix);

  // Make sure there is no overlap in packoffset - sort PackOffsetVec by offset
  // and compare adjacent values.
  bool IsValid = true;
  ASTContext &Context = S.getASTContext();
  std::sort(PackOffsetVec.begin(), PackOffsetVec.end(),
            [](const std::pair<VarDecl *, HLSLPackOffsetAttr *> &LHS,
               const std::pair<VarDecl *, HLSLPackOffsetAttr *> &RHS) {
              return LHS.second->getOffsetInBytes() <
                     RHS.second->getOffsetInBytes();
            });
  for (unsigned i = 0; i < PackOffsetVec.size() - 1; i++) {
    VarDecl *Var = PackOffsetVec[i].first;
    HLSLPackOffsetAttr *Attr = PackOffsetVec[i].second;
    unsigned Size = calculateLegacyCbufferSize(Context, Var->getType());
    unsigned Begin = Attr->getOffsetInBytes();
    unsigned End = Begin + Size;
    unsigned NextBegin = PackOffsetVec[i + 1].second->getOffsetInBytes();
    if (End > NextBegin) {
      VarDecl *NextVar = PackOffsetVec[i + 1].first;
      S.Diag(NextVar->getLocation(), diag::err_hlsl_packoffset_overlap)
          << NextVar << Var;
      IsValid = false;
    }
  }
  BufDecl->setHasValidPackoffset(IsValid);
}

// Returns true if the array has a zero size = if any of the dimensions is 0
static bool isZeroSizedArray(const ConstantArrayType *CAT) {
  while (CAT && !CAT->isZeroSize())
    CAT = dyn_cast<ConstantArrayType>(
        CAT->getElementType()->getUnqualifiedDesugaredType());
  return CAT != nullptr;
}

static bool isResourceRecordTypeOrArrayOf(VarDecl *VD) {
  const Type *Ty = VD->getType().getTypePtr();
  return Ty->isHLSLResourceRecord() || Ty->isHLSLResourceRecordArray();
}

static const HLSLAttributedResourceType *
getResourceArrayHandleType(VarDecl *VD) {
  assert(VD->getType()->isHLSLResourceRecordArray() &&
         "expected array of resource records");
  const Type *Ty = VD->getType()->getUnqualifiedDesugaredType();
  while (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(Ty))
    Ty = CAT->getArrayElementTypeNoTypeQual()->getUnqualifiedDesugaredType();
  return HLSLAttributedResourceType::findHandleTypeOnResource(Ty);
}

// Returns true if the type is a leaf element type that is not valid to be
// included in HLSL Buffer, such as a resource class, empty struct, zero-sized
// array, or a builtin intangible type. Returns false it is a valid leaf element
// type or if it is a record type that needs to be inspected further.
static bool isInvalidConstantBufferLeafElementType(const Type *Ty) {
  Ty = Ty->getUnqualifiedDesugaredType();
  if (Ty->isHLSLResourceRecord() || Ty->isHLSLResourceRecordArray())
    return true;
  if (Ty->isRecordType())
    return Ty->getAsCXXRecordDecl()->isEmpty();
  if (Ty->isConstantArrayType() &&
      isZeroSizedArray(cast<ConstantArrayType>(Ty)))
    return true;
  if (Ty->isHLSLBuiltinIntangibleType() || Ty->isHLSLAttributedResourceType())
    return true;
  return false;
}

// Returns true if the struct contains at least one element that prevents it
// from being included inside HLSL Buffer as is, such as an intangible type,
// empty struct, or zero-sized array. If it does, a new implicit layout struct
// needs to be created for HLSL Buffer use that will exclude these unwanted
// declarations (see createHostLayoutStruct function).
static bool requiresImplicitBufferLayoutStructure(const CXXRecordDecl *RD) {
  if (RD->isHLSLIntangible() || RD->isEmpty())
    return true;
  // check fields
  for (const FieldDecl *Field : RD->fields()) {
    QualType Ty = Field->getType();
    if (isInvalidConstantBufferLeafElementType(Ty.getTypePtr()))
      return true;
    if (Ty->isRecordType() &&
        requiresImplicitBufferLayoutStructure(Ty->getAsCXXRecordDecl()))
      return true;
  }
  // check bases
  for (const CXXBaseSpecifier &Base : RD->bases())
    if (requiresImplicitBufferLayoutStructure(
            Base.getType()->getAsCXXRecordDecl()))
      return true;
  return false;
}

static CXXRecordDecl *findRecordDeclInContext(IdentifierInfo *II,
                                              DeclContext *DC) {
  CXXRecordDecl *RD = nullptr;
  for (NamedDecl *Decl :
       DC->getNonTransparentContext()->lookup(DeclarationName(II))) {
    if (CXXRecordDecl *FoundRD = dyn_cast<CXXRecordDecl>(Decl)) {
      assert(RD == nullptr &&
             "there should be at most 1 record by a given name in a scope");
      RD = FoundRD;
    }
  }
  return RD;
}

// Creates a name for buffer layout struct using the provide name base.
// If the name must be unique (not previously defined), a suffix is added
// until a unique name is found.
static IdentifierInfo *getHostLayoutStructName(Sema &S, NamedDecl *BaseDecl,
                                               bool MustBeUnique) {
  ASTContext &AST = S.getASTContext();

  IdentifierInfo *NameBaseII = BaseDecl->getIdentifier();
  llvm::SmallString<64> Name("__cblayout_");
  if (NameBaseII) {
    Name.append(NameBaseII->getName());
  } else {
    // anonymous struct
    Name.append("anon");
    MustBeUnique = true;
  }

  size_t NameLength = Name.size();
  IdentifierInfo *II = &AST.Idents.get(Name, tok::TokenKind::identifier);
  if (!MustBeUnique)
    return II;

  unsigned suffix = 0;
  while (true) {
    if (suffix != 0) {
      Name.append("_");
      Name.append(llvm::Twine(suffix).str());
      II = &AST.Idents.get(Name, tok::TokenKind::identifier);
    }
    if (!findRecordDeclInContext(II, BaseDecl->getDeclContext()))
      return II;
    // declaration with that name already exists - increment suffix and try
    // again until unique name is found
    suffix++;
    Name.truncate(NameLength);
  };
}

// Creates a field declaration of given name and type for HLSL buffer layout
// struct. Returns nullptr if the type cannot be use in HLSL Buffer layout.
static FieldDecl *createFieldForHostLayoutStruct(Sema &S, const Type *Ty,
                                                 IdentifierInfo *II,
                                                 CXXRecordDecl *LayoutStruct) {
  if (isInvalidConstantBufferLeafElementType(Ty))
    return nullptr;

  if (Ty->isRecordType()) {
    CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    if (requiresImplicitBufferLayoutStructure(RD)) {
      RD = createHostLayoutStruct(S, RD);
      if (!RD)
        return nullptr;
      Ty = S.Context.getCanonicalTagType(RD)->getTypePtr();
    }
  }

  QualType QT = QualType(Ty, 0);
  ASTContext &AST = S.getASTContext();
  TypeSourceInfo *TSI = AST.getTrivialTypeSourceInfo(QT, SourceLocation());
  auto *Field = FieldDecl::Create(AST, LayoutStruct, SourceLocation(),
                                  SourceLocation(), II, QT, TSI, nullptr, false,
                                  InClassInitStyle::ICIS_NoInit);
  Field->setAccess(AccessSpecifier::AS_public);
  return Field;
}

// Creates host layout struct for a struct included in HLSL Buffer.
// The layout struct will include only fields that are allowed in HLSL buffer.
// These fields will be filtered out:
// - resource classes
// - empty structs
// - zero-sized arrays
// Returns nullptr if the resulting layout struct would be empty.
static CXXRecordDecl *createHostLayoutStruct(Sema &S,
                                             CXXRecordDecl *StructDecl) {
  assert(requiresImplicitBufferLayoutStructure(StructDecl) &&
         "struct is already HLSL buffer compatible");

  ASTContext &AST = S.getASTContext();
  DeclContext *DC = StructDecl->getDeclContext();
  IdentifierInfo *II = getHostLayoutStructName(S, StructDecl, false);

  // reuse existing if the layout struct if it already exists
  if (CXXRecordDecl *RD = findRecordDeclInContext(II, DC))
    return RD;

  CXXRecordDecl *LS =
      CXXRecordDecl::Create(AST, TagDecl::TagKind::Struct, DC, SourceLocation(),
                            SourceLocation(), II);
  LS->setImplicit(true);
  LS->addAttr(PackedAttr::CreateImplicit(AST));
  LS->startDefinition();

  // copy base struct, create HLSL Buffer compatible version if needed
  if (unsigned NumBases = StructDecl->getNumBases()) {
    assert(NumBases == 1 && "HLSL supports only one base type");
    (void)NumBases;
    CXXBaseSpecifier Base = *StructDecl->bases_begin();
    CXXRecordDecl *BaseDecl = Base.getType()->getAsCXXRecordDecl();
    if (requiresImplicitBufferLayoutStructure(BaseDecl)) {
      BaseDecl = createHostLayoutStruct(S, BaseDecl);
      if (BaseDecl) {
        TypeSourceInfo *TSI =
            AST.getTrivialTypeSourceInfo(AST.getCanonicalTagType(BaseDecl));
        Base = CXXBaseSpecifier(SourceRange(), false, StructDecl->isClass(),
                                AS_none, TSI, SourceLocation());
      }
    }
    if (BaseDecl) {
      const CXXBaseSpecifier *BasesArray[1] = {&Base};
      LS->setBases(BasesArray, 1);
    }
  }

  // filter struct fields
  for (const FieldDecl *FD : StructDecl->fields()) {
    const Type *Ty = FD->getType()->getUnqualifiedDesugaredType();
    if (FieldDecl *NewFD =
            createFieldForHostLayoutStruct(S, Ty, FD->getIdentifier(), LS))
      LS->addDecl(NewFD);
  }
  LS->completeDefinition();

  if (LS->field_empty() && LS->getNumBases() == 0)
    return nullptr;

  DC->addDecl(LS);
  return LS;
}

// Creates host layout struct for HLSL Buffer. The struct will include only
// fields of types that are allowed in HLSL buffer and it will filter out:
// - static or groupshared variable declarations
// - resource classes
// - empty structs
// - zero-sized arrays
// - non-variable declarations
// The layout struct will be added to the HLSLBufferDecl declarations.
void createHostLayoutStructForBuffer(Sema &S, HLSLBufferDecl *BufDecl) {
  ASTContext &AST = S.getASTContext();
  IdentifierInfo *II = getHostLayoutStructName(S, BufDecl, true);

  CXXRecordDecl *LS =
      CXXRecordDecl::Create(AST, TagDecl::TagKind::Struct, BufDecl,
                            SourceLocation(), SourceLocation(), II);
  LS->addAttr(PackedAttr::CreateImplicit(AST));
  LS->setImplicit(true);
  LS->startDefinition();

  for (Decl *D : BufDecl->buffer_decls()) {
    VarDecl *VD = dyn_cast<VarDecl>(D);
    if (!VD || VD->getStorageClass() == SC_Static ||
        VD->getType().getAddressSpace() == LangAS::hlsl_groupshared)
      continue;
    const Type *Ty = VD->getType()->getUnqualifiedDesugaredType();
    if (FieldDecl *FD =
            createFieldForHostLayoutStruct(S, Ty, VD->getIdentifier(), LS)) {
      // add the field decl to the layout struct
      LS->addDecl(FD);
      // update address space of the original decl to hlsl_constant
      QualType NewTy =
          AST.getAddrSpaceQualType(VD->getType(), LangAS::hlsl_constant);
      VD->setType(NewTy);
    }
  }
  LS->completeDefinition();
  BufDecl->addLayoutStruct(LS);
}

static void addImplicitBindingAttrToDecl(Sema &S, Decl *D, RegisterType RT,
                                         uint32_t ImplicitBindingOrderID) {
  auto *Attr =
      HLSLResourceBindingAttr::CreateImplicit(S.getASTContext(), "", "0", {});
  Attr->setBinding(RT, std::nullopt, 0);
  Attr->setImplicitBindingOrderID(ImplicitBindingOrderID);
  D->addAttr(Attr);
}

// Handle end of cbuffer/tbuffer declaration
void SemaHLSL::ActOnFinishBuffer(Decl *Dcl, SourceLocation RBrace) {
  auto *BufDecl = cast<HLSLBufferDecl>(Dcl);
  BufDecl->setRBraceLoc(RBrace);

  validatePackoffset(SemaRef, BufDecl);

  // create buffer layout struct
  createHostLayoutStructForBuffer(SemaRef, BufDecl);

  HLSLVkBindingAttr *VkBinding = Dcl->getAttr<HLSLVkBindingAttr>();
  HLSLResourceBindingAttr *RBA = Dcl->getAttr<HLSLResourceBindingAttr>();
  if (!VkBinding && (!RBA || !RBA->hasRegisterSlot())) {
    SemaRef.Diag(Dcl->getLocation(), diag::warn_hlsl_implicit_binding);
    // Use HLSLResourceBindingAttr to transfer implicit binding order_ID
    // to codegen. If it does not exist, create an implicit attribute.
    uint32_t OrderID = getNextImplicitBindingOrderID();
    if (RBA)
      RBA->setImplicitBindingOrderID(OrderID);
    else
      addImplicitBindingAttrToDecl(SemaRef, BufDecl,
                                   BufDecl->isCBuffer() ? RegisterType::CBuffer
                                                        : RegisterType::SRV,
                                   OrderID);
  }

  SemaRef.PopDeclContext();
}

HLSLNumThreadsAttr *SemaHLSL::mergeNumThreadsAttr(Decl *D,
                                                  const AttributeCommonInfo &AL,
                                                  int X, int Y, int Z) {
  if (HLSLNumThreadsAttr *NT = D->getAttr<HLSLNumThreadsAttr>()) {
    if (NT->getX() != X || NT->getY() != Y || NT->getZ() != Z) {
      Diag(NT->getLocation(), diag::err_hlsl_attribute_param_mismatch) << AL;
      Diag(AL.getLoc(), diag::note_conflicting_attribute);
    }
    return nullptr;
  }
  return ::new (getASTContext())
      HLSLNumThreadsAttr(getASTContext(), AL, X, Y, Z);
}

HLSLWaveSizeAttr *SemaHLSL::mergeWaveSizeAttr(Decl *D,
                                              const AttributeCommonInfo &AL,
                                              int Min, int Max, int Preferred,
                                              int SpelledArgsCount) {
  if (HLSLWaveSizeAttr *WS = D->getAttr<HLSLWaveSizeAttr>()) {
    if (WS->getMin() != Min || WS->getMax() != Max ||
        WS->getPreferred() != Preferred ||
        WS->getSpelledArgsCount() != SpelledArgsCount) {
      Diag(WS->getLocation(), diag::err_hlsl_attribute_param_mismatch) << AL;
      Diag(AL.getLoc(), diag::note_conflicting_attribute);
    }
    return nullptr;
  }
  HLSLWaveSizeAttr *Result = ::new (getASTContext())
      HLSLWaveSizeAttr(getASTContext(), AL, Min, Max, Preferred);
  Result->setSpelledArgsCount(SpelledArgsCount);
  return Result;
}

HLSLVkConstantIdAttr *
SemaHLSL::mergeVkConstantIdAttr(Decl *D, const AttributeCommonInfo &AL,
                                int Id) {

  auto &TargetInfo = getASTContext().getTargetInfo();
  if (TargetInfo.getTriple().getArch() != llvm::Triple::spirv) {
    Diag(AL.getLoc(), diag::warn_attribute_ignored) << AL;
    return nullptr;
  }

  auto *VD = cast<VarDecl>(D);

  if (getSpecConstBuiltinId(VD->getType()->getUnqualifiedDesugaredType()) ==
      Builtin::NotBuiltin) {
    Diag(VD->getLocation(), diag::err_specialization_const);
    return nullptr;
  }

  if (!VD->getType().isConstQualified()) {
    Diag(VD->getLocation(), diag::err_specialization_const);
    return nullptr;
  }

  if (HLSLVkConstantIdAttr *CI = D->getAttr<HLSLVkConstantIdAttr>()) {
    if (CI->getId() != Id) {
      Diag(CI->getLocation(), diag::err_hlsl_attribute_param_mismatch) << AL;
      Diag(AL.getLoc(), diag::note_conflicting_attribute);
    }
    return nullptr;
  }

  HLSLVkConstantIdAttr *Result =
      ::new (getASTContext()) HLSLVkConstantIdAttr(getASTContext(), AL, Id);
  return Result;
}

HLSLShaderAttr *
SemaHLSL::mergeShaderAttr(Decl *D, const AttributeCommonInfo &AL,
                          llvm::Triple::EnvironmentType ShaderType) {
  if (HLSLShaderAttr *NT = D->getAttr<HLSLShaderAttr>()) {
    if (NT->getType() != ShaderType) {
      Diag(NT->getLocation(), diag::err_hlsl_attribute_param_mismatch) << AL;
      Diag(AL.getLoc(), diag::note_conflicting_attribute);
    }
    return nullptr;
  }
  return HLSLShaderAttr::Create(getASTContext(), ShaderType, AL);
}

HLSLParamModifierAttr *
SemaHLSL::mergeParamModifierAttr(Decl *D, const AttributeCommonInfo &AL,
                                 HLSLParamModifierAttr::Spelling Spelling) {
  // We can only merge an `in` attribute with an `out` attribute. All other
  // combinations of duplicated attributes are ill-formed.
  if (HLSLParamModifierAttr *PA = D->getAttr<HLSLParamModifierAttr>()) {
    if ((PA->isIn() && Spelling == HLSLParamModifierAttr::Keyword_out) ||
        (PA->isOut() && Spelling == HLSLParamModifierAttr::Keyword_in)) {
      D->dropAttr<HLSLParamModifierAttr>();
      SourceRange AdjustedRange = {PA->getLocation(), AL.getRange().getEnd()};
      return HLSLParamModifierAttr::Create(
          getASTContext(), /*MergedSpelling=*/true, AdjustedRange,
          HLSLParamModifierAttr::Keyword_inout);
    }
    Diag(AL.getLoc(), diag::err_hlsl_duplicate_parameter_modifier) << AL;
    Diag(PA->getLocation(), diag::note_conflicting_attribute);
    return nullptr;
  }
  return HLSLParamModifierAttr::Create(getASTContext(), AL);
}

void SemaHLSL::ActOnTopLevelFunction(FunctionDecl *FD) {
  auto &TargetInfo = getASTContext().getTargetInfo();

  if (FD->getName() != TargetInfo.getTargetOpts().HLSLEntry)
    return;

  // If we have specified a root signature to override the entry function then
  // attach it now
  if (RootSigOverrideIdent) {
    LookupResult R(SemaRef, RootSigOverrideIdent, SourceLocation(),
                   Sema::LookupOrdinaryName);
    if (SemaRef.LookupQualifiedName(R, FD->getDeclContext()))
      if (auto *SignatureDecl =
              dyn_cast<HLSLRootSignatureDecl>(R.getFoundDecl())) {
        FD->dropAttr<RootSignatureAttr>();
        // We could look up the SourceRange of the macro here as well
        AttributeCommonInfo AL(RootSigOverrideIdent, AttributeScopeInfo(),
                               SourceRange(), ParsedAttr::Form::Microsoft());
        FD->addAttr(::new (getASTContext()) RootSignatureAttr(
            getASTContext(), AL, RootSigOverrideIdent, SignatureDecl));
      }
  }

  llvm::Triple::EnvironmentType Env = TargetInfo.getTriple().getEnvironment();
  if (HLSLShaderAttr::isValidShaderType(Env) && Env != llvm::Triple::Library) {
    if (const auto *Shader = FD->getAttr<HLSLShaderAttr>()) {
      // The entry point is already annotated - check that it matches the
      // triple.
      if (Shader->getType() != Env) {
        Diag(Shader->getLocation(), diag::err_hlsl_entry_shader_attr_mismatch)
            << Shader;
        FD->setInvalidDecl();
      }
    } else {
      // Implicitly add the shader attribute if the entry function isn't
      // explicitly annotated.
      FD->addAttr(HLSLShaderAttr::CreateImplicit(getASTContext(), Env,
                                                 FD->getBeginLoc()));
    }
  } else {
    switch (Env) {
    case llvm::Triple::UnknownEnvironment:
    case llvm::Triple::Library:
      break;
    default:
      llvm_unreachable("Unhandled environment in triple");
    }
  }
}

void SemaHLSL::CheckEntryPoint(FunctionDecl *FD) {
  const auto *ShaderAttr = FD->getAttr<HLSLShaderAttr>();
  assert(ShaderAttr && "Entry point has no shader attribute");
  llvm::Triple::EnvironmentType ST = ShaderAttr->getType();
  auto &TargetInfo = getASTContext().getTargetInfo();
  VersionTuple Ver = TargetInfo.getTriple().getOSVersion();
  switch (ST) {
  case llvm::Triple::Pixel:
  case llvm::Triple::Vertex:
  case llvm::Triple::Geometry:
  case llvm::Triple::Hull:
  case llvm::Triple::Domain:
  case llvm::Triple::RayGeneration:
  case llvm::Triple::Intersection:
  case llvm::Triple::AnyHit:
  case llvm::Triple::ClosestHit:
  case llvm::Triple::Miss:
  case llvm::Triple::Callable:
    if (const auto *NT = FD->getAttr<HLSLNumThreadsAttr>()) {
      DiagnoseAttrStageMismatch(NT, ST,
                                {llvm::Triple::Compute,
                                 llvm::Triple::Amplification,
                                 llvm::Triple::Mesh});
      FD->setInvalidDecl();
    }
    if (const auto *WS = FD->getAttr<HLSLWaveSizeAttr>()) {
      DiagnoseAttrStageMismatch(WS, ST,
                                {llvm::Triple::Compute,
                                 llvm::Triple::Amplification,
                                 llvm::Triple::Mesh});
      FD->setInvalidDecl();
    }
    break;

  case llvm::Triple::Compute:
  case llvm::Triple::Amplification:
  case llvm::Triple::Mesh:
    if (!FD->hasAttr<HLSLNumThreadsAttr>()) {
      Diag(FD->getLocation(), diag::err_hlsl_missing_numthreads)
          << llvm::Triple::getEnvironmentTypeName(ST);
      FD->setInvalidDecl();
    }
    if (const auto *WS = FD->getAttr<HLSLWaveSizeAttr>()) {
      if (Ver < VersionTuple(6, 6)) {
        Diag(WS->getLocation(), diag::err_hlsl_attribute_in_wrong_shader_model)
            << WS << "6.6";
        FD->setInvalidDecl();
      } else if (WS->getSpelledArgsCount() > 1 && Ver < VersionTuple(6, 8)) {
        Diag(
            WS->getLocation(),
            diag::err_hlsl_attribute_number_arguments_insufficient_shader_model)
            << WS << WS->getSpelledArgsCount() << "6.8";
        FD->setInvalidDecl();
      }
    }
    break;
  default:
    llvm_unreachable("Unhandled environment in triple");
  }

  for (ParmVarDecl *Param : FD->parameters()) {
    if (const auto *AnnotationAttr = Param->getAttr<HLSLAnnotationAttr>()) {
      CheckSemanticAnnotation(FD, Param, AnnotationAttr);
    } else {
      // FIXME: Handle struct parameters where annotations are on struct fields.
      // See: https://github.com/llvm/llvm-project/issues/57875
      Diag(FD->getLocation(), diag::err_hlsl_missing_semantic_annotation);
      Diag(Param->getLocation(), diag::note_previous_decl) << Param;
      FD->setInvalidDecl();
    }
  }
  // FIXME: Verify return type semantic annotation.
}

void SemaHLSL::CheckSemanticAnnotation(
    FunctionDecl *EntryPoint, const Decl *Param,
    const HLSLAnnotationAttr *AnnotationAttr) {
  auto *ShaderAttr = EntryPoint->getAttr<HLSLShaderAttr>();
  assert(ShaderAttr && "Entry point has no shader attribute");
  llvm::Triple::EnvironmentType ST = ShaderAttr->getType();

  switch (AnnotationAttr->getKind()) {
  case attr::HLSLSV_DispatchThreadID:
  case attr::HLSLSV_GroupIndex:
  case attr::HLSLSV_GroupThreadID:
  case attr::HLSLSV_GroupID:
    if (ST == llvm::Triple::Compute)
      return;
    DiagnoseAttrStageMismatch(AnnotationAttr, ST, {llvm::Triple::Compute});
    break;
  case attr::HLSLSV_Position:
    // TODO(#143523): allow use on other shader types & output once the overall
    // semantic logic is implemented.
    if (ST == llvm::Triple::Pixel)
      return;
    DiagnoseAttrStageMismatch(AnnotationAttr, ST, {llvm::Triple::Pixel});
    break;
  default:
    llvm_unreachable("Unknown HLSLAnnotationAttr");
  }
}

void SemaHLSL::DiagnoseAttrStageMismatch(
    const Attr *A, llvm::Triple::EnvironmentType Stage,
    std::initializer_list<llvm::Triple::EnvironmentType> AllowedStages) {
  SmallVector<StringRef, 8> StageStrings;
  llvm::transform(AllowedStages, std::back_inserter(StageStrings),
                  [](llvm::Triple::EnvironmentType ST) {
                    return StringRef(
                        HLSLShaderAttr::ConvertEnvironmentTypeToStr(ST));
                  });
  Diag(A->getLoc(), diag::err_hlsl_attr_unsupported_in_stage)
      << A->getAttrName() << llvm::Triple::getEnvironmentTypeName(Stage)
      << (AllowedStages.size() != 1) << join(StageStrings, ", ");
}

template <CastKind Kind>
static void castVector(Sema &S, ExprResult &E, QualType &Ty, unsigned Sz) {
  if (const auto *VTy = Ty->getAs<VectorType>())
    Ty = VTy->getElementType();
  Ty = S.getASTContext().getExtVectorType(Ty, Sz);
  E = S.ImpCastExprToType(E.get(), Ty, Kind);
}

template <CastKind Kind>
static QualType castElement(Sema &S, ExprResult &E, QualType Ty) {
  E = S.ImpCastExprToType(E.get(), Ty, Kind);
  return Ty;
}

static QualType handleFloatVectorBinOpConversion(
    Sema &SemaRef, ExprResult &LHS, ExprResult &RHS, QualType LHSType,
    QualType RHSType, QualType LElTy, QualType RElTy, bool IsCompAssign) {
  bool LHSFloat = LElTy->isRealFloatingType();
  bool RHSFloat = RElTy->isRealFloatingType();

  if (LHSFloat && RHSFloat) {
    if (IsCompAssign ||
        SemaRef.getASTContext().getFloatingTypeOrder(LElTy, RElTy) > 0)
      return castElement<CK_FloatingCast>(SemaRef, RHS, LHSType);

    return castElement<CK_FloatingCast>(SemaRef, LHS, RHSType);
  }

  if (LHSFloat)
    return castElement<CK_IntegralToFloating>(SemaRef, RHS, LHSType);

  assert(RHSFloat);
  if (IsCompAssign)
    return castElement<clang::CK_FloatingToIntegral>(SemaRef, RHS, LHSType);

  return castElement<CK_IntegralToFloating>(SemaRef, LHS, RHSType);
}

static QualType handleIntegerVectorBinOpConversion(
    Sema &SemaRef, ExprResult &LHS, ExprResult &RHS, QualType LHSType,
    QualType RHSType, QualType LElTy, QualType RElTy, bool IsCompAssign) {

  int IntOrder = SemaRef.Context.getIntegerTypeOrder(LElTy, RElTy);
  bool LHSSigned = LElTy->hasSignedIntegerRepresentation();
  bool RHSSigned = RElTy->hasSignedIntegerRepresentation();
  auto &Ctx = SemaRef.getASTContext();

  // If both types have the same signedness, use the higher ranked type.
  if (LHSSigned == RHSSigned) {
    if (IsCompAssign || IntOrder >= 0)
      return castElement<CK_IntegralCast>(SemaRef, RHS, LHSType);

    return castElement<CK_IntegralCast>(SemaRef, LHS, RHSType);
  }

  // If the unsigned type has greater than or equal rank of the signed type, use
  // the unsigned type.
  if (IntOrder != (LHSSigned ? 1 : -1)) {
    if (IsCompAssign || RHSSigned)
      return castElement<CK_IntegralCast>(SemaRef, RHS, LHSType);
    return castElement<CK_IntegralCast>(SemaRef, LHS, RHSType);
  }

  // At this point the signed type has higher rank than the unsigned type, which
  // means it will be the same size or bigger. If the signed type is bigger, it
  // can represent all the values of the unsigned type, so select it.
  if (Ctx.getIntWidth(LElTy) != Ctx.getIntWidth(RElTy)) {
    if (IsCompAssign || LHSSigned)
      return castElement<CK_IntegralCast>(SemaRef, RHS, LHSType);
    return castElement<CK_IntegralCast>(SemaRef, LHS, RHSType);
  }

  // This is a bit of an odd duck case in HLSL. It shouldn't happen, but can due
  // to C/C++ leaking through. The place this happens today is long vs long
  // long. When arguments are vector<unsigned long, N> and vector<long long, N>,
  // the long long has higher rank than long even though they are the same size.

  // If this is a compound assignment cast the right hand side to the left hand
  // side's type.
  if (IsCompAssign)
    return castElement<CK_IntegralCast>(SemaRef, RHS, LHSType);

  // If this isn't a compound assignment we convert to unsigned long long.
  QualType ElTy = Ctx.getCorrespondingUnsignedType(LHSSigned ? LElTy : RElTy);
  QualType NewTy = Ctx.getExtVectorType(
      ElTy, RHSType->castAs<VectorType>()->getNumElements());
  (void)castElement<CK_IntegralCast>(SemaRef, RHS, NewTy);

  return castElement<CK_IntegralCast>(SemaRef, LHS, NewTy);
}

static CastKind getScalarCastKind(ASTContext &Ctx, QualType DestTy,
                                  QualType SrcTy) {
  if (DestTy->isRealFloatingType() && SrcTy->isRealFloatingType())
    return CK_FloatingCast;
  if (DestTy->isIntegralType(Ctx) && SrcTy->isIntegralType(Ctx))
    return CK_IntegralCast;
  if (DestTy->isRealFloatingType())
    return CK_IntegralToFloating;
  assert(SrcTy->isRealFloatingType() && DestTy->isIntegralType(Ctx));
  return CK_FloatingToIntegral;
}

QualType SemaHLSL::handleVectorBinOpConversion(ExprResult &LHS, ExprResult &RHS,
                                               QualType LHSType,
                                               QualType RHSType,
                                               bool IsCompAssign) {
  const auto *LVecTy = LHSType->getAs<VectorType>();
  const auto *RVecTy = RHSType->getAs<VectorType>();
  auto &Ctx = getASTContext();

  // If the LHS is not a vector and this is a compound assignment, we truncate
  // the argument to a scalar then convert it to the LHS's type.
  if (!LVecTy && IsCompAssign) {
    QualType RElTy = RHSType->castAs<VectorType>()->getElementType();
    RHS = SemaRef.ImpCastExprToType(RHS.get(), RElTy, CK_HLSLVectorTruncation);
    RHSType = RHS.get()->getType();
    if (Ctx.hasSameUnqualifiedType(LHSType, RHSType))
      return LHSType;
    RHS = SemaRef.ImpCastExprToType(RHS.get(), LHSType,
                                    getScalarCastKind(Ctx, LHSType, RHSType));
    return LHSType;
  }

  unsigned EndSz = std::numeric_limits<unsigned>::max();
  unsigned LSz = 0;
  if (LVecTy)
    LSz = EndSz = LVecTy->getNumElements();
  if (RVecTy)
    EndSz = std::min(RVecTy->getNumElements(), EndSz);
  assert(EndSz != std::numeric_limits<unsigned>::max() &&
         "one of the above should have had a value");

  // In a compound assignment, the left operand does not change type, the right
  // operand is converted to the type of the left operand.
  if (IsCompAssign && LSz != EndSz) {
    Diag(LHS.get()->getBeginLoc(),
         diag::err_hlsl_vector_compound_assignment_truncation)
        << LHSType << RHSType;
    return QualType();
  }

  if (RVecTy && RVecTy->getNumElements() > EndSz)
    castVector<CK_HLSLVectorTruncation>(SemaRef, RHS, RHSType, EndSz);
  if (!IsCompAssign && LVecTy && LVecTy->getNumElements() > EndSz)
    castVector<CK_HLSLVectorTruncation>(SemaRef, LHS, LHSType, EndSz);

  if (!RVecTy)
    castVector<CK_VectorSplat>(SemaRef, RHS, RHSType, EndSz);
  if (!IsCompAssign && !LVecTy)
    castVector<CK_VectorSplat>(SemaRef, LHS, LHSType, EndSz);

  // If we're at the same type after resizing we can stop here.
  if (Ctx.hasSameUnqualifiedType(LHSType, RHSType))
    return Ctx.getCommonSugaredType(LHSType, RHSType);

  QualType LElTy = LHSType->castAs<VectorType>()->getElementType();
  QualType RElTy = RHSType->castAs<VectorType>()->getElementType();

  // Handle conversion for floating point vectors.
  if (LElTy->isRealFloatingType() || RElTy->isRealFloatingType())
    return handleFloatVectorBinOpConversion(SemaRef, LHS, RHS, LHSType, RHSType,
                                            LElTy, RElTy, IsCompAssign);

  assert(LElTy->isIntegralType(Ctx) && RElTy->isIntegralType(Ctx) &&
         "HLSL Vectors can only contain integer or floating point types");
  return handleIntegerVectorBinOpConversion(SemaRef, LHS, RHS, LHSType, RHSType,
                                            LElTy, RElTy, IsCompAssign);
}

void SemaHLSL::emitLogicalOperatorFixIt(Expr *LHS, Expr *RHS,
                                        BinaryOperatorKind Opc) {
  assert((Opc == BO_LOr || Opc == BO_LAnd) &&
         "Called with non-logical operator");
  llvm::SmallVector<char, 256> Buff;
  llvm::raw_svector_ostream OS(Buff);
  PrintingPolicy PP(SemaRef.getLangOpts());
  StringRef NewFnName = Opc == BO_LOr ? "or" : "and";
  OS << NewFnName << "(";
  LHS->printPretty(OS, nullptr, PP);
  OS << ", ";
  RHS->printPretty(OS, nullptr, PP);
  OS << ")";
  SourceRange FullRange = SourceRange(LHS->getBeginLoc(), RHS->getEndLoc());
  SemaRef.Diag(LHS->getBeginLoc(), diag::note_function_suggestion)
      << NewFnName << FixItHint::CreateReplacement(FullRange, OS.str());
}

std::pair<IdentifierInfo *, bool>
SemaHLSL::ActOnStartRootSignatureDecl(StringRef Signature) {
  llvm::hash_code Hash = llvm::hash_value(Signature);
  std::string IdStr = "__hlsl_rootsig_decl_" + std::to_string(Hash);
  IdentifierInfo *DeclIdent = &(getASTContext().Idents.get(IdStr));

  // Check if we have already found a decl of the same name.
  LookupResult R(SemaRef, DeclIdent, SourceLocation(),
                 Sema::LookupOrdinaryName);
  bool Found = SemaRef.LookupQualifiedName(R, SemaRef.CurContext);
  return {DeclIdent, Found};
}

void SemaHLSL::ActOnFinishRootSignatureDecl(
    SourceLocation Loc, IdentifierInfo *DeclIdent,
    ArrayRef<hlsl::RootSignatureElement> RootElements) {

  if (handleRootSignatureElements(RootElements))
    return;

  SmallVector<llvm::hlsl::rootsig::RootElement> Elements;
  for (auto &RootSigElement : RootElements)
    Elements.push_back(RootSigElement.getElement());

  auto *SignatureDecl = HLSLRootSignatureDecl::Create(
      SemaRef.getASTContext(), /*DeclContext=*/SemaRef.CurContext, Loc,
      DeclIdent, SemaRef.getLangOpts().HLSLRootSigVer, Elements);

  SignatureDecl->setImplicit();
  SemaRef.PushOnScopeChains(SignatureDecl, SemaRef.getCurScope());
}

namespace {

struct PerVisibilityBindingChecker {
  SemaHLSL *S;
  // We need one builder per `llvm::dxbc::ShaderVisibility` value.
  std::array<llvm::hlsl::BindingInfoBuilder, 8> Builders;

  struct ElemInfo {
    const hlsl::RootSignatureElement *Elem;
    llvm::dxbc::ShaderVisibility Vis;
    bool Diagnosed;
  };
  llvm::SmallVector<ElemInfo> ElemInfoMap;

  PerVisibilityBindingChecker(SemaHLSL *S) : S(S) {}

  void trackBinding(llvm::dxbc::ShaderVisibility Visibility,
                    llvm::dxil::ResourceClass RC, uint32_t Space,
                    uint32_t LowerBound, uint32_t UpperBound,
                    const hlsl::RootSignatureElement *Elem) {
    uint32_t BuilderIndex = llvm::to_underlying(Visibility);
    assert(BuilderIndex < Builders.size() &&
           "Not enough builders for visibility type");
    Builders[BuilderIndex].trackBinding(RC, Space, LowerBound, UpperBound,
                                        static_cast<const void *>(Elem));

    static_assert(llvm::to_underlying(llvm::dxbc::ShaderVisibility::All) == 0,
                  "'All' visibility must come first");
    if (Visibility == llvm::dxbc::ShaderVisibility::All)
      for (size_t I = 1, E = Builders.size(); I < E; ++I)
        Builders[I].trackBinding(RC, Space, LowerBound, UpperBound,
                                 static_cast<const void *>(Elem));

    ElemInfoMap.push_back({Elem, Visibility, false});
  }

  ElemInfo &getInfo(const hlsl::RootSignatureElement *Elem) {
    auto It = llvm::lower_bound(
        ElemInfoMap, Elem,
        [](const auto &LHS, const auto &RHS) { return LHS.Elem < RHS; });
    assert(It->Elem == Elem && "Element not in map");
    return *It;
  }

  bool checkOverlap() {
    llvm::sort(ElemInfoMap, [](const auto &LHS, const auto &RHS) {
      return LHS.Elem < RHS.Elem;
    });

    bool HadOverlap = false;

    using llvm::hlsl::BindingInfoBuilder;
    auto ReportOverlap = [this, &HadOverlap](
                             const BindingInfoBuilder &Builder,
                             const BindingInfoBuilder::Binding &Reported) {
      HadOverlap = true;

      const auto *Elem =
          static_cast<const hlsl::RootSignatureElement *>(Reported.Cookie);
      const BindingInfoBuilder::Binding &Previous =
          Builder.findOverlapping(Reported);
      const auto *PrevElem =
          static_cast<const hlsl::RootSignatureElement *>(Previous.Cookie);

      ElemInfo &Info = getInfo(Elem);
      // We will have already diagnosed this binding if there's overlap in the
      // "All" visibility as well as any particular visibility.
      if (Info.Diagnosed)
        return;
      Info.Diagnosed = true;

      ElemInfo &PrevInfo = getInfo(PrevElem);
      llvm::dxbc::ShaderVisibility CommonVis =
          Info.Vis == llvm::dxbc::ShaderVisibility::All ? PrevInfo.Vis
                                                        : Info.Vis;

      this->S->Diag(Elem->getLocation(), diag::err_hlsl_resource_range_overlap)
          << llvm::to_underlying(Reported.RC) << Reported.LowerBound
          << Reported.isUnbounded() << Reported.UpperBound
          << llvm::to_underlying(Previous.RC) << Previous.LowerBound
          << Previous.isUnbounded() << Previous.UpperBound << Reported.Space
          << CommonVis;

      this->S->Diag(PrevElem->getLocation(),
                    diag::note_hlsl_resource_range_here);
    };

    for (BindingInfoBuilder &Builder : Builders)
      Builder.calculateBindingInfo(ReportOverlap);

    return HadOverlap;
  }
};

} // end anonymous namespace

bool SemaHLSL::handleRootSignatureElements(
    ArrayRef<hlsl::RootSignatureElement> Elements) {
  // Define some common error handling functions
  bool HadError = false;
  auto ReportError = [this, &HadError](SourceLocation Loc, uint32_t LowerBound,
                                       uint32_t UpperBound) {
    HadError = true;
    this->Diag(Loc, diag::err_hlsl_invalid_rootsig_value)
        << LowerBound << UpperBound;
  };

  auto ReportFloatError = [this, &HadError](SourceLocation Loc,
                                            float LowerBound,
                                            float UpperBound) {
    HadError = true;
    this->Diag(Loc, diag::err_hlsl_invalid_rootsig_value)
        << llvm::formatv("{0:f}", LowerBound).sstr<6>()
        << llvm::formatv("{0:f}", UpperBound).sstr<6>();
  };

  auto VerifyRegister = [ReportError](SourceLocation Loc, uint32_t Register) {
    if (!llvm::hlsl::rootsig::verifyRegisterValue(Register))
      ReportError(Loc, 0, 0xfffffffe);
  };

  auto VerifySpace = [ReportError](SourceLocation Loc, uint32_t Space) {
    if (!llvm::hlsl::rootsig::verifyRegisterSpace(Space))
      ReportError(Loc, 0, 0xffffffef);
  };

  const uint32_t Version =
      llvm::to_underlying(SemaRef.getLangOpts().HLSLRootSigVer);
  const uint32_t VersionEnum = Version - 1;
  auto ReportFlagError = [this, &HadError, VersionEnum](SourceLocation Loc) {
    HadError = true;
    this->Diag(Loc, diag::err_hlsl_invalid_rootsig_flag)
        << /*version minor*/ VersionEnum;
  };

  // Iterate through the elements and do basic validations
  for (const hlsl::RootSignatureElement &RootSigElem : Elements) {
    SourceLocation Loc = RootSigElem.getLocation();
    const llvm::hlsl::rootsig::RootElement &Elem = RootSigElem.getElement();
    if (const auto *Descriptor =
            std::get_if<llvm::hlsl::rootsig::RootDescriptor>(&Elem)) {
      VerifyRegister(Loc, Descriptor->Reg.Number);
      VerifySpace(Loc, Descriptor->Space);

      if (!llvm::hlsl::rootsig::verifyRootDescriptorFlag(
              Version, llvm::to_underlying(Descriptor->Flags)))
        ReportFlagError(Loc);
    } else if (const auto *Constants =
                   std::get_if<llvm::hlsl::rootsig::RootConstants>(&Elem)) {
      VerifyRegister(Loc, Constants->Reg.Number);
      VerifySpace(Loc, Constants->Space);
    } else if (const auto *Sampler =
                   std::get_if<llvm::hlsl::rootsig::StaticSampler>(&Elem)) {
      VerifyRegister(Loc, Sampler->Reg.Number);
      VerifySpace(Loc, Sampler->Space);

      assert(!std::isnan(Sampler->MaxLOD) && !std::isnan(Sampler->MinLOD) &&
             "By construction, parseFloatParam can't produce a NaN from a "
             "float_literal token");

      if (!llvm::hlsl::rootsig::verifyMaxAnisotropy(Sampler->MaxAnisotropy))
        ReportError(Loc, 0, 16);
      if (!llvm::hlsl::rootsig::verifyMipLODBias(Sampler->MipLODBias))
        ReportFloatError(Loc, -16.f, 15.99f);
    } else if (const auto *Clause =
                   std::get_if<llvm::hlsl::rootsig::DescriptorTableClause>(
                       &Elem)) {
      VerifyRegister(Loc, Clause->Reg.Number);
      VerifySpace(Loc, Clause->Space);

      if (!llvm::hlsl::rootsig::verifyNumDescriptors(Clause->NumDescriptors)) {
        // NumDescriptor could techincally be ~0u but that is reserved for
        // unbounded, so the diagnostic will not report that as a valid int
        // value
        ReportError(Loc, 1, 0xfffffffe);
      }

      if (!llvm::hlsl::rootsig::verifyDescriptorRangeFlag(
              Version, llvm::to_underlying(Clause->Type),
              llvm::to_underlying(Clause->Flags)))
        ReportFlagError(Loc);
    }
  }

  PerVisibilityBindingChecker BindingChecker(this);
  SmallVector<std::pair<const llvm::hlsl::rootsig::DescriptorTableClause *,
                        const hlsl::RootSignatureElement *>>
      UnboundClauses;

  for (const hlsl::RootSignatureElement &RootSigElem : Elements) {
    const llvm::hlsl::rootsig::RootElement &Elem = RootSigElem.getElement();
    if (const auto *Descriptor =
            std::get_if<llvm::hlsl::rootsig::RootDescriptor>(&Elem)) {
      uint32_t LowerBound(Descriptor->Reg.Number);
      uint32_t UpperBound(LowerBound); // inclusive range

      BindingChecker.trackBinding(
          Descriptor->Visibility,
          static_cast<llvm::dxil::ResourceClass>(Descriptor->Type),
          Descriptor->Space, LowerBound, UpperBound, &RootSigElem);
    } else if (const auto *Constants =
                   std::get_if<llvm::hlsl::rootsig::RootConstants>(&Elem)) {
      uint32_t LowerBound(Constants->Reg.Number);
      uint32_t UpperBound(LowerBound); // inclusive range

      BindingChecker.trackBinding(
          Constants->Visibility, llvm::dxil::ResourceClass::CBuffer,
          Constants->Space, LowerBound, UpperBound, &RootSigElem);
    } else if (const auto *Sampler =
                   std::get_if<llvm::hlsl::rootsig::StaticSampler>(&Elem)) {
      uint32_t LowerBound(Sampler->Reg.Number);
      uint32_t UpperBound(LowerBound); // inclusive range

      BindingChecker.trackBinding(
          Sampler->Visibility, llvm::dxil::ResourceClass::Sampler,
          Sampler->Space, LowerBound, UpperBound, &RootSigElem);
    } else if (const auto *Clause =
                   std::get_if<llvm::hlsl::rootsig::DescriptorTableClause>(
                       &Elem)) {
      // We'll process these once we see the table element.
      UnboundClauses.emplace_back(Clause, &RootSigElem);
    } else if (const auto *Table =
                   std::get_if<llvm::hlsl::rootsig::DescriptorTable>(&Elem)) {
      assert(UnboundClauses.size() == Table->NumClauses &&
             "Number of unbound elements must match the number of clauses");
      for (const auto &[Clause, ClauseElem] : UnboundClauses) {
        uint32_t LowerBound(Clause->Reg.Number);
        // Relevant error will have already been reported above and needs to be
        // fixed before we can conduct range analysis, so shortcut error return
        if (Clause->NumDescriptors == 0)
          return true;
        uint32_t UpperBound = Clause->NumDescriptors == ~0u
                                  ? ~0u
                                  : LowerBound + Clause->NumDescriptors - 1;

        BindingChecker.trackBinding(
            Table->Visibility,
            static_cast<llvm::dxil::ResourceClass>(Clause->Type), Clause->Space,
            LowerBound, UpperBound, ClauseElem);
      }
      UnboundClauses.clear();
    }
  }

  return BindingChecker.checkOverlap();
}

void SemaHLSL::handleRootSignatureAttr(Decl *D, const ParsedAttr &AL) {
  if (AL.getNumArgs() != 1) {
    Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << AL << 1;
    return;
  }

  IdentifierInfo *Ident = AL.getArgAsIdent(0)->getIdentifierInfo();
  if (auto *RS = D->getAttr<RootSignatureAttr>()) {
    if (RS->getSignatureIdent() != Ident) {
      Diag(AL.getLoc(), diag::err_disallowed_duplicate_attribute) << RS;
      return;
    }

    Diag(AL.getLoc(), diag::warn_duplicate_attribute_exact) << RS;
    return;
  }

  LookupResult R(SemaRef, Ident, SourceLocation(), Sema::LookupOrdinaryName);
  if (SemaRef.LookupQualifiedName(R, D->getDeclContext()))
    if (auto *SignatureDecl =
            dyn_cast<HLSLRootSignatureDecl>(R.getFoundDecl())) {
      D->addAttr(::new (getASTContext()) RootSignatureAttr(
          getASTContext(), AL, Ident, SignatureDecl));
    }
}

void SemaHLSL::handleNumThreadsAttr(Decl *D, const ParsedAttr &AL) {
  llvm::VersionTuple SMVersion =
      getASTContext().getTargetInfo().getTriple().getOSVersion();
  bool IsDXIL = getASTContext().getTargetInfo().getTriple().getArch() ==
                llvm::Triple::dxil;

  uint32_t ZMax = 1024;
  uint32_t ThreadMax = 1024;
  if (IsDXIL && SMVersion.getMajor() <= 4) {
    ZMax = 1;
    ThreadMax = 768;
  } else if (IsDXIL && SMVersion.getMajor() == 5) {
    ZMax = 64;
    ThreadMax = 1024;
  }

  uint32_t X;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), X))
    return;
  if (X > 1024) {
    Diag(AL.getArgAsExpr(0)->getExprLoc(),
         diag::err_hlsl_numthreads_argument_oor)
        << 0 << 1024;
    return;
  }
  uint32_t Y;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(1), Y))
    return;
  if (Y > 1024) {
    Diag(AL.getArgAsExpr(1)->getExprLoc(),
         diag::err_hlsl_numthreads_argument_oor)
        << 1 << 1024;
    return;
  }
  uint32_t Z;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(2), Z))
    return;
  if (Z > ZMax) {
    SemaRef.Diag(AL.getArgAsExpr(2)->getExprLoc(),
                 diag::err_hlsl_numthreads_argument_oor)
        << 2 << ZMax;
    return;
  }

  if (X * Y * Z > ThreadMax) {
    Diag(AL.getLoc(), diag::err_hlsl_numthreads_invalid) << ThreadMax;
    return;
  }

  HLSLNumThreadsAttr *NewAttr = mergeNumThreadsAttr(D, AL, X, Y, Z);
  if (NewAttr)
    D->addAttr(NewAttr);
}

static bool isValidWaveSizeValue(unsigned Value) {
  return llvm::isPowerOf2_32(Value) && Value >= 4 && Value <= 128;
}

void SemaHLSL::handleWaveSizeAttr(Decl *D, const ParsedAttr &AL) {
  // validate that the wavesize argument is a power of 2 between 4 and 128
  // inclusive
  unsigned SpelledArgsCount = AL.getNumArgs();
  if (SpelledArgsCount == 0 || SpelledArgsCount > 3)
    return;

  uint32_t Min;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), Min))
    return;

  uint32_t Max = 0;
  if (SpelledArgsCount > 1 &&
      !SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(1), Max))
    return;

  uint32_t Preferred = 0;
  if (SpelledArgsCount > 2 &&
      !SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(2), Preferred))
    return;

  if (SpelledArgsCount > 2) {
    if (!isValidWaveSizeValue(Preferred)) {
      Diag(AL.getArgAsExpr(2)->getExprLoc(),
           diag::err_attribute_power_of_two_in_range)
          << AL << llvm::dxil::MinWaveSize << llvm::dxil::MaxWaveSize
          << Preferred;
      return;
    }
    // Preferred not in range.
    if (Preferred < Min || Preferred > Max) {
      Diag(AL.getArgAsExpr(2)->getExprLoc(),
           diag::err_attribute_power_of_two_in_range)
          << AL << Min << Max << Preferred;
      return;
    }
  } else if (SpelledArgsCount > 1) {
    if (!isValidWaveSizeValue(Max)) {
      Diag(AL.getArgAsExpr(1)->getExprLoc(),
           diag::err_attribute_power_of_two_in_range)
          << AL << llvm::dxil::MinWaveSize << llvm::dxil::MaxWaveSize << Max;
      return;
    }
    if (Max < Min) {
      Diag(AL.getLoc(), diag::err_attribute_argument_invalid) << AL << 1;
      return;
    } else if (Max == Min) {
      Diag(AL.getLoc(), diag::warn_attr_min_eq_max) << AL;
    }
  } else {
    if (!isValidWaveSizeValue(Min)) {
      Diag(AL.getArgAsExpr(0)->getExprLoc(),
           diag::err_attribute_power_of_two_in_range)
          << AL << llvm::dxil::MinWaveSize << llvm::dxil::MaxWaveSize << Min;
      return;
    }
  }

  HLSLWaveSizeAttr *NewAttr =
      mergeWaveSizeAttr(D, AL, Min, Max, Preferred, SpelledArgsCount);
  if (NewAttr)
    D->addAttr(NewAttr);
}

void SemaHLSL::handleVkExtBuiltinInputAttr(Decl *D, const ParsedAttr &AL) {
  uint32_t ID;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), ID))
    return;
  D->addAttr(::new (getASTContext())
                 HLSLVkExtBuiltinInputAttr(getASTContext(), AL, ID));
}

void SemaHLSL::handleVkConstantIdAttr(Decl *D, const ParsedAttr &AL) {
  uint32_t Id;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), Id))
    return;
  HLSLVkConstantIdAttr *NewAttr = mergeVkConstantIdAttr(D, AL, Id);
  if (NewAttr)
    D->addAttr(NewAttr);
}

void SemaHLSL::handleVkBindingAttr(Decl *D, const ParsedAttr &AL) {
  // The vk::binding attribute only applies to SPIR-V.
  if (!getASTContext().getTargetInfo().getTriple().isSPIRV())
    return;

  uint32_t Binding = 0;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), Binding))
    return;
  uint32_t Set = 0;
  if (AL.getNumArgs() > 1 &&
      !SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(1), Set))
    return;

  D->addAttr(::new (getASTContext())
                 HLSLVkBindingAttr(getASTContext(), AL, Binding, Set));
}

bool SemaHLSL::diagnoseInputIDType(QualType T, const ParsedAttr &AL) {
  const auto *VT = T->getAs<VectorType>();

  if (!T->hasUnsignedIntegerRepresentation() ||
      (VT && VT->getNumElements() > 3)) {
    Diag(AL.getLoc(), diag::err_hlsl_attr_invalid_type)
        << AL << "uint/uint2/uint3";
    return false;
  }

  return true;
}

void SemaHLSL::handleSV_DispatchThreadIDAttr(Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<ValueDecl>(D);
  if (!diagnoseInputIDType(VD->getType(), AL))
    return;

  D->addAttr(::new (getASTContext())
                 HLSLSV_DispatchThreadIDAttr(getASTContext(), AL));
}

bool SemaHLSL::diagnosePositionType(QualType T, const ParsedAttr &AL) {
  const auto *VT = T->getAs<VectorType>();

  if (!T->hasFloatingRepresentation() || (VT && VT->getNumElements() > 4)) {
    Diag(AL.getLoc(), diag::err_hlsl_attr_invalid_type)
        << AL << "float/float1/float2/float3/float4";
    return false;
  }

  return true;
}

void SemaHLSL::handleSV_PositionAttr(Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<ValueDecl>(D);
  if (!diagnosePositionType(VD->getType(), AL))
    return;

  D->addAttr(::new (getASTContext()) HLSLSV_PositionAttr(getASTContext(), AL));
}

void SemaHLSL::handleSV_GroupThreadIDAttr(Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<ValueDecl>(D);
  if (!diagnoseInputIDType(VD->getType(), AL))
    return;

  D->addAttr(::new (getASTContext())
                 HLSLSV_GroupThreadIDAttr(getASTContext(), AL));
}

void SemaHLSL::handleSV_GroupIDAttr(Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<ValueDecl>(D);
  if (!diagnoseInputIDType(VD->getType(), AL))
    return;

  D->addAttr(::new (getASTContext()) HLSLSV_GroupIDAttr(getASTContext(), AL));
}

void SemaHLSL::handlePackOffsetAttr(Decl *D, const ParsedAttr &AL) {
  if (!isa<VarDecl>(D) || !isa<HLSLBufferDecl>(D->getDeclContext())) {
    Diag(AL.getLoc(), diag::err_hlsl_attr_invalid_ast_node)
        << AL << "shader constant in a constant buffer";
    return;
  }

  uint32_t SubComponent;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), SubComponent))
    return;
  uint32_t Component;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(1), Component))
    return;

  QualType T = cast<VarDecl>(D)->getType().getCanonicalType();
  // Check if T is an array or struct type.
  // TODO: mark matrix type as aggregate type.
  bool IsAggregateTy = (T->isArrayType() || T->isStructureType());

  // Check Component is valid for T.
  if (Component) {
    unsigned Size = getASTContext().getTypeSize(T);
    if (IsAggregateTy || Size > 128) {
      Diag(AL.getLoc(), diag::err_hlsl_packoffset_cross_reg_boundary);
      return;
    } else {
      // Make sure Component + sizeof(T) <= 4.
      if ((Component * 32 + Size) > 128) {
        Diag(AL.getLoc(), diag::err_hlsl_packoffset_cross_reg_boundary);
        return;
      }
      QualType EltTy = T;
      if (const auto *VT = T->getAs<VectorType>())
        EltTy = VT->getElementType();
      unsigned Align = getASTContext().getTypeAlign(EltTy);
      if (Align > 32 && Component == 1) {
        // NOTE: Component 3 will hit err_hlsl_packoffset_cross_reg_boundary.
        // So we only need to check Component 1 here.
        Diag(AL.getLoc(), diag::err_hlsl_packoffset_alignment_mismatch)
            << Align << EltTy;
        return;
      }
    }
  }

  D->addAttr(::new (getASTContext()) HLSLPackOffsetAttr(
      getASTContext(), AL, SubComponent, Component));
}

void SemaHLSL::handleShaderAttr(Decl *D, const ParsedAttr &AL) {
  StringRef Str;
  SourceLocation ArgLoc;
  if (!SemaRef.checkStringLiteralArgumentAttr(AL, 0, Str, &ArgLoc))
    return;

  llvm::Triple::EnvironmentType ShaderType;
  if (!HLSLShaderAttr::ConvertStrToEnvironmentType(Str, ShaderType)) {
    Diag(AL.getLoc(), diag::warn_attribute_type_not_supported)
        << AL << Str << ArgLoc;
    return;
  }

  // FIXME: check function match the shader stage.

  HLSLShaderAttr *NewAttr = mergeShaderAttr(D, AL, ShaderType);
  if (NewAttr)
    D->addAttr(NewAttr);
}

bool clang::CreateHLSLAttributedResourceType(
    Sema &S, QualType Wrapped, ArrayRef<const Attr *> AttrList,
    QualType &ResType, HLSLAttributedResourceLocInfo *LocInfo) {
  assert(AttrList.size() && "expected list of resource attributes");

  QualType ContainedTy = QualType();
  TypeSourceInfo *ContainedTyInfo = nullptr;
  SourceLocation LocBegin = AttrList[0]->getRange().getBegin();
  SourceLocation LocEnd = AttrList[0]->getRange().getEnd();

  HLSLAttributedResourceType::Attributes ResAttrs;

  bool HasResourceClass = false;
  for (const Attr *A : AttrList) {
    if (!A)
      continue;
    LocEnd = A->getRange().getEnd();
    switch (A->getKind()) {
    case attr::HLSLResourceClass: {
      ResourceClass RC = cast<HLSLResourceClassAttr>(A)->getResourceClass();
      if (HasResourceClass) {
        S.Diag(A->getLocation(), ResAttrs.ResourceClass == RC
                                     ? diag::warn_duplicate_attribute_exact
                                     : diag::warn_duplicate_attribute)
            << A;
        return false;
      }
      ResAttrs.ResourceClass = RC;
      HasResourceClass = true;
      break;
    }
    case attr::HLSLROV:
      if (ResAttrs.IsROV) {
        S.Diag(A->getLocation(), diag::warn_duplicate_attribute_exact) << A;
        return false;
      }
      ResAttrs.IsROV = true;
      break;
    case attr::HLSLRawBuffer:
      if (ResAttrs.RawBuffer) {
        S.Diag(A->getLocation(), diag::warn_duplicate_attribute_exact) << A;
        return false;
      }
      ResAttrs.RawBuffer = true;
      break;
    case attr::HLSLContainedType: {
      const HLSLContainedTypeAttr *CTAttr = cast<HLSLContainedTypeAttr>(A);
      QualType Ty = CTAttr->getType();
      if (!ContainedTy.isNull()) {
        S.Diag(A->getLocation(), ContainedTy == Ty
                                     ? diag::warn_duplicate_attribute_exact
                                     : diag::warn_duplicate_attribute)
            << A;
        return false;
      }
      ContainedTy = Ty;
      ContainedTyInfo = CTAttr->getTypeLoc();
      break;
    }
    default:
      llvm_unreachable("unhandled resource attribute type");
    }
  }

  if (!HasResourceClass) {
    S.Diag(AttrList.back()->getRange().getEnd(),
           diag::err_hlsl_missing_resource_class);
    return false;
  }

  ResType = S.getASTContext().getHLSLAttributedResourceType(
      Wrapped, ContainedTy, ResAttrs);

  if (LocInfo && ContainedTyInfo) {
    LocInfo->Range = SourceRange(LocBegin, LocEnd);
    LocInfo->ContainedTyInfo = ContainedTyInfo;
  }
  return true;
}

// Validates and creates an HLSL attribute that is applied as type attribute on
// HLSL resource. The attributes are collected in HLSLResourcesTypeAttrs and at
// the end of the declaration they are applied to the declaration type by
// wrapping it in HLSLAttributedResourceType.
bool SemaHLSL::handleResourceTypeAttr(QualType T, const ParsedAttr &AL) {
  // only allow resource type attributes on intangible types
  if (!T->isHLSLResourceType()) {
    Diag(AL.getLoc(), diag::err_hlsl_attribute_needs_intangible_type)
        << AL << getASTContext().HLSLResourceTy;
    return false;
  }

  // validate number of arguments
  if (!AL.checkExactlyNumArgs(SemaRef, AL.getMinArgs()))
    return false;

  Attr *A = nullptr;

  AttributeCommonInfo ACI(
      AL.getLoc(), AttributeScopeInfo(AL.getScopeName(), AL.getScopeLoc()),
      AttributeCommonInfo::NoSemaHandlerAttribute,
      {
          AttributeCommonInfo::AS_CXX11, 0, false /*IsAlignas*/,
          false /*IsRegularKeywordAttribute*/
      });

  switch (AL.getKind()) {
  case ParsedAttr::AT_HLSLResourceClass: {
    if (!AL.isArgIdent(0)) {
      Diag(AL.getLoc(), diag::err_attribute_argument_type)
          << AL << AANT_ArgumentIdentifier;
      return false;
    }

    IdentifierLoc *Loc = AL.getArgAsIdent(0);
    StringRef Identifier = Loc->getIdentifierInfo()->getName();
    SourceLocation ArgLoc = Loc->getLoc();

    // Validate resource class value
    ResourceClass RC;
    if (!HLSLResourceClassAttr::ConvertStrToResourceClass(Identifier, RC)) {
      Diag(ArgLoc, diag::warn_attribute_type_not_supported)
          << "ResourceClass" << Identifier;
      return false;
    }
    A = HLSLResourceClassAttr::Create(getASTContext(), RC, ACI);
    break;
  }

  case ParsedAttr::AT_HLSLROV:
    A = HLSLROVAttr::Create(getASTContext(), ACI);
    break;

  case ParsedAttr::AT_HLSLRawBuffer:
    A = HLSLRawBufferAttr::Create(getASTContext(), ACI);
    break;

  case ParsedAttr::AT_HLSLContainedType: {
    if (AL.getNumArgs() != 1 && !AL.hasParsedType()) {
      Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << AL << 1;
      return false;
    }

    TypeSourceInfo *TSI = nullptr;
    QualType QT = SemaRef.GetTypeFromParser(AL.getTypeArg(), &TSI);
    assert(TSI && "no type source info for attribute argument");
    if (SemaRef.RequireCompleteType(TSI->getTypeLoc().getBeginLoc(), QT,
                                    diag::err_incomplete_type))
      return false;
    A = HLSLContainedTypeAttr::Create(getASTContext(), TSI, ACI);
    break;
  }

  default:
    llvm_unreachable("unhandled HLSL attribute");
  }

  HLSLResourcesTypeAttrs.emplace_back(A);
  return true;
}

// Combines all resource type attributes and creates HLSLAttributedResourceType.
QualType SemaHLSL::ProcessResourceTypeAttributes(QualType CurrentType) {
  if (!HLSLResourcesTypeAttrs.size())
    return CurrentType;

  QualType QT = CurrentType;
  HLSLAttributedResourceLocInfo LocInfo;
  if (CreateHLSLAttributedResourceType(SemaRef, CurrentType,
                                       HLSLResourcesTypeAttrs, QT, &LocInfo)) {
    const HLSLAttributedResourceType *RT =
        cast<HLSLAttributedResourceType>(QT.getTypePtr());

    // Temporarily store TypeLoc information for the new type.
    // It will be transferred to HLSLAttributesResourceTypeLoc
    // shortly after the type is created by TypeSpecLocFiller which
    // will call the TakeLocForHLSLAttribute method below.
    LocsForHLSLAttributedResources.insert(std::pair(RT, LocInfo));
  }
  HLSLResourcesTypeAttrs.clear();
  return QT;
}

// Returns source location for the HLSLAttributedResourceType
HLSLAttributedResourceLocInfo
SemaHLSL::TakeLocForHLSLAttribute(const HLSLAttributedResourceType *RT) {
  HLSLAttributedResourceLocInfo LocInfo = {};
  auto I = LocsForHLSLAttributedResources.find(RT);
  if (I != LocsForHLSLAttributedResources.end()) {
    LocInfo = I->second;
    LocsForHLSLAttributedResources.erase(I);
    return LocInfo;
  }
  LocInfo.Range = SourceRange();
  return LocInfo;
}

// Walks though the global variable declaration, collects all resource binding
// requirements and adds them to Bindings
void SemaHLSL::collectResourceBindingsOnUserRecordDecl(const VarDecl *VD,
                                                       const RecordType *RT) {
  const RecordDecl *RD = RT->getOriginalDecl()->getDefinitionOrSelf();
  for (FieldDecl *FD : RD->fields()) {
    const Type *Ty = FD->getType()->getUnqualifiedDesugaredType();

    // Unwrap arrays
    // FIXME: Calculate array size while unwrapping
    assert(!Ty->isIncompleteArrayType() &&
           "incomplete arrays inside user defined types are not supported");
    while (Ty->isConstantArrayType()) {
      const ConstantArrayType *CAT = cast<ConstantArrayType>(Ty);
      Ty = CAT->getElementType()->getUnqualifiedDesugaredType();
    }

    if (!Ty->isRecordType())
      continue;

    if (const HLSLAttributedResourceType *AttrResType =
            HLSLAttributedResourceType::findHandleTypeOnResource(Ty)) {
      // Add a new DeclBindingInfo to Bindings if it does not already exist
      ResourceClass RC = AttrResType->getAttrs().ResourceClass;
      DeclBindingInfo *DBI = Bindings.getDeclBindingInfo(VD, RC);
      if (!DBI)
        Bindings.addDeclBindingInfo(VD, RC);
    } else if (const RecordType *RT = dyn_cast<RecordType>(Ty)) {
      // Recursively scan embedded struct or class; it would be nice to do this
      // without recursion, but tricky to correctly calculate the size of the
      // binding, which is something we are probably going to need to do later
      // on. Hopefully nesting of structs in structs too many levels is
      // unlikely.
      collectResourceBindingsOnUserRecordDecl(VD, RT);
    }
  }
}

// Diagnose localized register binding errors for a single binding; does not
// diagnose resource binding on user record types, that will be done later
// in processResourceBindingOnDecl based on the information collected in
// collectResourceBindingsOnVarDecl.
// Returns false if the register binding is not valid.
static bool DiagnoseLocalRegisterBinding(Sema &S, SourceLocation &ArgLoc,
                                         Decl *D, RegisterType RegType,
                                         bool SpecifiedSpace) {
  int RegTypeNum = static_cast<int>(RegType);

  // check if the decl type is groupshared
  if (D->hasAttr<HLSLGroupSharedAddressSpaceAttr>()) {
    S.Diag(ArgLoc, diag::err_hlsl_binding_type_mismatch) << RegTypeNum;
    return false;
  }

  // Cbuffers and Tbuffers are HLSLBufferDecl types
  if (HLSLBufferDecl *CBufferOrTBuffer = dyn_cast<HLSLBufferDecl>(D)) {
    ResourceClass RC = CBufferOrTBuffer->isCBuffer() ? ResourceClass::CBuffer
                                                     : ResourceClass::SRV;
    if (RegType == getRegisterType(RC))
      return true;

    S.Diag(D->getLocation(), diag::err_hlsl_binding_type_mismatch)
        << RegTypeNum;
    return false;
  }

  // Samplers, UAVs, and SRVs are VarDecl types
  assert(isa<VarDecl>(D) && "D is expected to be VarDecl or HLSLBufferDecl");
  VarDecl *VD = cast<VarDecl>(D);

  // Resource
  if (const HLSLAttributedResourceType *AttrResType =
          HLSLAttributedResourceType::findHandleTypeOnResource(
              VD->getType().getTypePtr())) {
    if (RegType == getRegisterType(AttrResType))
      return true;

    S.Diag(D->getLocation(), diag::err_hlsl_binding_type_mismatch)
        << RegTypeNum;
    return false;
  }

  const clang::Type *Ty = VD->getType().getTypePtr();
  while (Ty->isArrayType())
    Ty = Ty->getArrayElementTypeNoTypeQual();

  // Basic types
  if (Ty->isArithmeticType() || Ty->isVectorType()) {
    bool DeclaredInCOrTBuffer = isa<HLSLBufferDecl>(D->getDeclContext());
    if (SpecifiedSpace && !DeclaredInCOrTBuffer)
      S.Diag(ArgLoc, diag::err_hlsl_space_on_global_constant);

    if (!DeclaredInCOrTBuffer && (Ty->isIntegralType(S.getASTContext()) ||
                                  Ty->isFloatingType() || Ty->isVectorType())) {
      // Register annotation on default constant buffer declaration ($Globals)
      if (RegType == RegisterType::CBuffer)
        S.Diag(ArgLoc, diag::warn_hlsl_deprecated_register_type_b);
      else if (RegType != RegisterType::C)
        S.Diag(ArgLoc, diag::err_hlsl_binding_type_mismatch) << RegTypeNum;
      else
        return true;
    } else {
      if (RegType == RegisterType::C)
        S.Diag(ArgLoc, diag::warn_hlsl_register_type_c_packoffset);
      else
        S.Diag(ArgLoc, diag::err_hlsl_binding_type_mismatch) << RegTypeNum;
    }
    return false;
  }
  if (Ty->isRecordType())
    // RecordTypes will be diagnosed in processResourceBindingOnDecl
    // that is called from ActOnVariableDeclarator
    return true;

  // Anything else is an error
  S.Diag(ArgLoc, diag::err_hlsl_binding_type_mismatch) << RegTypeNum;
  return false;
}

static bool ValidateMultipleRegisterAnnotations(Sema &S, Decl *TheDecl,
                                                RegisterType regType) {
  // make sure that there are no two register annotations
  // applied to the decl with the same register type
  bool RegisterTypesDetected[5] = {false};
  RegisterTypesDetected[static_cast<int>(regType)] = true;

  for (auto it = TheDecl->attr_begin(); it != TheDecl->attr_end(); ++it) {
    if (HLSLResourceBindingAttr *attr =
            dyn_cast<HLSLResourceBindingAttr>(*it)) {

      RegisterType otherRegType = attr->getRegisterType();
      if (RegisterTypesDetected[static_cast<int>(otherRegType)]) {
        int otherRegTypeNum = static_cast<int>(otherRegType);
        S.Diag(TheDecl->getLocation(),
               diag::err_hlsl_duplicate_register_annotation)
            << otherRegTypeNum;
        return false;
      }
      RegisterTypesDetected[static_cast<int>(otherRegType)] = true;
    }
  }
  return true;
}

static bool DiagnoseHLSLRegisterAttribute(Sema &S, SourceLocation &ArgLoc,
                                          Decl *D, RegisterType RegType,
                                          bool SpecifiedSpace) {

  // exactly one of these two types should be set
  assert(((isa<VarDecl>(D) && !isa<HLSLBufferDecl>(D)) ||
          (!isa<VarDecl>(D) && isa<HLSLBufferDecl>(D))) &&
         "expecting VarDecl or HLSLBufferDecl");

  // check if the declaration contains resource matching the register type
  if (!DiagnoseLocalRegisterBinding(S, ArgLoc, D, RegType, SpecifiedSpace))
    return false;

  // next, if multiple register annotations exist, check that none conflict.
  return ValidateMultipleRegisterAnnotations(S, D, RegType);
}

void SemaHLSL::handleResourceBindingAttr(Decl *TheDecl, const ParsedAttr &AL) {
  if (isa<VarDecl>(TheDecl)) {
    if (SemaRef.RequireCompleteType(TheDecl->getBeginLoc(),
                                    cast<ValueDecl>(TheDecl)->getType(),
                                    diag::err_incomplete_type))
      return;
  }

  StringRef Slot = "";
  StringRef Space = "";
  SourceLocation SlotLoc, SpaceLoc;

  if (!AL.isArgIdent(0)) {
    Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }
  IdentifierLoc *Loc = AL.getArgAsIdent(0);

  if (AL.getNumArgs() == 2) {
    Slot = Loc->getIdentifierInfo()->getName();
    SlotLoc = Loc->getLoc();
    if (!AL.isArgIdent(1)) {
      Diag(AL.getLoc(), diag::err_attribute_argument_type)
          << AL << AANT_ArgumentIdentifier;
      return;
    }
    Loc = AL.getArgAsIdent(1);
    Space = Loc->getIdentifierInfo()->getName();
    SpaceLoc = Loc->getLoc();
  } else {
    StringRef Str = Loc->getIdentifierInfo()->getName();
    if (Str.starts_with("space")) {
      Space = Str;
      SpaceLoc = Loc->getLoc();
    } else {
      Slot = Str;
      SlotLoc = Loc->getLoc();
      Space = "space0";
    }
  }

  RegisterType RegType = RegisterType::SRV;
  std::optional<unsigned> SlotNum;
  unsigned SpaceNum = 0;

  // Validate slot
  if (!Slot.empty()) {
    if (!convertToRegisterType(Slot, &RegType)) {
      Diag(SlotLoc, diag::err_hlsl_binding_type_invalid) << Slot.substr(0, 1);
      return;
    }
    if (RegType == RegisterType::I) {
      Diag(SlotLoc, diag::warn_hlsl_deprecated_register_type_i);
      return;
    }
    StringRef SlotNumStr = Slot.substr(1);
    unsigned N;
    if (SlotNumStr.getAsInteger(10, N)) {
      Diag(SlotLoc, diag::err_hlsl_unsupported_register_number);
      return;
    }
    SlotNum = N;
  }

  // Validate space
  if (!Space.starts_with("space")) {
    Diag(SpaceLoc, diag::err_hlsl_expected_space) << Space;
    return;
  }
  StringRef SpaceNumStr = Space.substr(5);
  if (SpaceNumStr.getAsInteger(10, SpaceNum)) {
    Diag(SpaceLoc, diag::err_hlsl_expected_space) << Space;
    return;
  }

  // If we have slot, diagnose it is the right register type for the decl
  if (SlotNum.has_value())
    if (!DiagnoseHLSLRegisterAttribute(SemaRef, SlotLoc, TheDecl, RegType,
                                       !SpaceLoc.isInvalid()))
      return;

  HLSLResourceBindingAttr *NewAttr =
      HLSLResourceBindingAttr::Create(getASTContext(), Slot, Space, AL);
  if (NewAttr) {
    NewAttr->setBinding(RegType, SlotNum, SpaceNum);
    TheDecl->addAttr(NewAttr);
  }
}

void SemaHLSL::handleParamModifierAttr(Decl *D, const ParsedAttr &AL) {
  HLSLParamModifierAttr *NewAttr = mergeParamModifierAttr(
      D, AL,
      static_cast<HLSLParamModifierAttr::Spelling>(AL.getSemanticSpelling()));
  if (NewAttr)
    D->addAttr(NewAttr);
}

namespace {

/// This class implements HLSL availability diagnostics for default
/// and relaxed mode
///
/// The goal of this diagnostic is to emit an error or warning when an
/// unavailable API is found in code that is reachable from the shader
/// entry function or from an exported function (when compiling a shader
/// library).
///
/// This is done by traversing the AST of all shader entry point functions
/// and of all exported functions, and any functions that are referenced
/// from this AST. In other words, any functions that are reachable from
/// the entry points.
class DiagnoseHLSLAvailability : public DynamicRecursiveASTVisitor {
  Sema &SemaRef;

  // Stack of functions to be scaned
  llvm::SmallVector<const FunctionDecl *, 8> DeclsToScan;

  // Tracks which environments functions have been scanned in.
  //
  // Maps FunctionDecl to an unsigned number that represents the set of shader
  // environments the function has been scanned for.
  // The llvm::Triple::EnvironmentType enum values for shader stages guaranteed
  // to be numbered from llvm::Triple::Pixel to llvm::Triple::Amplification
  // (verified by static_asserts in Triple.cpp), we can use it to index
  // individual bits in the set, as long as we shift the values to start with 0
  // by subtracting the value of llvm::Triple::Pixel first.
  //
  // The N'th bit in the set will be set if the function has been scanned
  // in shader environment whose llvm::Triple::EnvironmentType integer value
  // equals (llvm::Triple::Pixel + N).
  //
  // For example, if a function has been scanned in compute and pixel stage
  // environment, the value will be 0x21 (100001 binary) because:
  //
  //   (int)(llvm::Triple::Pixel - llvm::Triple::Pixel) == 0
  //   (int)(llvm::Triple::Compute - llvm::Triple::Pixel) == 5
  //
  // A FunctionDecl is mapped to 0 (or not included in the map) if it has not
  // been scanned in any environment.
  llvm::DenseMap<const FunctionDecl *, unsigned> ScannedDecls;

  // Do not access these directly, use the get/set methods below to make
  // sure the values are in sync
  llvm::Triple::EnvironmentType CurrentShaderEnvironment;
  unsigned CurrentShaderStageBit;

  // True if scanning a function that was already scanned in a different
  // shader stage context, and therefore we should not report issues that
  // depend only on shader model version because they would be duplicate.
  bool ReportOnlyShaderStageIssues;

  // Helper methods for dealing with current stage context / environment
  void SetShaderStageContext(llvm::Triple::EnvironmentType ShaderType) {
    static_assert(sizeof(unsigned) >= 4);
    assert(HLSLShaderAttr::isValidShaderType(ShaderType));
    assert((unsigned)(ShaderType - llvm::Triple::Pixel) < 31 &&
           "ShaderType is too big for this bitmap"); // 31 is reserved for
                                                     // "unknown"

    unsigned bitmapIndex = ShaderType - llvm::Triple::Pixel;
    CurrentShaderEnvironment = ShaderType;
    CurrentShaderStageBit = (1 << bitmapIndex);
  }

  void SetUnknownShaderStageContext() {
    CurrentShaderEnvironment = llvm::Triple::UnknownEnvironment;
    CurrentShaderStageBit = (1 << 31);
  }

  llvm::Triple::EnvironmentType GetCurrentShaderEnvironment() const {
    return CurrentShaderEnvironment;
  }

  bool InUnknownShaderStageContext() const {
    return CurrentShaderEnvironment == llvm::Triple::UnknownEnvironment;
  }

  // Helper methods for dealing with shader stage bitmap
  void AddToScannedFunctions(const FunctionDecl *FD) {
    unsigned &ScannedStages = ScannedDecls[FD];
    ScannedStages |= CurrentShaderStageBit;
  }

  unsigned GetScannedStages(const FunctionDecl *FD) { return ScannedDecls[FD]; }

  bool WasAlreadyScannedInCurrentStage(const FunctionDecl *FD) {
    return WasAlreadyScannedInCurrentStage(GetScannedStages(FD));
  }

  bool WasAlreadyScannedInCurrentStage(unsigned ScannerStages) {
    return ScannerStages & CurrentShaderStageBit;
  }

  static bool NeverBeenScanned(unsigned ScannedStages) {
    return ScannedStages == 0;
  }

  // Scanning methods
  void HandleFunctionOrMethodRef(FunctionDecl *FD, Expr *RefExpr);
  void CheckDeclAvailability(NamedDecl *D, const AvailabilityAttr *AA,
                             SourceRange Range);
  const AvailabilityAttr *FindAvailabilityAttr(const Decl *D);
  bool HasMatchingEnvironmentOrNone(const AvailabilityAttr *AA);

public:
  DiagnoseHLSLAvailability(Sema &SemaRef)
      : SemaRef(SemaRef),
        CurrentShaderEnvironment(llvm::Triple::UnknownEnvironment),
        CurrentShaderStageBit(0), ReportOnlyShaderStageIssues(false) {}

  // AST traversal methods
  void RunOnTranslationUnit(const TranslationUnitDecl *TU);
  void RunOnFunction(const FunctionDecl *FD);

  bool VisitDeclRefExpr(DeclRefExpr *DRE) override {
    FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(DRE->getDecl());
    if (FD)
      HandleFunctionOrMethodRef(FD, DRE);
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) override {
    FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(ME->getMemberDecl());
    if (FD)
      HandleFunctionOrMethodRef(FD, ME);
    return true;
  }
};

void DiagnoseHLSLAvailability::HandleFunctionOrMethodRef(FunctionDecl *FD,
                                                         Expr *RefExpr) {
  assert((isa<DeclRefExpr>(RefExpr) || isa<MemberExpr>(RefExpr)) &&
         "expected DeclRefExpr or MemberExpr");

  // has a definition -> add to stack to be scanned
  const FunctionDecl *FDWithBody = nullptr;
  if (FD->hasBody(FDWithBody)) {
    if (!WasAlreadyScannedInCurrentStage(FDWithBody))
      DeclsToScan.push_back(FDWithBody);
    return;
  }

  // no body -> diagnose availability
  const AvailabilityAttr *AA = FindAvailabilityAttr(FD);
  if (AA)
    CheckDeclAvailability(
        FD, AA, SourceRange(RefExpr->getBeginLoc(), RefExpr->getEndLoc()));
}

void DiagnoseHLSLAvailability::RunOnTranslationUnit(
    const TranslationUnitDecl *TU) {

  // Iterate over all shader entry functions and library exports, and for those
  // that have a body (definiton), run diag scan on each, setting appropriate
  // shader environment context based on whether it is a shader entry function
  // or an exported function. Exported functions can be in namespaces and in
  // export declarations so we need to scan those declaration contexts as well.
  llvm::SmallVector<const DeclContext *, 8> DeclContextsToScan;
  DeclContextsToScan.push_back(TU);

  while (!DeclContextsToScan.empty()) {
    const DeclContext *DC = DeclContextsToScan.pop_back_val();
    for (auto &D : DC->decls()) {
      // do not scan implicit declaration generated by the implementation
      if (D->isImplicit())
        continue;

      // for namespace or export declaration add the context to the list to be
      // scanned later
      if (llvm::dyn_cast<NamespaceDecl>(D) || llvm::dyn_cast<ExportDecl>(D)) {
        DeclContextsToScan.push_back(llvm::dyn_cast<DeclContext>(D));
        continue;
      }

      // skip over other decls or function decls without body
      const FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(D);
      if (!FD || !FD->isThisDeclarationADefinition())
        continue;

      // shader entry point
      if (HLSLShaderAttr *ShaderAttr = FD->getAttr<HLSLShaderAttr>()) {
        SetShaderStageContext(ShaderAttr->getType());
        RunOnFunction(FD);
        continue;
      }
      // exported library function
      // FIXME: replace this loop with external linkage check once issue #92071
      // is resolved
      bool isExport = FD->isInExportDeclContext();
      if (!isExport) {
        for (const auto *Redecl : FD->redecls()) {
          if (Redecl->isInExportDeclContext()) {
            isExport = true;
            break;
          }
        }
      }
      if (isExport) {
        SetUnknownShaderStageContext();
        RunOnFunction(FD);
        continue;
      }
    }
  }
}

void DiagnoseHLSLAvailability::RunOnFunction(const FunctionDecl *FD) {
  assert(DeclsToScan.empty() && "DeclsToScan should be empty");
  DeclsToScan.push_back(FD);

  while (!DeclsToScan.empty()) {
    // Take one decl from the stack and check it by traversing its AST.
    // For any CallExpr found during the traversal add it's callee to the top of
    // the stack to be processed next. Functions already processed are stored in
    // ScannedDecls.
    const FunctionDecl *FD = DeclsToScan.pop_back_val();

    // Decl was already scanned
    const unsigned ScannedStages = GetScannedStages(FD);
    if (WasAlreadyScannedInCurrentStage(ScannedStages))
      continue;

    ReportOnlyShaderStageIssues = !NeverBeenScanned(ScannedStages);

    AddToScannedFunctions(FD);
    TraverseStmt(FD->getBody());
  }
}

bool DiagnoseHLSLAvailability::HasMatchingEnvironmentOrNone(
    const AvailabilityAttr *AA) {
  IdentifierInfo *IIEnvironment = AA->getEnvironment();
  if (!IIEnvironment)
    return true;

  llvm::Triple::EnvironmentType CurrentEnv = GetCurrentShaderEnvironment();
  if (CurrentEnv == llvm::Triple::UnknownEnvironment)
    return false;

  llvm::Triple::EnvironmentType AttrEnv =
      AvailabilityAttr::getEnvironmentType(IIEnvironment->getName());

  return CurrentEnv == AttrEnv;
}

const AvailabilityAttr *
DiagnoseHLSLAvailability::FindAvailabilityAttr(const Decl *D) {
  AvailabilityAttr const *PartialMatch = nullptr;
  // Check each AvailabilityAttr to find the one for this platform.
  // For multiple attributes with the same platform try to find one for this
  // environment.
  for (const auto *A : D->attrs()) {
    if (const auto *Avail = dyn_cast<AvailabilityAttr>(A)) {
      StringRef AttrPlatform = Avail->getPlatform()->getName();
      StringRef TargetPlatform =
          SemaRef.getASTContext().getTargetInfo().getPlatformName();

      // Match the platform name.
      if (AttrPlatform == TargetPlatform) {
        // Find the best matching attribute for this environment
        if (HasMatchingEnvironmentOrNone(Avail))
          return Avail;
        PartialMatch = Avail;
      }
    }
  }
  return PartialMatch;
}

// Check availability against target shader model version and current shader
// stage and emit diagnostic
void DiagnoseHLSLAvailability::CheckDeclAvailability(NamedDecl *D,
                                                     const AvailabilityAttr *AA,
                                                     SourceRange Range) {

  IdentifierInfo *IIEnv = AA->getEnvironment();

  if (!IIEnv) {
    // The availability attribute does not have environment -> it depends only
    // on shader model version and not on specific the shader stage.

    // Skip emitting the diagnostics if the diagnostic mode is set to
    // strict (-fhlsl-strict-availability) because all relevant diagnostics
    // were already emitted in the DiagnoseUnguardedAvailability scan
    // (SemaAvailability.cpp).
    if (SemaRef.getLangOpts().HLSLStrictAvailability)
      return;

    // Do not report shader-stage-independent issues if scanning a function
    // that was already scanned in a different shader stage context (they would
    // be duplicate)
    if (ReportOnlyShaderStageIssues)
      return;

  } else {
    // The availability attribute has environment -> we need to know
    // the current stage context to property diagnose it.
    if (InUnknownShaderStageContext())
      return;
  }

  // Check introduced version and if environment matches
  bool EnvironmentMatches = HasMatchingEnvironmentOrNone(AA);
  VersionTuple Introduced = AA->getIntroduced();
  VersionTuple TargetVersion =
      SemaRef.Context.getTargetInfo().getPlatformMinVersion();

  if (TargetVersion >= Introduced && EnvironmentMatches)
    return;

  // Emit diagnostic message
  const TargetInfo &TI = SemaRef.getASTContext().getTargetInfo();
  llvm::StringRef PlatformName(
      AvailabilityAttr::getPrettyPlatformName(TI.getPlatformName()));

  llvm::StringRef CurrentEnvStr =
      llvm::Triple::getEnvironmentTypeName(GetCurrentShaderEnvironment());

  llvm::StringRef AttrEnvStr =
      AA->getEnvironment() ? AA->getEnvironment()->getName() : "";
  bool UseEnvironment = !AttrEnvStr.empty();

  if (EnvironmentMatches) {
    SemaRef.Diag(Range.getBegin(), diag::warn_hlsl_availability)
        << Range << D << PlatformName << Introduced.getAsString()
        << UseEnvironment << CurrentEnvStr;
  } else {
    SemaRef.Diag(Range.getBegin(), diag::warn_hlsl_availability_unavailable)
        << Range << D;
  }

  SemaRef.Diag(D->getLocation(), diag::note_partial_availability_specified_here)
      << D << PlatformName << Introduced.getAsString()
      << SemaRef.Context.getTargetInfo().getPlatformMinVersion().getAsString()
      << UseEnvironment << AttrEnvStr << CurrentEnvStr;
}

} // namespace

void SemaHLSL::ActOnEndOfTranslationUnit(TranslationUnitDecl *TU) {
  // process default CBuffer - create buffer layout struct and invoke codegenCGH
  if (!DefaultCBufferDecls.empty()) {
    HLSLBufferDecl *DefaultCBuffer = HLSLBufferDecl::CreateDefaultCBuffer(
        SemaRef.getASTContext(), SemaRef.getCurLexicalContext(),
        DefaultCBufferDecls);
    addImplicitBindingAttrToDecl(SemaRef, DefaultCBuffer, RegisterType::CBuffer,
                                 getNextImplicitBindingOrderID());
    SemaRef.getCurLexicalContext()->addDecl(DefaultCBuffer);
    createHostLayoutStructForBuffer(SemaRef, DefaultCBuffer);

    // Set HasValidPackoffset if any of the decls has a register(c#) annotation;
    for (const Decl *VD : DefaultCBufferDecls) {
      const HLSLResourceBindingAttr *RBA =
          VD->getAttr<HLSLResourceBindingAttr>();
      if (RBA && RBA->hasRegisterSlot() &&
          RBA->getRegisterType() == HLSLResourceBindingAttr::RegisterType::C) {
        DefaultCBuffer->setHasValidPackoffset(true);
        break;
      }
    }

    DeclGroupRef DG(DefaultCBuffer);
    SemaRef.Consumer.HandleTopLevelDecl(DG);
  }
  diagnoseAvailabilityViolations(TU);
}

void SemaHLSL::diagnoseAvailabilityViolations(TranslationUnitDecl *TU) {
  // Skip running the diagnostics scan if the diagnostic mode is
  // strict (-fhlsl-strict-availability) and the target shader stage is known
  // because all relevant diagnostics were already emitted in the
  // DiagnoseUnguardedAvailability scan (SemaAvailability.cpp).
  const TargetInfo &TI = SemaRef.getASTContext().getTargetInfo();
  if (SemaRef.getLangOpts().HLSLStrictAvailability &&
      TI.getTriple().getEnvironment() != llvm::Triple::EnvironmentType::Library)
    return;

  DiagnoseHLSLAvailability(SemaRef).RunOnTranslationUnit(TU);
}

static bool CheckAllArgsHaveSameType(Sema *S, CallExpr *TheCall) {
  assert(TheCall->getNumArgs() > 1);
  QualType ArgTy0 = TheCall->getArg(0)->getType();

  for (unsigned I = 1, N = TheCall->getNumArgs(); I < N; ++I) {
    if (!S->getASTContext().hasSameUnqualifiedType(
            ArgTy0, TheCall->getArg(I)->getType())) {
      S->Diag(TheCall->getBeginLoc(), diag::err_vec_builtin_incompatible_vector)
          << TheCall->getDirectCallee() << /*useAllTerminology*/ true
          << SourceRange(TheCall->getArg(0)->getBeginLoc(),
                         TheCall->getArg(N - 1)->getEndLoc());
      return true;
    }
  }
  return false;
}

static bool CheckArgTypeMatches(Sema *S, Expr *Arg, QualType ExpectedType) {
  QualType ArgType = Arg->getType();
  if (!S->getASTContext().hasSameUnqualifiedType(ArgType, ExpectedType)) {
    S->Diag(Arg->getBeginLoc(), diag::err_typecheck_convert_incompatible)
        << ArgType << ExpectedType << 1 << 0 << 0;
    return true;
  }
  return false;
}

static bool CheckAllArgTypesAreCorrect(
    Sema *S, CallExpr *TheCall,
    llvm::function_ref<bool(Sema *S, SourceLocation Loc, int ArgOrdinal,
                            clang::QualType PassedType)>
        Check) {
  for (unsigned I = 0; I < TheCall->getNumArgs(); ++I) {
    Expr *Arg = TheCall->getArg(I);
    if (Check(S, Arg->getBeginLoc(), I + 1, Arg->getType()))
      return true;
  }
  return false;
}

static bool CheckFloatOrHalfRepresentation(Sema *S, SourceLocation Loc,
                                           int ArgOrdinal,
                                           clang::QualType PassedType) {
  clang::QualType BaseType =
      PassedType->isVectorType()
          ? PassedType->castAs<clang::VectorType>()->getElementType()
          : PassedType;
  if (!BaseType->isHalfType() && !BaseType->isFloat32Type())
    return S->Diag(Loc, diag::err_builtin_invalid_arg_type)
           << ArgOrdinal << /* scalar or vector of */ 5 << /* no int */ 0
           << /* half or float */ 2 << PassedType;
  return false;
}

static bool CheckModifiableLValue(Sema *S, CallExpr *TheCall,
                                  unsigned ArgIndex) {
  auto *Arg = TheCall->getArg(ArgIndex);
  SourceLocation OrigLoc = Arg->getExprLoc();
  if (Arg->IgnoreCasts()->isModifiableLvalue(S->Context, &OrigLoc) ==
      Expr::MLV_Valid)
    return false;
  S->Diag(OrigLoc, diag::error_hlsl_inout_lvalue) << Arg << 0;
  return true;
}

static bool CheckNoDoubleVectors(Sema *S, SourceLocation Loc, int ArgOrdinal,
                                 clang::QualType PassedType) {
  const auto *VecTy = PassedType->getAs<VectorType>();
  if (!VecTy)
    return false;

  if (VecTy->getElementType()->isDoubleType())
    return S->Diag(Loc, diag::err_builtin_invalid_arg_type)
           << ArgOrdinal << /* scalar */ 1 << /* no int */ 0 << /* fp */ 1
           << PassedType;
  return false;
}

static bool CheckFloatingOrIntRepresentation(Sema *S, SourceLocation Loc,
                                             int ArgOrdinal,
                                             clang::QualType PassedType) {
  if (!PassedType->hasIntegerRepresentation() &&
      !PassedType->hasFloatingRepresentation())
    return S->Diag(Loc, diag::err_builtin_invalid_arg_type)
           << ArgOrdinal << /* scalar or vector of */ 5 << /* integer */ 1
           << /* fp */ 1 << PassedType;
  return false;
}

static bool CheckUnsignedIntVecRepresentation(Sema *S, SourceLocation Loc,
                                              int ArgOrdinal,
                                              clang::QualType PassedType) {
  if (auto *VecTy = PassedType->getAs<VectorType>())
    if (VecTy->getElementType()->isUnsignedIntegerType())
      return false;

  return S->Diag(Loc, diag::err_builtin_invalid_arg_type)
         << ArgOrdinal << /* vector of */ 4 << /* uint */ 3 << /* no fp */ 0
         << PassedType;
}

// checks for unsigned ints of all sizes
static bool CheckUnsignedIntRepresentation(Sema *S, SourceLocation Loc,
                                           int ArgOrdinal,
                                           clang::QualType PassedType) {
  if (!PassedType->hasUnsignedIntegerRepresentation())
    return S->Diag(Loc, diag::err_builtin_invalid_arg_type)
           << ArgOrdinal << /* scalar or vector of */ 5 << /* unsigned int */ 3
           << /* no fp */ 0 << PassedType;
  return false;
}

static void SetElementTypeAsReturnType(Sema *S, CallExpr *TheCall,
                                       QualType ReturnType) {
  auto *VecTyA = TheCall->getArg(0)->getType()->getAs<VectorType>();
  if (VecTyA)
    ReturnType =
        S->Context.getExtVectorType(ReturnType, VecTyA->getNumElements());

  TheCall->setType(ReturnType);
}

static bool CheckScalarOrVector(Sema *S, CallExpr *TheCall, QualType Scalar,
                                unsigned ArgIndex) {
  assert(TheCall->getNumArgs() >= ArgIndex);
  QualType ArgType = TheCall->getArg(ArgIndex)->getType();
  auto *VTy = ArgType->getAs<VectorType>();
  // not the scalar or vector<scalar>
  if (!(S->Context.hasSameUnqualifiedType(ArgType, Scalar) ||
        (VTy &&
         S->Context.hasSameUnqualifiedType(VTy->getElementType(), Scalar)))) {
    S->Diag(TheCall->getArg(0)->getBeginLoc(),
            diag::err_typecheck_expect_scalar_or_vector)
        << ArgType << Scalar;
    return true;
  }
  return false;
}

static bool CheckAnyScalarOrVector(Sema *S, CallExpr *TheCall,
                                   unsigned ArgIndex) {
  assert(TheCall->getNumArgs() >= ArgIndex);
  QualType ArgType = TheCall->getArg(ArgIndex)->getType();
  auto *VTy = ArgType->getAs<VectorType>();
  // not the scalar or vector<scalar>
  if (!(ArgType->isScalarType() ||
        (VTy && VTy->getElementType()->isScalarType()))) {
    S->Diag(TheCall->getArg(0)->getBeginLoc(),
            diag::err_typecheck_expect_any_scalar_or_vector)
        << ArgType << 1;
    return true;
  }
  return false;
}

static bool CheckWaveActive(Sema *S, CallExpr *TheCall) {
  QualType BoolType = S->getASTContext().BoolTy;
  assert(TheCall->getNumArgs() >= 1);
  QualType ArgType = TheCall->getArg(0)->getType();
  auto *VTy = ArgType->getAs<VectorType>();
  // is the bool or vector<bool>
  if (S->Context.hasSameUnqualifiedType(ArgType, BoolType) ||
      (VTy &&
       S->Context.hasSameUnqualifiedType(VTy->getElementType(), BoolType))) {
    S->Diag(TheCall->getArg(0)->getBeginLoc(),
            diag::err_typecheck_expect_any_scalar_or_vector)
        << ArgType << 0;
    return true;
  }
  return false;
}

static bool CheckBoolSelect(Sema *S, CallExpr *TheCall) {
  assert(TheCall->getNumArgs() == 3);
  Expr *Arg1 = TheCall->getArg(1);
  Expr *Arg2 = TheCall->getArg(2);
  if (!S->Context.hasSameUnqualifiedType(Arg1->getType(), Arg2->getType())) {
    S->Diag(TheCall->getBeginLoc(),
            diag::err_typecheck_call_different_arg_types)
        << Arg1->getType() << Arg2->getType() << Arg1->getSourceRange()
        << Arg2->getSourceRange();
    return true;
  }

  TheCall->setType(Arg1->getType());
  return false;
}

static bool CheckVectorSelect(Sema *S, CallExpr *TheCall) {
  assert(TheCall->getNumArgs() == 3);
  Expr *Arg1 = TheCall->getArg(1);
  QualType Arg1Ty = Arg1->getType();
  Expr *Arg2 = TheCall->getArg(2);
  QualType Arg2Ty = Arg2->getType();

  QualType Arg1ScalarTy = Arg1Ty;
  if (auto VTy = Arg1ScalarTy->getAs<VectorType>())
    Arg1ScalarTy = VTy->getElementType();

  QualType Arg2ScalarTy = Arg2Ty;
  if (auto VTy = Arg2ScalarTy->getAs<VectorType>())
    Arg2ScalarTy = VTy->getElementType();

  if (!S->Context.hasSameUnqualifiedType(Arg1ScalarTy, Arg2ScalarTy))
    S->Diag(Arg1->getBeginLoc(), diag::err_hlsl_builtin_scalar_vector_mismatch)
        << /* second and third */ 1 << TheCall->getCallee() << Arg1Ty << Arg2Ty;

  QualType Arg0Ty = TheCall->getArg(0)->getType();
  unsigned Arg0Length = Arg0Ty->getAs<VectorType>()->getNumElements();
  unsigned Arg1Length = Arg1Ty->isVectorType()
                            ? Arg1Ty->getAs<VectorType>()->getNumElements()
                            : 0;
  unsigned Arg2Length = Arg2Ty->isVectorType()
                            ? Arg2Ty->getAs<VectorType>()->getNumElements()
                            : 0;
  if (Arg1Length > 0 && Arg0Length != Arg1Length) {
    S->Diag(TheCall->getBeginLoc(),
            diag::err_typecheck_vector_lengths_not_equal)
        << Arg0Ty << Arg1Ty << TheCall->getArg(0)->getSourceRange()
        << Arg1->getSourceRange();
    return true;
  }

  if (Arg2Length > 0 && Arg0Length != Arg2Length) {
    S->Diag(TheCall->getBeginLoc(),
            diag::err_typecheck_vector_lengths_not_equal)
        << Arg0Ty << Arg2Ty << TheCall->getArg(0)->getSourceRange()
        << Arg2->getSourceRange();
    return true;
  }

  TheCall->setType(
      S->getASTContext().getExtVectorType(Arg1ScalarTy, Arg0Length));
  return false;
}

static bool CheckResourceHandle(
    Sema *S, CallExpr *TheCall, unsigned ArgIndex,
    llvm::function_ref<bool(const HLSLAttributedResourceType *ResType)> Check =
        nullptr) {
  assert(TheCall->getNumArgs() >= ArgIndex);
  QualType ArgType = TheCall->getArg(ArgIndex)->getType();
  const HLSLAttributedResourceType *ResTy =
      ArgType.getTypePtr()->getAs<HLSLAttributedResourceType>();
  if (!ResTy) {
    S->Diag(TheCall->getArg(ArgIndex)->getBeginLoc(),
            diag::err_typecheck_expect_hlsl_resource)
        << ArgType;
    return true;
  }
  if (Check && Check(ResTy)) {
    S->Diag(TheCall->getArg(ArgIndex)->getExprLoc(),
            diag::err_invalid_hlsl_resource_type)
        << ArgType;
    return true;
  }
  return false;
}

// Note: returning true in this case results in CheckBuiltinFunctionCall
// returning an ExprError
bool SemaHLSL::CheckBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall) {
  switch (BuiltinID) {
  case Builtin::BI__builtin_hlsl_adduint64: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;

    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckUnsignedIntVecRepresentation))
      return true;

    auto *VTy = TheCall->getArg(0)->getType()->getAs<VectorType>();
    // ensure arg integers are 32-bits
    uint64_t ElementBitCount = getASTContext()
                                   .getTypeSizeInChars(VTy->getElementType())
                                   .getQuantity() *
                               8;
    if (ElementBitCount != 32) {
      SemaRef.Diag(TheCall->getBeginLoc(),
                   diag::err_integer_incorrect_bit_count)
          << 32 << ElementBitCount;
      return true;
    }

    // ensure both args are vectors of total bit size of a multiple of 64
    int NumElementsArg = VTy->getNumElements();
    if (NumElementsArg != 2 && NumElementsArg != 4) {
      SemaRef.Diag(TheCall->getBeginLoc(), diag::err_vector_incorrect_bit_count)
          << 1 /*a multiple of*/ << 64 << NumElementsArg * ElementBitCount;
      return true;
    }

    // ensure first arg and second arg have the same type
    if (CheckAllArgsHaveSameType(&SemaRef, TheCall))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    // return type is the same as the input type
    TheCall->setType(ArgTyA);
    break;
  }
  case Builtin::BI__builtin_hlsl_resource_getpointer: {
    if (SemaRef.checkArgCount(TheCall, 2) ||
        CheckResourceHandle(&SemaRef, TheCall, 0) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(1),
                            SemaRef.getASTContext().UnsignedIntTy))
      return true;

    auto *ResourceTy =
        TheCall->getArg(0)->getType()->castAs<HLSLAttributedResourceType>();
    QualType ContainedTy = ResourceTy->getContainedType();
    auto ReturnType =
        SemaRef.Context.getAddrSpaceQualType(ContainedTy, LangAS::hlsl_device);
    ReturnType = SemaRef.Context.getPointerType(ReturnType);
    TheCall->setType(ReturnType);
    TheCall->setValueKind(VK_LValue);

    break;
  }
  case Builtin::BI__builtin_hlsl_resource_uninitializedhandle: {
    if (SemaRef.checkArgCount(TheCall, 1) ||
        CheckResourceHandle(&SemaRef, TheCall, 0))
      return true;
    // use the type of the handle (arg0) as a return type
    QualType ResourceTy = TheCall->getArg(0)->getType();
    TheCall->setType(ResourceTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_resource_handlefrombinding: {
    ASTContext &AST = SemaRef.getASTContext();
    if (SemaRef.checkArgCount(TheCall, 6) ||
        CheckResourceHandle(&SemaRef, TheCall, 0) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(1), AST.UnsignedIntTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(2), AST.UnsignedIntTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(3), AST.IntTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(4), AST.UnsignedIntTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(5),
                            AST.getPointerType(AST.CharTy.withConst())))
      return true;
    // use the type of the handle (arg0) as a return type
    QualType ResourceTy = TheCall->getArg(0)->getType();
    TheCall->setType(ResourceTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_resource_handlefromimplicitbinding: {
    ASTContext &AST = SemaRef.getASTContext();
    if (SemaRef.checkArgCount(TheCall, 6) ||
        CheckResourceHandle(&SemaRef, TheCall, 0) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(1), AST.UnsignedIntTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(2), AST.IntTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(3), AST.UnsignedIntTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(4), AST.UnsignedIntTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(5),
                            AST.getPointerType(AST.CharTy.withConst())))
      return true;
    // use the type of the handle (arg0) as a return type
    QualType ResourceTy = TheCall->getArg(0)->getType();
    TheCall->setType(ResourceTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_and:
  case Builtin::BI__builtin_hlsl_or: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;
    if (CheckScalarOrVector(&SemaRef, TheCall, getASTContext().BoolTy, 0))
      return true;
    if (CheckAllArgsHaveSameType(&SemaRef, TheCall))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    // return type is the same as the input type
    TheCall->setType(ArgTyA);
    break;
  }
  case Builtin::BI__builtin_hlsl_all:
  case Builtin::BI__builtin_hlsl_any: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;
    if (CheckAnyScalarOrVector(&SemaRef, TheCall, 0))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_asdouble: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;
    if (CheckScalarOrVector(
            &SemaRef, TheCall,
            /*only check for uint*/ SemaRef.Context.UnsignedIntTy,
            /* arg index */ 0))
      return true;
    if (CheckScalarOrVector(
            &SemaRef, TheCall,
            /*only check for uint*/ SemaRef.Context.UnsignedIntTy,
            /* arg index */ 1))
      return true;
    if (CheckAllArgsHaveSameType(&SemaRef, TheCall))
      return true;

    SetElementTypeAsReturnType(&SemaRef, TheCall, getASTContext().DoubleTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_clamp: {
    if (SemaRef.BuiltinElementwiseTernaryMath(
            TheCall, /*ArgTyRestr=*/
            Sema::EltwiseBuiltinArgTyRestriction::None))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_dot: {
    // arg count is checked by BuiltinVectorToScalarMath
    if (SemaRef.BuiltinVectorToScalarMath(TheCall))
      return true;
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall, CheckNoDoubleVectors))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_firstbithigh:
  case Builtin::BI__builtin_hlsl_elementwise_firstbitlow: {
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;

    const Expr *Arg = TheCall->getArg(0);
    QualType ArgTy = Arg->getType();
    QualType EltTy = ArgTy;

    QualType ResTy = SemaRef.Context.UnsignedIntTy;

    if (auto *VecTy = EltTy->getAs<VectorType>()) {
      EltTy = VecTy->getElementType();
      ResTy = SemaRef.Context.getExtVectorType(ResTy, VecTy->getNumElements());
    }

    if (!EltTy->isIntegerType()) {
      Diag(Arg->getBeginLoc(), diag::err_builtin_invalid_arg_type)
          << 1 << /* scalar or vector of */ 5 << /* integer ty */ 1
          << /* no fp */ 0 << ArgTy;
      return true;
    }

    TheCall->setType(ResTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_select: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;
    if (CheckScalarOrVector(&SemaRef, TheCall, getASTContext().BoolTy, 0))
      return true;
    QualType ArgTy = TheCall->getArg(0)->getType();
    if (ArgTy->isBooleanType() && CheckBoolSelect(&SemaRef, TheCall))
      return true;
    auto *VTy = ArgTy->getAs<VectorType>();
    if (VTy && VTy->getElementType()->isBooleanType() &&
        CheckVectorSelect(&SemaRef, TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_saturate:
  case Builtin::BI__builtin_hlsl_elementwise_rcp: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;
    if (!TheCall->getArg(0)
             ->getType()
             ->hasFloatingRepresentation()) // half or float or double
      return SemaRef.Diag(TheCall->getArg(0)->getBeginLoc(),
                          diag::err_builtin_invalid_arg_type)
             << /* ordinal */ 1 << /* scalar or vector */ 5 << /* no int */ 0
             << /* fp */ 1 << TheCall->getArg(0)->getType();
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_degrees:
  case Builtin::BI__builtin_hlsl_elementwise_radians:
  case Builtin::BI__builtin_hlsl_elementwise_rsqrt:
  case Builtin::BI__builtin_hlsl_elementwise_frac: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckFloatOrHalfRepresentation))
      return true;
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_isinf: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckFloatOrHalfRepresentation))
      return true;
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;
    SetElementTypeAsReturnType(&SemaRef, TheCall, getASTContext().BoolTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_lerp: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckFloatOrHalfRepresentation))
      return true;
    if (CheckAllArgsHaveSameType(&SemaRef, TheCall))
      return true;
    if (SemaRef.BuiltinElementwiseTernaryMath(TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_mad: {
    if (SemaRef.BuiltinElementwiseTernaryMath(
            TheCall, /*ArgTyRestr=*/
            Sema::EltwiseBuiltinArgTyRestriction::None))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_normalize: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckFloatOrHalfRepresentation))
      return true;
    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    // return type is the same as the input type
    TheCall->setType(ArgTyA);
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_sign: {
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckFloatingOrIntRepresentation))
      return true;
    SetElementTypeAsReturnType(&SemaRef, TheCall, getASTContext().IntTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_step: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckFloatOrHalfRepresentation))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    // return type is the same as the input type
    TheCall->setType(ArgTyA);
    break;
  }
  case Builtin::BI__builtin_hlsl_wave_active_max:
  case Builtin::BI__builtin_hlsl_wave_active_sum: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;

    // Ensure input expr type is a scalar/vector and the same as the return type
    if (CheckAnyScalarOrVector(&SemaRef, TheCall, 0))
      return true;
    if (CheckWaveActive(&SemaRef, TheCall))
      return true;
    ExprResult Expr = TheCall->getArg(0);
    QualType ArgTyExpr = Expr.get()->getType();
    TheCall->setType(ArgTyExpr);
    break;
  }
  // Note these are llvm builtins that we want to catch invalid intrinsic
  // generation. Normal handling of these builitns will occur elsewhere.
  case Builtin::BI__builtin_elementwise_bitreverse: {
    // does not include a check for number of arguments
    // because that is done previously
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckUnsignedIntRepresentation))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_wave_read_lane_at: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;

    // Ensure index parameter type can be interpreted as a uint
    ExprResult Index = TheCall->getArg(1);
    QualType ArgTyIndex = Index.get()->getType();
    if (!ArgTyIndex->isIntegerType()) {
      SemaRef.Diag(TheCall->getArg(1)->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyIndex << SemaRef.Context.UnsignedIntTy << 1 << 0 << 0;
      return true;
    }

    // Ensure input expr type is a scalar/vector and the same as the return type
    if (CheckAnyScalarOrVector(&SemaRef, TheCall, 0))
      return true;

    ExprResult Expr = TheCall->getArg(0);
    QualType ArgTyExpr = Expr.get()->getType();
    TheCall->setType(ArgTyExpr);
    break;
  }
  case Builtin::BI__builtin_hlsl_wave_get_lane_index: {
    if (SemaRef.checkArgCount(TheCall, 0))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_splitdouble: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    if (CheckScalarOrVector(&SemaRef, TheCall, SemaRef.Context.DoubleTy, 0) ||
        CheckScalarOrVector(&SemaRef, TheCall, SemaRef.Context.UnsignedIntTy,
                            1) ||
        CheckScalarOrVector(&SemaRef, TheCall, SemaRef.Context.UnsignedIntTy,
                            2))
      return true;

    if (CheckModifiableLValue(&SemaRef, TheCall, 1) ||
        CheckModifiableLValue(&SemaRef, TheCall, 2))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_clip: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;

    if (CheckScalarOrVector(&SemaRef, TheCall, SemaRef.Context.FloatTy, 0))
      return true;
    break;
  }
  case Builtin::BI__builtin_elementwise_acos:
  case Builtin::BI__builtin_elementwise_asin:
  case Builtin::BI__builtin_elementwise_atan:
  case Builtin::BI__builtin_elementwise_atan2:
  case Builtin::BI__builtin_elementwise_ceil:
  case Builtin::BI__builtin_elementwise_cos:
  case Builtin::BI__builtin_elementwise_cosh:
  case Builtin::BI__builtin_elementwise_exp:
  case Builtin::BI__builtin_elementwise_exp2:
  case Builtin::BI__builtin_elementwise_exp10:
  case Builtin::BI__builtin_elementwise_floor:
  case Builtin::BI__builtin_elementwise_fmod:
  case Builtin::BI__builtin_elementwise_log:
  case Builtin::BI__builtin_elementwise_log2:
  case Builtin::BI__builtin_elementwise_log10:
  case Builtin::BI__builtin_elementwise_pow:
  case Builtin::BI__builtin_elementwise_roundeven:
  case Builtin::BI__builtin_elementwise_sin:
  case Builtin::BI__builtin_elementwise_sinh:
  case Builtin::BI__builtin_elementwise_sqrt:
  case Builtin::BI__builtin_elementwise_tan:
  case Builtin::BI__builtin_elementwise_tanh:
  case Builtin::BI__builtin_elementwise_trunc: {
    if (CheckAllArgTypesAreCorrect(&SemaRef, TheCall,
                                   CheckFloatOrHalfRepresentation))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_buffer_update_counter: {
    auto checkResTy = [](const HLSLAttributedResourceType *ResTy) -> bool {
      return !(ResTy->getAttrs().ResourceClass == ResourceClass::UAV &&
               ResTy->getAttrs().RawBuffer && ResTy->hasContainedType());
    };
    if (SemaRef.checkArgCount(TheCall, 2) ||
        CheckResourceHandle(&SemaRef, TheCall, 0, checkResTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(1),
                            SemaRef.getASTContext().IntTy))
      return true;
    Expr *OffsetExpr = TheCall->getArg(1);
    std::optional<llvm::APSInt> Offset =
        OffsetExpr->getIntegerConstantExpr(SemaRef.getASTContext());
    if (!Offset.has_value() || std::abs(Offset->getExtValue()) != 1) {
      SemaRef.Diag(TheCall->getArg(1)->getBeginLoc(),
                   diag::err_hlsl_expect_arg_const_int_one_or_neg_one)
          << 1;
      return true;
    }
    break;
  }
  }
  return false;
}

static void BuildFlattenedTypeList(QualType BaseTy,
                                   llvm::SmallVectorImpl<QualType> &List) {
  llvm::SmallVector<QualType, 16> WorkList;
  WorkList.push_back(BaseTy);
  while (!WorkList.empty()) {
    QualType T = WorkList.pop_back_val();
    T = T.getCanonicalType().getUnqualifiedType();
    assert(!isa<MatrixType>(T) && "Matrix types not yet supported in HLSL");
    if (const auto *AT = dyn_cast<ConstantArrayType>(T)) {
      llvm::SmallVector<QualType, 16> ElementFields;
      // Generally I've avoided recursion in this algorithm, but arrays of
      // structs could be time-consuming to flatten and churn through on the
      // work list. Hopefully nesting arrays of structs containing arrays
      // of structs too many levels deep is unlikely.
      BuildFlattenedTypeList(AT->getElementType(), ElementFields);
      // Repeat the element's field list n times.
      for (uint64_t Ct = 0; Ct < AT->getZExtSize(); ++Ct)
        llvm::append_range(List, ElementFields);
      continue;
    }
    // Vectors can only have element types that are builtin types, so this can
    // add directly to the list instead of to the WorkList.
    if (const auto *VT = dyn_cast<VectorType>(T)) {
      List.insert(List.end(), VT->getNumElements(), VT->getElementType());
      continue;
    }
    if (const auto *RT = dyn_cast<RecordType>(T)) {
      const CXXRecordDecl *RD = RT->getAsCXXRecordDecl();
      assert(RD && "HLSL record types should all be CXXRecordDecls!");

      if (RD->isStandardLayout())
        RD = RD->getStandardLayoutBaseWithFields();

      // For types that we shouldn't decompose (unions and non-aggregates), just
      // add the type itself to the list.
      if (RD->isUnion() || !RD->isAggregate()) {
        List.push_back(T);
        continue;
      }

      llvm::SmallVector<QualType, 16> FieldTypes;
      for (const auto *FD : RD->fields())
        FieldTypes.push_back(FD->getType());
      // Reverse the newly added sub-range.
      std::reverse(FieldTypes.begin(), FieldTypes.end());
      llvm::append_range(WorkList, FieldTypes);

      // If this wasn't a standard layout type we may also have some base
      // classes to deal with.
      if (!RD->isStandardLayout()) {
        FieldTypes.clear();
        for (const auto &Base : RD->bases())
          FieldTypes.push_back(Base.getType());
        std::reverse(FieldTypes.begin(), FieldTypes.end());
        llvm::append_range(WorkList, FieldTypes);
      }
      continue;
    }
    List.push_back(T);
  }
}

bool SemaHLSL::IsTypedResourceElementCompatible(clang::QualType QT) {
  // null and array types are not allowed.
  if (QT.isNull() || QT->isArrayType())
    return false;

  // UDT types are not allowed
  if (QT->isRecordType())
    return false;

  if (QT->isBooleanType() || QT->isEnumeralType())
    return false;

  // the only other valid builtin types are scalars or vectors
  if (QT->isArithmeticType()) {
    if (SemaRef.Context.getTypeSize(QT) / 8 > 16)
      return false;
    return true;
  }

  if (const VectorType *VT = QT->getAs<VectorType>()) {
    int ArraySize = VT->getNumElements();

    if (ArraySize > 4)
      return false;

    QualType ElTy = VT->getElementType();
    if (ElTy->isBooleanType())
      return false;

    if (SemaRef.Context.getTypeSize(QT) / 8 > 16)
      return false;
    return true;
  }

  return false;
}

bool SemaHLSL::IsScalarizedLayoutCompatible(QualType T1, QualType T2) const {
  if (T1.isNull() || T2.isNull())
    return false;

  T1 = T1.getCanonicalType().getUnqualifiedType();
  T2 = T2.getCanonicalType().getUnqualifiedType();

  // If both types are the same canonical type, they're obviously compatible.
  if (SemaRef.getASTContext().hasSameType(T1, T2))
    return true;

  llvm::SmallVector<QualType, 16> T1Types;
  BuildFlattenedTypeList(T1, T1Types);
  llvm::SmallVector<QualType, 16> T2Types;
  BuildFlattenedTypeList(T2, T2Types);

  // Check the flattened type list
  return llvm::equal(T1Types, T2Types,
                     [this](QualType LHS, QualType RHS) -> bool {
                       return SemaRef.IsLayoutCompatible(LHS, RHS);
                     });
}

bool SemaHLSL::CheckCompatibleParameterABI(FunctionDecl *New,
                                           FunctionDecl *Old) {
  if (New->getNumParams() != Old->getNumParams())
    return true;

  bool HadError = false;

  for (unsigned i = 0, e = New->getNumParams(); i != e; ++i) {
    ParmVarDecl *NewParam = New->getParamDecl(i);
    ParmVarDecl *OldParam = Old->getParamDecl(i);

    // HLSL parameter declarations for inout and out must match between
    // declarations. In HLSL inout and out are ambiguous at the call site,
    // but have different calling behavior, so you cannot overload a
    // method based on a difference between inout and out annotations.
    const auto *NDAttr = NewParam->getAttr<HLSLParamModifierAttr>();
    unsigned NSpellingIdx = (NDAttr ? NDAttr->getSpellingListIndex() : 0);
    const auto *ODAttr = OldParam->getAttr<HLSLParamModifierAttr>();
    unsigned OSpellingIdx = (ODAttr ? ODAttr->getSpellingListIndex() : 0);

    if (NSpellingIdx != OSpellingIdx) {
      SemaRef.Diag(NewParam->getLocation(),
                   diag::err_hlsl_param_qualifier_mismatch)
          << NDAttr << NewParam;
      SemaRef.Diag(OldParam->getLocation(), diag::note_previous_declaration_as)
          << ODAttr;
      HadError = true;
    }
  }
  return HadError;
}

// Generally follows PerformScalarCast, with cases reordered for
// clarity of what types are supported
bool SemaHLSL::CanPerformScalarCast(QualType SrcTy, QualType DestTy) {

  if (!SrcTy->isScalarType() || !DestTy->isScalarType())
    return false;

  if (SemaRef.getASTContext().hasSameUnqualifiedType(SrcTy, DestTy))
    return true;

  switch (SrcTy->getScalarTypeKind()) {
  case Type::STK_Bool: // casting from bool is like casting from an integer
  case Type::STK_Integral:
    switch (DestTy->getScalarTypeKind()) {
    case Type::STK_Bool:
    case Type::STK_Integral:
    case Type::STK_Floating:
      return true;
    case Type::STK_CPointer:
    case Type::STK_ObjCObjectPointer:
    case Type::STK_BlockPointer:
    case Type::STK_MemberPointer:
      llvm_unreachable("HLSL doesn't support pointers.");
    case Type::STK_IntegralComplex:
    case Type::STK_FloatingComplex:
      llvm_unreachable("HLSL doesn't support complex types.");
    case Type::STK_FixedPoint:
      llvm_unreachable("HLSL doesn't support fixed point types.");
    }
    llvm_unreachable("Should have returned before this");

  case Type::STK_Floating:
    switch (DestTy->getScalarTypeKind()) {
    case Type::STK_Floating:
    case Type::STK_Bool:
    case Type::STK_Integral:
      return true;
    case Type::STK_FloatingComplex:
    case Type::STK_IntegralComplex:
      llvm_unreachable("HLSL doesn't support complex types.");
    case Type::STK_FixedPoint:
      llvm_unreachable("HLSL doesn't support fixed point types.");
    case Type::STK_CPointer:
    case Type::STK_ObjCObjectPointer:
    case Type::STK_BlockPointer:
    case Type::STK_MemberPointer:
      llvm_unreachable("HLSL doesn't support pointers.");
    }
    llvm_unreachable("Should have returned before this");

  case Type::STK_MemberPointer:
  case Type::STK_CPointer:
  case Type::STK_BlockPointer:
  case Type::STK_ObjCObjectPointer:
    llvm_unreachable("HLSL doesn't support pointers.");

  case Type::STK_FixedPoint:
    llvm_unreachable("HLSL doesn't support fixed point types.");

  case Type::STK_FloatingComplex:
  case Type::STK_IntegralComplex:
    llvm_unreachable("HLSL doesn't support complex types.");
  }

  llvm_unreachable("Unhandled scalar cast");
}

// Detect if a type contains a bitfield. Will be removed when
// bitfield support is added to HLSLElementwiseCast and HLSLAggregateSplatCast
bool SemaHLSL::ContainsBitField(QualType BaseTy) {
  llvm::SmallVector<QualType, 16> WorkList;
  WorkList.push_back(BaseTy);
  while (!WorkList.empty()) {
    QualType T = WorkList.pop_back_val();
    T = T.getCanonicalType().getUnqualifiedType();
    // only check aggregate types
    if (const auto *AT = dyn_cast<ConstantArrayType>(T)) {
      WorkList.push_back(AT->getElementType());
      continue;
    }
    if (const auto *RT = dyn_cast<RecordType>(T)) {
      const RecordDecl *RD = RT->getOriginalDecl()->getDefinitionOrSelf();
      if (RD->isUnion())
        continue;

      const CXXRecordDecl *CXXD = dyn_cast<CXXRecordDecl>(RD);

      if (CXXD && CXXD->isStandardLayout())
        RD = CXXD->getStandardLayoutBaseWithFields();

      for (const auto *FD : RD->fields()) {
        if (FD->isBitField())
          return true;
        WorkList.push_back(FD->getType());
      }
      continue;
    }
  }
  return false;
}

// Can perform an HLSL Aggregate splat cast if the Dest is an aggregate and the
// Src is a scalar or a vector of length 1
// Or if Dest is a vector and Src is a vector of length 1
bool SemaHLSL::CanPerformAggregateSplatCast(Expr *Src, QualType DestTy) {

  QualType SrcTy = Src->getType();
  // Not a valid HLSL Aggregate Splat cast if Dest is a scalar or if this is
  // going to be a vector splat from a scalar.
  if ((SrcTy->isScalarType() && DestTy->isVectorType()) ||
      DestTy->isScalarType())
    return false;

  const VectorType *SrcVecTy = SrcTy->getAs<VectorType>();

  // Src isn't a scalar or a vector of length 1
  if (!SrcTy->isScalarType() && !(SrcVecTy && SrcVecTy->getNumElements() == 1))
    return false;

  if (SrcVecTy)
    SrcTy = SrcVecTy->getElementType();

  if (ContainsBitField(DestTy))
    return false;

  llvm::SmallVector<QualType> DestTypes;
  BuildFlattenedTypeList(DestTy, DestTypes);

  for (unsigned I = 0, Size = DestTypes.size(); I < Size; ++I) {
    if (DestTypes[I]->isUnionType())
      return false;
    if (!CanPerformScalarCast(SrcTy, DestTypes[I]))
      return false;
  }
  return true;
}

// Can we perform an HLSL Elementwise cast?
// TODO: update this code when matrices are added; see issue #88060
bool SemaHLSL::CanPerformElementwiseCast(Expr *Src, QualType DestTy) {

  // Don't handle casts where LHS and RHS are any combination of scalar/vector
  // There must be an aggregate somewhere
  QualType SrcTy = Src->getType();
  if (SrcTy->isScalarType()) // always a splat and this cast doesn't handle that
    return false;

  if (SrcTy->isVectorType() &&
      (DestTy->isScalarType() || DestTy->isVectorType()))
    return false;

  if (ContainsBitField(DestTy) || ContainsBitField(SrcTy))
    return false;

  llvm::SmallVector<QualType> DestTypes;
  BuildFlattenedTypeList(DestTy, DestTypes);
  llvm::SmallVector<QualType> SrcTypes;
  BuildFlattenedTypeList(SrcTy, SrcTypes);

  // Usually the size of SrcTypes must be greater than or equal to the size of
  // DestTypes.
  if (SrcTypes.size() < DestTypes.size())
    return false;

  unsigned SrcSize = SrcTypes.size();
  unsigned DstSize = DestTypes.size();
  unsigned I;
  for (I = 0; I < DstSize && I < SrcSize; I++) {
    if (SrcTypes[I]->isUnionType() || DestTypes[I]->isUnionType())
      return false;
    if (!CanPerformScalarCast(SrcTypes[I], DestTypes[I])) {
      return false;
    }
  }

  // check the rest of the source type for unions.
  for (; I < SrcSize; I++) {
    if (SrcTypes[I]->isUnionType())
      return false;
  }
  return true;
}

ExprResult SemaHLSL::ActOnOutParamExpr(ParmVarDecl *Param, Expr *Arg) {
  assert(Param->hasAttr<HLSLParamModifierAttr>() &&
         "We should not get here without a parameter modifier expression");
  const auto *Attr = Param->getAttr<HLSLParamModifierAttr>();
  if (Attr->getABI() == ParameterABI::Ordinary)
    return ExprResult(Arg);

  bool IsInOut = Attr->getABI() == ParameterABI::HLSLInOut;
  if (!Arg->isLValue()) {
    SemaRef.Diag(Arg->getBeginLoc(), diag::error_hlsl_inout_lvalue)
        << Arg << (IsInOut ? 1 : 0);
    return ExprError();
  }

  ASTContext &Ctx = SemaRef.getASTContext();

  QualType Ty = Param->getType().getNonLValueExprType(Ctx);

  // HLSL allows implicit conversions from scalars to vectors, but not the
  // inverse, so we need to disallow `inout` with scalar->vector or
  // scalar->matrix conversions.
  if (Arg->getType()->isScalarType() != Ty->isScalarType()) {
    SemaRef.Diag(Arg->getBeginLoc(), diag::error_hlsl_inout_scalar_extension)
        << Arg << (IsInOut ? 1 : 0);
    return ExprError();
  }

  auto *ArgOpV = new (Ctx) OpaqueValueExpr(Param->getBeginLoc(), Arg->getType(),
                                           VK_LValue, OK_Ordinary, Arg);

  // Parameters are initialized via copy initialization. This allows for
  // overload resolution of argument constructors.
  InitializedEntity Entity =
      InitializedEntity::InitializeParameter(Ctx, Ty, false);
  ExprResult Res =
      SemaRef.PerformCopyInitialization(Entity, Param->getBeginLoc(), ArgOpV);
  if (Res.isInvalid())
    return ExprError();
  Expr *Base = Res.get();
  // After the cast, drop the reference type when creating the exprs.
  Ty = Ty.getNonLValueExprType(Ctx);
  auto *OpV = new (Ctx)
      OpaqueValueExpr(Param->getBeginLoc(), Ty, VK_LValue, OK_Ordinary, Base);

  // Writebacks are performed with `=` binary operator, which allows for
  // overload resolution on writeback result expressions.
  Res = SemaRef.ActOnBinOp(SemaRef.getCurScope(), Param->getBeginLoc(),
                           tok::equal, ArgOpV, OpV);

  if (Res.isInvalid())
    return ExprError();
  Expr *Writeback = Res.get();
  auto *OutExpr =
      HLSLOutArgExpr::Create(Ctx, Ty, ArgOpV, OpV, Writeback, IsInOut);

  return ExprResult(OutExpr);
}

QualType SemaHLSL::getInoutParameterType(QualType Ty) {
  // If HLSL gains support for references, all the cites that use this will need
  // to be updated with semantic checking to produce errors for
  // pointers/references.
  assert(!Ty->isReferenceType() &&
         "Pointer and reference types cannot be inout or out parameters");
  Ty = SemaRef.getASTContext().getLValueReferenceType(Ty);
  Ty.addRestrict();
  return Ty;
}

static bool IsDefaultBufferConstantDecl(VarDecl *VD) {
  QualType QT = VD->getType();
  return VD->getDeclContext()->isTranslationUnit() &&
         QT.getAddressSpace() == LangAS::Default &&
         VD->getStorageClass() != SC_Static &&
         !VD->hasAttr<HLSLVkConstantIdAttr>() &&
         !isInvalidConstantBufferLeafElementType(QT.getTypePtr());
}

void SemaHLSL::deduceAddressSpace(VarDecl *Decl) {
  // The variable already has an address space (groupshared for ex).
  if (Decl->getType().hasAddressSpace())
    return;

  if (Decl->getType()->isDependentType())
    return;

  QualType Type = Decl->getType();

  if (Decl->hasAttr<HLSLVkExtBuiltinInputAttr>()) {
    LangAS ImplAS = LangAS::hlsl_input;
    Type = SemaRef.getASTContext().getAddrSpaceQualType(Type, ImplAS);
    Decl->setType(Type);
    return;
  }

  if (Type->isSamplerT() || Type->isVoidType())
    return;

  // Resource handles.
  if (Type->isHLSLResourceRecord() || Type->isHLSLResourceRecordArray())
    return;

  // Only static globals belong to the Private address space.
  // Non-static globals belongs to the cbuffer.
  if (Decl->getStorageClass() != SC_Static && !Decl->isStaticDataMember())
    return;

  LangAS ImplAS = LangAS::hlsl_private;
  Type = SemaRef.getASTContext().getAddrSpaceQualType(Type, ImplAS);
  Decl->setType(Type);
}

void SemaHLSL::ActOnVariableDeclarator(VarDecl *VD) {
  if (VD->hasGlobalStorage()) {
    // make sure the declaration has a complete type
    if (SemaRef.RequireCompleteType(
            VD->getLocation(),
            SemaRef.getASTContext().getBaseElementType(VD->getType()),
            diag::err_typecheck_decl_incomplete_type)) {
      VD->setInvalidDecl();
      deduceAddressSpace(VD);
      return;
    }

    // Global variables outside a cbuffer block that are not a resource, static,
    // groupshared, or an empty array or struct belong to the default constant
    // buffer $Globals (to be created at the end of the translation unit).
    if (IsDefaultBufferConstantDecl(VD)) {
      // update address space to hlsl_constant
      QualType NewTy = getASTContext().getAddrSpaceQualType(
          VD->getType(), LangAS::hlsl_constant);
      VD->setType(NewTy);
      DefaultCBufferDecls.push_back(VD);
    }

    // find all resources bindings on decl
    if (VD->getType()->isHLSLIntangibleType())
      collectResourceBindingsOnVarDecl(VD);

    if (isResourceRecordTypeOrArrayOf(VD) ||
        VD->hasAttr<HLSLVkConstantIdAttr>()) {
      // Make the variable for resources static. The global externally visible
      // storage is accessed through the handle, which is a member. The variable
      // itself is not externally visible.
      VD->setStorageClass(StorageClass::SC_Static);
    }

    // process explicit bindings
    processExplicitBindingsOnDecl(VD);

    if (VD->getType()->isHLSLResourceRecordArray()) {
      // If the resource array does not have an explicit binding attribute,
      // create an implicit one. It will be used to transfer implicit binding
      // order_ID to codegen.
      if (!VD->hasAttr<HLSLVkBindingAttr>()) {
        HLSLResourceBindingAttr *RBA = VD->getAttr<HLSLResourceBindingAttr>();
        if (!RBA || !RBA->hasRegisterSlot()) {
          uint32_t OrderID = getNextImplicitBindingOrderID();
          if (RBA)
            RBA->setImplicitBindingOrderID(OrderID);
          else
            addImplicitBindingAttrToDecl(
                SemaRef, VD, getRegisterType(getResourceArrayHandleType(VD)),
                OrderID);
        }
      }
    }
  }

  deduceAddressSpace(VD);
}

static bool initVarDeclWithCtor(Sema &S, VarDecl *VD,
                                MutableArrayRef<Expr *> Args) {
  InitializedEntity Entity = InitializedEntity::InitializeVariable(VD);
  InitializationKind Kind = InitializationKind::CreateDirect(
      VD->getLocation(), SourceLocation(), SourceLocation());

  InitializationSequence InitSeq(S, Entity, Kind, Args);
  if (InitSeq.Failed())
    return false;

  ExprResult Init = InitSeq.Perform(S, Entity, Kind, Args);
  if (!Init.get())
    return false;

  VD->setInit(S.MaybeCreateExprWithCleanups(Init.get()));
  VD->setInitStyle(VarDecl::CallInit);
  S.CheckCompleteVariableDeclaration(VD);
  return true;
}

void SemaHLSL::createResourceRecordCtorArgs(
    const Type *ResourceTy, StringRef VarName, HLSLResourceBindingAttr *RBA,
    HLSLVkBindingAttr *VkBinding, uint32_t ArrayIndex,
    llvm::SmallVectorImpl<Expr *> &Args) {
  std::optional<uint32_t> RegisterSlot;
  uint32_t SpaceNo = 0;
  if (VkBinding) {
    RegisterSlot = VkBinding->getBinding();
    SpaceNo = VkBinding->getSet();
  } else if (RBA) {
    if (RBA->hasRegisterSlot())
      RegisterSlot = RBA->getSlotNumber();
    SpaceNo = RBA->getSpaceNumber();
  }

  ASTContext &AST = SemaRef.getASTContext();
  uint64_t UIntTySize = AST.getTypeSize(AST.UnsignedIntTy);
  uint64_t IntTySize = AST.getTypeSize(AST.IntTy);
  IntegerLiteral *RangeSize = IntegerLiteral::Create(
      AST, llvm::APInt(IntTySize, 1), AST.IntTy, SourceLocation());
  IntegerLiteral *Index =
      IntegerLiteral::Create(AST, llvm::APInt(UIntTySize, ArrayIndex),
                             AST.UnsignedIntTy, SourceLocation());
  IntegerLiteral *Space =
      IntegerLiteral::Create(AST, llvm::APInt(UIntTySize, SpaceNo),
                             AST.UnsignedIntTy, SourceLocation());
  StringLiteral *Name = StringLiteral::Create(
      AST, VarName, StringLiteralKind::Ordinary, false,
      AST.getStringLiteralArrayType(AST.CharTy.withConst(), VarName.size()),
      SourceLocation());

  // resource with explicit binding
  if (RegisterSlot.has_value()) {
    IntegerLiteral *RegSlot = IntegerLiteral::Create(
        AST, llvm::APInt(UIntTySize, RegisterSlot.value()), AST.UnsignedIntTy,
        SourceLocation());
    Args.append({RegSlot, Space, RangeSize, Index, Name});
  } else {
    // resource with implicit binding
    uint32_t OrderID = (RBA && RBA->hasImplicitBindingOrderID())
                           ? RBA->getImplicitBindingOrderID()
                           : getNextImplicitBindingOrderID();
    IntegerLiteral *OrderId =
        IntegerLiteral::Create(AST, llvm::APInt(UIntTySize, OrderID),
                               AST.UnsignedIntTy, SourceLocation());
    Args.append({Space, RangeSize, Index, OrderId, Name});
  }
}

bool SemaHLSL::initGlobalResourceDecl(VarDecl *VD) {
  SmallVector<Expr *> Args;
  createResourceRecordCtorArgs(VD->getType().getTypePtr(), VD->getName(),
                               VD->getAttr<HLSLResourceBindingAttr>(),
                               VD->getAttr<HLSLVkBindingAttr>(), 0, Args);
  return initVarDeclWithCtor(SemaRef, VD, Args);
}

bool SemaHLSL::initGlobalResourceArrayDecl(VarDecl *VD) {
  assert(VD->getType()->isHLSLResourceRecordArray() &&
         "expected array of resource records");

  // Individual resources in a resource array are not initialized here. They
  // are initialized later on during codegen when the individual resources are
  // accessed. Codegen will emit a call to the resource constructor with the
  // specified array index. We need to make sure though that the constructor
  // for the specific resource type is instantiated, so codegen can emit a call
  // to it when the array element is accessed.
  SmallVector<Expr *> Args;
  QualType ResElementTy = VD->getASTContext().getBaseElementType(VD->getType());
  createResourceRecordCtorArgs(ResElementTy.getTypePtr(), VD->getName(),
                               VD->getAttr<HLSLResourceBindingAttr>(),
                               VD->getAttr<HLSLVkBindingAttr>(), 0, Args);

  SourceLocation Loc = VD->getLocation();
  InitializedEntity Entity =
      InitializedEntity::InitializeTemporary(ResElementTy);
  InitializationKind Kind = InitializationKind::CreateDirect(Loc, Loc, Loc);
  InitializationSequence InitSeq(SemaRef, Entity, Kind, Args);
  if (InitSeq.Failed())
    return false;

  // This takes care of instantiating and emitting of the constructor that will
  // be called from codegen when the array is accessed.
  ExprResult OneResInit = InitSeq.Perform(SemaRef, Entity, Kind, Args);
  return !OneResInit.isInvalid();
}

// Returns true if the initialization has been handled.
// Returns false to use default initialization.
bool SemaHLSL::ActOnUninitializedVarDecl(VarDecl *VD) {
  // Objects in the hlsl_constant address space are initialized
  // externally, so don't synthesize an implicit initializer.
  if (VD->getType().getAddressSpace() == LangAS::hlsl_constant)
    return true;

  // Initialize resources at the global scope
  if (VD->hasGlobalStorage()) {
    const Type *Ty = VD->getType().getTypePtr();
    if (Ty->isHLSLResourceRecord())
      return initGlobalResourceDecl(VD);
    if (Ty->isHLSLResourceRecordArray())
      return initGlobalResourceArrayDecl(VD);
  }
  return false;
}

// Walks though the global variable declaration, collects all resource binding
// requirements and adds them to Bindings
void SemaHLSL::collectResourceBindingsOnVarDecl(VarDecl *VD) {
  assert(VD->hasGlobalStorage() && VD->getType()->isHLSLIntangibleType() &&
         "expected global variable that contains HLSL resource");

  // Cbuffers and Tbuffers are HLSLBufferDecl types
  if (const HLSLBufferDecl *CBufferOrTBuffer = dyn_cast<HLSLBufferDecl>(VD)) {
    Bindings.addDeclBindingInfo(VD, CBufferOrTBuffer->isCBuffer()
                                        ? ResourceClass::CBuffer
                                        : ResourceClass::SRV);
    return;
  }

  // Unwrap arrays
  // FIXME: Calculate array size while unwrapping
  const Type *Ty = VD->getType()->getUnqualifiedDesugaredType();
  while (Ty->isConstantArrayType()) {
    const ConstantArrayType *CAT = cast<ConstantArrayType>(Ty);
    Ty = CAT->getElementType()->getUnqualifiedDesugaredType();
  }

  // Resource (or array of resources)
  if (const HLSLAttributedResourceType *AttrResType =
          HLSLAttributedResourceType::findHandleTypeOnResource(Ty)) {
    Bindings.addDeclBindingInfo(VD, AttrResType->getAttrs().ResourceClass);
    return;
  }

  // User defined record type
  if (const RecordType *RT = dyn_cast<RecordType>(Ty))
    collectResourceBindingsOnUserRecordDecl(VD, RT);
}

// Walks though the explicit resource binding attributes on the declaration,
// and makes sure there is a resource that matched the binding and updates
// DeclBindingInfoLists
void SemaHLSL::processExplicitBindingsOnDecl(VarDecl *VD) {
  assert(VD->hasGlobalStorage() && "expected global variable");

  bool HasBinding = false;
  for (Attr *A : VD->attrs()) {
    if (isa<HLSLVkBindingAttr>(A))
      HasBinding = true;

    HLSLResourceBindingAttr *RBA = dyn_cast<HLSLResourceBindingAttr>(A);
    if (!RBA || !RBA->hasRegisterSlot())
      continue;
    HasBinding = true;

    RegisterType RT = RBA->getRegisterType();
    assert(RT != RegisterType::I && "invalid or obsolete register type should "
                                    "never have an attribute created");

    if (RT == RegisterType::C) {
      if (Bindings.hasBindingInfoForDecl(VD))
        SemaRef.Diag(VD->getLocation(),
                     diag::warn_hlsl_user_defined_type_missing_member)
            << static_cast<int>(RT);
      continue;
    }

    // Find DeclBindingInfo for this binding and update it, or report error
    // if it does not exist (user type does to contain resources with the
    // expected resource class).
    ResourceClass RC = getResourceClass(RT);
    if (DeclBindingInfo *BI = Bindings.getDeclBindingInfo(VD, RC)) {
      // update binding info
      BI->setBindingAttribute(RBA, BindingType::Explicit);
    } else {
      SemaRef.Diag(VD->getLocation(),
                   diag::warn_hlsl_user_defined_type_missing_member)
          << static_cast<int>(RT);
    }
  }

  if (!HasBinding && isResourceRecordTypeOrArrayOf(VD))
    SemaRef.Diag(VD->getLocation(), diag::warn_hlsl_implicit_binding);
}
namespace {
class InitListTransformer {
  Sema &S;
  ASTContext &Ctx;
  QualType InitTy;
  QualType *DstIt = nullptr;
  Expr **ArgIt = nullptr;
  // Is wrapping the destination type iterator required? This is only used for
  // incomplete array types where we loop over the destination type since we
  // don't know the full number of elements from the declaration.
  bool Wrap;

  bool castInitializer(Expr *E) {
    assert(DstIt && "This should always be something!");
    if (DstIt == DestTypes.end()) {
      if (!Wrap) {
        ArgExprs.push_back(E);
        // This is odd, but it isn't technically a failure due to conversion, we
        // handle mismatched counts of arguments differently.
        return true;
      }
      DstIt = DestTypes.begin();
    }
    InitializedEntity Entity = InitializedEntity::InitializeParameter(
        Ctx, *DstIt, /* Consumed (ObjC) */ false);
    ExprResult Res = S.PerformCopyInitialization(Entity, E->getBeginLoc(), E);
    if (Res.isInvalid())
      return false;
    Expr *Init = Res.get();
    ArgExprs.push_back(Init);
    DstIt++;
    return true;
  }

  bool buildInitializerListImpl(Expr *E) {
    // If this is an initialization list, traverse the sub initializers.
    if (auto *Init = dyn_cast<InitListExpr>(E)) {
      for (auto *SubInit : Init->inits())
        if (!buildInitializerListImpl(SubInit))
          return false;
      return true;
    }

    // If this is a scalar type, just enqueue the expression.
    QualType Ty = E->getType();

    if (Ty->isScalarType() || (Ty->isRecordType() && !Ty->isAggregateType()))
      return castInitializer(E);

    if (auto *VecTy = Ty->getAs<VectorType>()) {
      uint64_t Size = VecTy->getNumElements();

      QualType SizeTy = Ctx.getSizeType();
      uint64_t SizeTySize = Ctx.getTypeSize(SizeTy);
      for (uint64_t I = 0; I < Size; ++I) {
        auto *Idx = IntegerLiteral::Create(Ctx, llvm::APInt(SizeTySize, I),
                                           SizeTy, SourceLocation());

        ExprResult ElExpr = S.CreateBuiltinArraySubscriptExpr(
            E, E->getBeginLoc(), Idx, E->getEndLoc());
        if (ElExpr.isInvalid())
          return false;
        if (!castInitializer(ElExpr.get()))
          return false;
      }
      return true;
    }

    if (auto *ArrTy = dyn_cast<ConstantArrayType>(Ty.getTypePtr())) {
      uint64_t Size = ArrTy->getZExtSize();
      QualType SizeTy = Ctx.getSizeType();
      uint64_t SizeTySize = Ctx.getTypeSize(SizeTy);
      for (uint64_t I = 0; I < Size; ++I) {
        auto *Idx = IntegerLiteral::Create(Ctx, llvm::APInt(SizeTySize, I),
                                           SizeTy, SourceLocation());
        ExprResult ElExpr = S.CreateBuiltinArraySubscriptExpr(
            E, E->getBeginLoc(), Idx, E->getEndLoc());
        if (ElExpr.isInvalid())
          return false;
        if (!buildInitializerListImpl(ElExpr.get()))
          return false;
      }
      return true;
    }

    if (auto *RTy = Ty->getAs<RecordType>()) {
      llvm::SmallVector<const RecordType *> RecordTypes;
      RecordTypes.push_back(RTy);
      while (RecordTypes.back()->getAsCXXRecordDecl()->getNumBases()) {
        CXXRecordDecl *D = RecordTypes.back()->getAsCXXRecordDecl();
        assert(D->getNumBases() == 1 &&
               "HLSL doesn't support multiple inheritance");
        RecordTypes.push_back(D->bases_begin()->getType()->getAs<RecordType>());
      }
      while (!RecordTypes.empty()) {
        const RecordType *RT = RecordTypes.pop_back_val();
        for (auto *FD :
             RT->getOriginalDecl()->getDefinitionOrSelf()->fields()) {
          DeclAccessPair Found = DeclAccessPair::make(FD, FD->getAccess());
          DeclarationNameInfo NameInfo(FD->getDeclName(), E->getBeginLoc());
          ExprResult Res = S.BuildFieldReferenceExpr(
              E, false, E->getBeginLoc(), CXXScopeSpec(), FD, Found, NameInfo);
          if (Res.isInvalid())
            return false;
          if (!buildInitializerListImpl(Res.get()))
            return false;
        }
      }
    }
    return true;
  }

  Expr *generateInitListsImpl(QualType Ty) {
    assert(ArgIt != ArgExprs.end() && "Something is off in iteration!");
    if (Ty->isScalarType() || (Ty->isRecordType() && !Ty->isAggregateType()))
      return *(ArgIt++);

    llvm::SmallVector<Expr *> Inits;
    assert(!isa<MatrixType>(Ty) && "Matrix types not yet supported in HLSL");
    Ty = Ty.getDesugaredType(Ctx);
    if (Ty->isVectorType() || Ty->isConstantArrayType()) {
      QualType ElTy;
      uint64_t Size = 0;
      if (auto *ATy = Ty->getAs<VectorType>()) {
        ElTy = ATy->getElementType();
        Size = ATy->getNumElements();
      } else {
        auto *VTy = cast<ConstantArrayType>(Ty.getTypePtr());
        ElTy = VTy->getElementType();
        Size = VTy->getZExtSize();
      }
      for (uint64_t I = 0; I < Size; ++I)
        Inits.push_back(generateInitListsImpl(ElTy));
    }
    if (auto *RTy = Ty->getAs<RecordType>()) {
      llvm::SmallVector<const RecordType *> RecordTypes;
      RecordTypes.push_back(RTy);
      while (RecordTypes.back()->getAsCXXRecordDecl()->getNumBases()) {
        CXXRecordDecl *D = RecordTypes.back()->getAsCXXRecordDecl();
        assert(D->getNumBases() == 1 &&
               "HLSL doesn't support multiple inheritance");
        RecordTypes.push_back(D->bases_begin()->getType()->getAs<RecordType>());
      }
      while (!RecordTypes.empty()) {
        const RecordType *RT = RecordTypes.pop_back_val();
        for (auto *FD :
             RT->getOriginalDecl()->getDefinitionOrSelf()->fields()) {
          Inits.push_back(generateInitListsImpl(FD->getType()));
        }
      }
    }
    auto *NewInit = new (Ctx) InitListExpr(Ctx, Inits.front()->getBeginLoc(),
                                           Inits, Inits.back()->getEndLoc());
    NewInit->setType(Ty);
    return NewInit;
  }

public:
  llvm::SmallVector<QualType, 16> DestTypes;
  llvm::SmallVector<Expr *, 16> ArgExprs;
  InitListTransformer(Sema &SemaRef, const InitializedEntity &Entity)
      : S(SemaRef), Ctx(SemaRef.getASTContext()),
        Wrap(Entity.getType()->isIncompleteArrayType()) {
    InitTy = Entity.getType().getNonReferenceType();
    // When we're generating initializer lists for incomplete array types we
    // need to wrap around both when building the initializers and when
    // generating the final initializer lists.
    if (Wrap) {
      assert(InitTy->isIncompleteArrayType());
      const IncompleteArrayType *IAT = Ctx.getAsIncompleteArrayType(InitTy);
      InitTy = IAT->getElementType();
    }
    BuildFlattenedTypeList(InitTy, DestTypes);
    DstIt = DestTypes.begin();
  }

  bool buildInitializerList(Expr *E) { return buildInitializerListImpl(E); }

  Expr *generateInitLists() {
    assert(!ArgExprs.empty() &&
           "Call buildInitializerList to generate argument expressions.");
    ArgIt = ArgExprs.begin();
    if (!Wrap)
      return generateInitListsImpl(InitTy);
    llvm::SmallVector<Expr *> Inits;
    while (ArgIt != ArgExprs.end())
      Inits.push_back(generateInitListsImpl(InitTy));

    auto *NewInit = new (Ctx) InitListExpr(Ctx, Inits.front()->getBeginLoc(),
                                           Inits, Inits.back()->getEndLoc());
    llvm::APInt ArySize(64, Inits.size());
    NewInit->setType(Ctx.getConstantArrayType(InitTy, ArySize, nullptr,
                                              ArraySizeModifier::Normal, 0));
    return NewInit;
  }
};
} // namespace

bool SemaHLSL::transformInitList(const InitializedEntity &Entity,
                                 InitListExpr *Init) {
  // If the initializer is a scalar, just return it.
  if (Init->getType()->isScalarType())
    return true;
  ASTContext &Ctx = SemaRef.getASTContext();
  InitListTransformer ILT(SemaRef, Entity);

  for (unsigned I = 0; I < Init->getNumInits(); ++I) {
    Expr *E = Init->getInit(I);
    if (E->HasSideEffects(Ctx)) {
      QualType Ty = E->getType();
      if (Ty->isRecordType())
        E = new (Ctx) MaterializeTemporaryExpr(Ty, E, E->isLValue());
      E = new (Ctx) OpaqueValueExpr(E->getBeginLoc(), Ty, E->getValueKind(),
                                    E->getObjectKind(), E);
      Init->setInit(I, E);
    }
    if (!ILT.buildInitializerList(E))
      return false;
  }
  size_t ExpectedSize = ILT.DestTypes.size();
  size_t ActualSize = ILT.ArgExprs.size();
  // For incomplete arrays it is completely arbitrary to choose whether we think
  // the user intended fewer or more elements. This implementation assumes that
  // the user intended more, and errors that there are too few initializers to
  // complete the final element.
  if (Entity.getType()->isIncompleteArrayType())
    ExpectedSize =
        ((ActualSize + ExpectedSize - 1) / ExpectedSize) * ExpectedSize;

  // An initializer list might be attempting to initialize a reference or
  // rvalue-reference. When checking the initializer we should look through
  // the reference.
  QualType InitTy = Entity.getType().getNonReferenceType();
  if (InitTy.hasAddressSpace())
    InitTy = SemaRef.getASTContext().removeAddrSpaceQualType(InitTy);
  if (ExpectedSize != ActualSize) {
    int TooManyOrFew = ActualSize > ExpectedSize ? 1 : 0;
    SemaRef.Diag(Init->getBeginLoc(), diag::err_hlsl_incorrect_num_initializers)
        << TooManyOrFew << InitTy << ExpectedSize << ActualSize;
    return false;
  }

  // generateInitListsImpl will always return an InitListExpr here, because the
  // scalar case is handled above.
  auto *NewInit = cast<InitListExpr>(ILT.generateInitLists());
  Init->resizeInits(Ctx, NewInit->getNumInits());
  for (unsigned I = 0; I < NewInit->getNumInits(); ++I)
    Init->updateInit(Ctx, I, NewInit->getInit(I));
  return true;
}

bool SemaHLSL::handleInitialization(VarDecl *VDecl, Expr *&Init) {
  const HLSLVkConstantIdAttr *ConstIdAttr =
      VDecl->getAttr<HLSLVkConstantIdAttr>();
  if (!ConstIdAttr)
    return true;

  ASTContext &Context = SemaRef.getASTContext();

  APValue InitValue;
  if (!Init->isCXX11ConstantExpr(Context, &InitValue)) {
    Diag(VDecl->getLocation(), diag::err_specialization_const);
    VDecl->setInvalidDecl();
    return false;
  }

  Builtin::ID BID =
      getSpecConstBuiltinId(VDecl->getType()->getUnqualifiedDesugaredType());

  // Argument 1: The ID from the attribute
  int ConstantID = ConstIdAttr->getId();
  llvm::APInt IDVal(Context.getIntWidth(Context.IntTy), ConstantID);
  Expr *IdExpr = IntegerLiteral::Create(Context, IDVal, Context.IntTy,
                                        ConstIdAttr->getLocation());

  SmallVector<Expr *, 2> Args = {IdExpr, Init};
  Expr *C = SemaRef.BuildBuiltinCallExpr(Init->getExprLoc(), BID, Args);
  if (C->getType()->getCanonicalTypeUnqualified() !=
      VDecl->getType()->getCanonicalTypeUnqualified()) {
    C = SemaRef
            .BuildCStyleCastExpr(SourceLocation(),
                                 Context.getTrivialTypeSourceInfo(
                                     Init->getType(), Init->getExprLoc()),
                                 SourceLocation(), C)
            .get();
  }
  Init = C;
  return true;
}
