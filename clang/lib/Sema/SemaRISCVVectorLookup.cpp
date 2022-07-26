//==- SemaRISCVVectorLookup.cpp - Name Lookup for RISC-V Vector Intrinsic -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements name lookup for RISC-V vector intrinsic.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/RISCVIntrinsicManager.h"
#include "clang/Sema/Sema.h"
#include "clang/Support/RISCVVIntrinsicUtils.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace clang;
using namespace clang::RISCV;

namespace {

// Function definition of a RVV intrinsic.
struct RVVIntrinsicDef {
  /// Full function name with suffix, e.g. vadd_vv_i32m1.
  std::string Name;

  /// Overloaded function name, e.g. vadd.
  std::string OverloadName;

  /// Mapping to which clang built-in function, e.g. __builtin_rvv_vadd.
  std::string BuiltinName;

  /// Function signature, first element is return type.
  RVVTypes Signature;
};

struct RVVOverloadIntrinsicDef {
  // Indexes of RISCVIntrinsicManagerImpl::IntrinsicList.
  SmallVector<size_t, 8> Indexes;
};

} // namespace

static const PrototypeDescriptor RVVSignatureTable[] = {
#define DECL_SIGNATURE_TABLE
#include "clang/Basic/riscv_vector_builtin_sema.inc"
#undef DECL_SIGNATURE_TABLE
};

static const RVVIntrinsicRecord RVVIntrinsicRecords[] = {
#define DECL_INTRINSIC_RECORDS
#include "clang/Basic/riscv_vector_builtin_sema.inc"
#undef DECL_INTRINSIC_RECORDS
};

// Get subsequence of signature table.
static ArrayRef<PrototypeDescriptor> ProtoSeq2ArrayRef(uint16_t Index,
                                                       uint8_t Length) {
  return makeArrayRef(&RVVSignatureTable[Index], Length);
}

static QualType RVVType2Qual(ASTContext &Context, const RVVType *Type) {
  QualType QT;
  switch (Type->getScalarType()) {
  case ScalarTypeKind::Void:
    QT = Context.VoidTy;
    break;
  case ScalarTypeKind::Size_t:
    QT = Context.getSizeType();
    break;
  case ScalarTypeKind::Ptrdiff_t:
    QT = Context.getPointerDiffType();
    break;
  case ScalarTypeKind::UnsignedLong:
    QT = Context.UnsignedLongTy;
    break;
  case ScalarTypeKind::SignedLong:
    QT = Context.LongTy;
    break;
  case ScalarTypeKind::Boolean:
    QT = Context.BoolTy;
    break;
  case ScalarTypeKind::SignedInteger:
    QT = Context.getIntTypeForBitwidth(Type->getElementBitwidth(), true);
    break;
  case ScalarTypeKind::UnsignedInteger:
    QT = Context.getIntTypeForBitwidth(Type->getElementBitwidth(), false);
    break;
  case ScalarTypeKind::Float:
    switch (Type->getElementBitwidth()) {
    case 64:
      QT = Context.DoubleTy;
      break;
    case 32:
      QT = Context.FloatTy;
      break;
    case 16:
      QT = Context.Float16Ty;
      break;
    default:
      llvm_unreachable("Unsupported floating point width.");
    }
    break;
  case Invalid:
    llvm_unreachable("Unhandled type.");
  }
  if (Type->isVector())
    QT = Context.getScalableVectorType(QT, Type->getScale().getValue());

  if (Type->isConstant())
    QT = Context.getConstType(QT);

  // Transform the type to a pointer as the last step, if necessary.
  if (Type->isPointer())
    QT = Context.getPointerType(QT);

  return QT;
}

namespace {
class RISCVIntrinsicManagerImpl : public sema::RISCVIntrinsicManager {
private:
  Sema &S;
  ASTContext &Context;

  // List of all RVV intrinsic.
  std::vector<RVVIntrinsicDef> IntrinsicList;
  // Mapping function name to index of IntrinsicList.
  StringMap<size_t> Intrinsics;
  // Mapping function name to RVVOverloadIntrinsicDef.
  StringMap<RVVOverloadIntrinsicDef> OverloadIntrinsics;

  // Create IntrinsicList
  void InitIntrinsicList();

  // Create RVVIntrinsicDef.
  void InitRVVIntrinsic(const RVVIntrinsicRecord &Record, StringRef SuffixStr,
                        StringRef OverloadedSuffixStr, bool IsMask,
                        RVVTypes &Types);

  // Create FunctionDecl for a vector intrinsic.
  void CreateRVVIntrinsicDecl(LookupResult &LR, IdentifierInfo *II,
                              Preprocessor &PP, unsigned Index,
                              bool IsOverload);

public:
  RISCVIntrinsicManagerImpl(clang::Sema &S) : S(S), Context(S.Context) {
    InitIntrinsicList();
  }

  // Create RISC-V vector intrinsic and insert into symbol table if found, and
  // return true, otherwise return false.
  bool CreateIntrinsicIfFound(LookupResult &LR, IdentifierInfo *II,
                              Preprocessor &PP) override;
};
} // namespace

void RISCVIntrinsicManagerImpl::InitIntrinsicList() {
  const TargetInfo &TI = Context.getTargetInfo();
  bool HasVectorFloat32 = TI.hasFeature("zve32f");
  bool HasVectorFloat64 = TI.hasFeature("zve64d");
  bool HasZvfh = TI.hasFeature("experimental-zvfh");
  bool HasRV64 = TI.hasFeature("64bit");
  bool HasFullMultiply = TI.hasFeature("v");

  // Construction of RVVIntrinsicRecords need to sync with createRVVIntrinsics
  // in RISCVVEmitter.cpp.
  for (auto &Record : RVVIntrinsicRecords) {
    // Create Intrinsics for each type and LMUL.
    BasicType BaseType = BasicType::Unknown;
    ArrayRef<PrototypeDescriptor> BasicProtoSeq =
        ProtoSeq2ArrayRef(Record.PrototypeIndex, Record.PrototypeLength);
    ArrayRef<PrototypeDescriptor> SuffixProto =
        ProtoSeq2ArrayRef(Record.SuffixIndex, Record.SuffixLength);
    ArrayRef<PrototypeDescriptor> OverloadedSuffixProto = ProtoSeq2ArrayRef(
        Record.OverloadedSuffixIndex, Record.OverloadedSuffixSize);

    llvm::SmallVector<PrototypeDescriptor> ProtoSeq =
        RVVIntrinsic::computeBuiltinTypes(BasicProtoSeq, /*IsMasked=*/false,
                                          /*HasMaskedOffOperand=*/false,
                                          Record.HasVL, Record.NF);

    llvm::SmallVector<PrototypeDescriptor> ProtoMaskSeq =
        RVVIntrinsic::computeBuiltinTypes(BasicProtoSeq, /*IsMasked=*/true,
                                          Record.HasMaskedOffOperand,
                                          Record.HasVL, Record.NF);

    for (unsigned int TypeRangeMaskShift = 0;
         TypeRangeMaskShift <= static_cast<unsigned int>(BasicType::MaxOffset);
         ++TypeRangeMaskShift) {
      unsigned int BaseTypeI = 1 << TypeRangeMaskShift;
      BaseType = static_cast<BasicType>(BaseTypeI);

      if ((BaseTypeI & Record.TypeRangeMask) != BaseTypeI)
        continue;

      // Check requirement.
      if (BaseType == BasicType::Float16 && !HasZvfh)
        continue;

      if (BaseType == BasicType::Float32 && !HasVectorFloat32)
        continue;

      if (BaseType == BasicType::Float64 && !HasVectorFloat64)
        continue;

      if (((Record.RequiredExtensions & RVV_REQ_RV64) == RVV_REQ_RV64) &&
          !HasRV64)
        continue;

      if ((BaseType == BasicType::Int64) &&
          ((Record.RequiredExtensions & RVV_REQ_FullMultiply) ==
           RVV_REQ_FullMultiply) &&
          !HasFullMultiply)
        continue;

      // Expanded with different LMUL.
      for (int Log2LMUL = -3; Log2LMUL <= 3; Log2LMUL++) {
        if (!(Record.Log2LMULMask & (1 << (Log2LMUL + 3))))
          continue;

        Optional<RVVTypes> Types =
            RVVType::computeTypes(BaseType, Log2LMUL, Record.NF, ProtoSeq);

        // Ignored to create new intrinsic if there are any illegal types.
        if (!Types.hasValue())
          continue;

        std::string SuffixStr =
            RVVIntrinsic::getSuffixStr(BaseType, Log2LMUL, SuffixProto);
        std::string OverloadedSuffixStr = RVVIntrinsic::getSuffixStr(
            BaseType, Log2LMUL, OverloadedSuffixProto);

        // Create non-masked intrinsic.
        InitRVVIntrinsic(Record, SuffixStr, OverloadedSuffixStr, false, *Types);

        if (Record.HasMasked) {
          // Create masked intrinsic.
          Optional<RVVTypes> MaskTypes = RVVType::computeTypes(
              BaseType, Log2LMUL, Record.NF, ProtoMaskSeq);

          InitRVVIntrinsic(Record, SuffixStr, OverloadedSuffixStr, true,
                           *MaskTypes);
        }
      }
    }
  }
}

// Compute name and signatures for intrinsic with practical types.
void RISCVIntrinsicManagerImpl::InitRVVIntrinsic(
    const RVVIntrinsicRecord &Record, StringRef SuffixStr,
    StringRef OverloadedSuffixStr, bool IsMask, RVVTypes &Signature) {
  // Function name, e.g. vadd_vv_i32m1.
  std::string Name = Record.Name;
  if (!SuffixStr.empty())
    Name += "_" + SuffixStr.str();

  if (IsMask)
    Name += "_m";

  // Overloaded function name, e.g. vadd.
  std::string OverloadedName;
  if (!Record.OverloadedName)
    OverloadedName = StringRef(Record.Name).split("_").first.str();
  else
    OverloadedName = Record.OverloadedName;
  if (!OverloadedSuffixStr.empty())
    OverloadedName += "_" + OverloadedSuffixStr.str();

  // clang built-in function name, e.g. __builtin_rvv_vadd.
  std::string BuiltinName = "__builtin_rvv_" + std::string(Record.Name);
  if (IsMask)
    BuiltinName += "_m";

  // Put into IntrinsicList.
  size_t Index = IntrinsicList.size();
  IntrinsicList.push_back({Name, OverloadedName, BuiltinName, Signature});

  // Creating mapping to Intrinsics.
  Intrinsics.insert({Name, Index});

  // Get the RVVOverloadIntrinsicDef.
  RVVOverloadIntrinsicDef &OverloadIntrinsicDef =
      OverloadIntrinsics[OverloadedName];

  // And added the index.
  OverloadIntrinsicDef.Indexes.push_back(Index);
}

void RISCVIntrinsicManagerImpl::CreateRVVIntrinsicDecl(LookupResult &LR,
                                                       IdentifierInfo *II,
                                                       Preprocessor &PP,
                                                       unsigned Index,
                                                       bool IsOverload) {
  ASTContext &Context = S.Context;
  RVVIntrinsicDef &IDef = IntrinsicList[Index];
  RVVTypes Sigs = IDef.Signature;
  size_t SigLength = Sigs.size();
  RVVType *ReturnType = Sigs[0];
  QualType RetType = RVVType2Qual(Context, ReturnType);
  SmallVector<QualType, 8> ArgTypes;
  QualType BuiltinFuncType;

  // Skip return type, and convert RVVType to QualType for arguments.
  for (size_t i = 1; i < SigLength; ++i)
    ArgTypes.push_back(RVVType2Qual(Context, Sigs[i]));

  FunctionProtoType::ExtProtoInfo PI(
      Context.getDefaultCallingConvention(false, false, true));

  PI.Variadic = false;

  SourceLocation Loc = LR.getNameLoc();
  BuiltinFuncType = Context.getFunctionType(RetType, ArgTypes, PI);
  DeclContext *Parent = Context.getTranslationUnitDecl();

  FunctionDecl *RVVIntrinsicDecl = FunctionDecl::Create(
      Context, Parent, Loc, Loc, II, BuiltinFuncType, /*TInfo=*/nullptr,
      SC_Extern, S.getCurFPFeatures().isFPConstrained(),
      /*isInlineSpecified*/ false,
      /*hasWrittenPrototype*/ true);

  // Create Decl objects for each parameter, adding them to the
  // FunctionDecl.
  const auto *FP = cast<FunctionProtoType>(BuiltinFuncType);
  SmallVector<ParmVarDecl *, 8> ParmList;
  for (unsigned IParm = 0, E = FP->getNumParams(); IParm != E; ++IParm) {
    ParmVarDecl *Parm =
        ParmVarDecl::Create(Context, RVVIntrinsicDecl, Loc, Loc, nullptr,
                            FP->getParamType(IParm), nullptr, SC_None, nullptr);
    Parm->setScopeInfo(0, IParm);
    ParmList.push_back(Parm);
  }
  RVVIntrinsicDecl->setParams(ParmList);

  // Add function attributes.
  if (IsOverload)
    RVVIntrinsicDecl->addAttr(OverloadableAttr::CreateImplicit(Context));

  // Setup alias to __builtin_rvv_*
  IdentifierInfo &IntrinsicII = PP.getIdentifierTable().get(IDef.BuiltinName);
  RVVIntrinsicDecl->addAttr(
      BuiltinAliasAttr::CreateImplicit(S.Context, &IntrinsicII));

  // Add to symbol table.
  LR.addDecl(RVVIntrinsicDecl);
}

bool RISCVIntrinsicManagerImpl::CreateIntrinsicIfFound(LookupResult &LR,
                                                       IdentifierInfo *II,
                                                       Preprocessor &PP) {
  StringRef Name = II->getName();

  // Lookup the function name from the overload intrinsics first.
  auto OvIItr = OverloadIntrinsics.find(Name);
  if (OvIItr != OverloadIntrinsics.end()) {
    const RVVOverloadIntrinsicDef &OvIntrinsicDef = OvIItr->second;
    for (auto Index : OvIntrinsicDef.Indexes)
      CreateRVVIntrinsicDecl(LR, II, PP, Index,
                             /*IsOverload*/ true);

    // If we added overloads, need to resolve the lookup result.
    LR.resolveKind();
    return true;
  }

  // Lookup the function name from the intrinsics.
  auto Itr = Intrinsics.find(Name);
  if (Itr != Intrinsics.end()) {
    CreateRVVIntrinsicDecl(LR, II, PP, Itr->second,
                           /*IsOverload*/ false);
    return true;
  }

  // It's not an RVV intrinsics.
  return false;
}

namespace clang {
std::unique_ptr<clang::sema::RISCVIntrinsicManager>
CreateRISCVIntrinsicManager(Sema &S) {
  return std::make_unique<RISCVIntrinsicManagerImpl>(S);
}
} // namespace clang
