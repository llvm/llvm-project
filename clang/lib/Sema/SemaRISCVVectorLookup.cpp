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
#include <optional>
#include <string>
#include <vector>

using namespace llvm;
using namespace clang;
using namespace clang::RISCV;

using IntrinsicKind = sema::RISCVIntrinsicManager::IntrinsicKind;

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

static const PrototypeDescriptor RVSiFiveVectorSignatureTable[] = {
#define DECL_SIGNATURE_TABLE
#include "clang/Basic/riscv_sifive_vector_builtin_sema.inc"
#undef DECL_SIGNATURE_TABLE
};

static const RVVIntrinsicRecord RVVIntrinsicRecords[] = {
#define DECL_INTRINSIC_RECORDS
#include "clang/Basic/riscv_vector_builtin_sema.inc"
#undef DECL_INTRINSIC_RECORDS
};

static const RVVIntrinsicRecord RVSiFiveVectorIntrinsicRecords[] = {
#define DECL_INTRINSIC_RECORDS
#include "clang/Basic/riscv_sifive_vector_builtin_sema.inc"
#undef DECL_INTRINSIC_RECORDS
};

// Get subsequence of signature table.
static ArrayRef<PrototypeDescriptor>
ProtoSeq2ArrayRef(IntrinsicKind K, uint16_t Index, uint8_t Length) {
  switch (K) {
  case IntrinsicKind::RVV:
    return ArrayRef(&RVVSignatureTable[Index], Length);
  case IntrinsicKind::SIFIVE_VECTOR:
    return ArrayRef(&RVSiFiveVectorSignatureTable[Index], Length);
  }
  llvm_unreachable("Unhandled IntrinsicKind");
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
  case Undefined:
    llvm_unreachable("Unhandled type.");
  }
  if (Type->isVector()) {
    if (Type->isTuple())
      QT = Context.getScalableVectorType(QT, *Type->getScale(), Type->getNF());
    else
      QT = Context.getScalableVectorType(QT, *Type->getScale());
  }

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
  RVVTypeCache TypeCache;
  bool ConstructedRISCVVBuiltins;
  bool ConstructedRISCVSiFiveVectorBuiltins;

  // List of all RVV intrinsic.
  std::vector<RVVIntrinsicDef> IntrinsicList;
  // Mapping function name to index of IntrinsicList.
  StringMap<size_t> Intrinsics;
  // Mapping function name to RVVOverloadIntrinsicDef.
  StringMap<RVVOverloadIntrinsicDef> OverloadIntrinsics;


  // Create RVVIntrinsicDef.
  void InitRVVIntrinsic(const RVVIntrinsicRecord &Record, StringRef SuffixStr,
                        StringRef OverloadedSuffixStr, bool IsMask,
                        RVVTypes &Types, bool HasPolicy, Policy PolicyAttrs);

  // Create FunctionDecl for a vector intrinsic.
  void CreateRVVIntrinsicDecl(LookupResult &LR, IdentifierInfo *II,
                              Preprocessor &PP, unsigned Index,
                              bool IsOverload);

  void ConstructRVVIntrinsics(ArrayRef<RVVIntrinsicRecord> Recs,
                              IntrinsicKind K);

public:
  RISCVIntrinsicManagerImpl(clang::Sema &S) : S(S), Context(S.Context) {
    ConstructedRISCVVBuiltins = false;
    ConstructedRISCVSiFiveVectorBuiltins = false;
  }

  // Initialize IntrinsicList
  void InitIntrinsicList() override;

  // Create RISC-V vector intrinsic and insert into symbol table if found, and
  // return true, otherwise return false.
  bool CreateIntrinsicIfFound(LookupResult &LR, IdentifierInfo *II,
                              Preprocessor &PP) override;
};
} // namespace

void RISCVIntrinsicManagerImpl::ConstructRVVIntrinsics(
    ArrayRef<RVVIntrinsicRecord> Recs, IntrinsicKind K) {
  const TargetInfo &TI = Context.getTargetInfo();
  static const std::pair<const char *, RVVRequire> FeatureCheckList[] = {
      {"64bit", RVV_REQ_RV64},
      {"xsfvcp", RVV_REQ_Xsfvcp},
      {"xsfvqmaccdod", RVV_REQ_Xsfvqmaccdod},
      {"xsfvqmaccqoq", RVV_REQ_Xsfvqmaccqoq},
      {"experimental-zvbb", RVV_REQ_Zvbb},
      {"experimental-zvbc", RVV_REQ_Zvbc},
      {"experimental-zvkb", RVV_REQ_Zvkb},
      {"experimental-zvkg", RVV_REQ_Zvkg},
      {"experimental-zvkned", RVV_REQ_Zvkned},
      {"experimental-zvknha", RVV_REQ_Zvknha},
      {"experimental-zvksed", RVV_REQ_Zvksed},
      {"experimental-zvksh", RVV_REQ_Zvksh}};

  // Construction of RVVIntrinsicRecords need to sync with createRVVIntrinsics
  // in RISCVVEmitter.cpp.
  for (auto &Record : Recs) {
    // Check requirements.
    if (llvm::any_of(FeatureCheckList, [&](const auto &Item) {
          return (Record.RequiredExtensions & Item.second) == Item.second &&
                 !TI.hasFeature(Item.first);
        }))
      continue;

    // Create Intrinsics for each type and LMUL.
    BasicType BaseType = BasicType::Unknown;
    ArrayRef<PrototypeDescriptor> BasicProtoSeq =
        ProtoSeq2ArrayRef(K, Record.PrototypeIndex, Record.PrototypeLength);
    ArrayRef<PrototypeDescriptor> SuffixProto =
        ProtoSeq2ArrayRef(K, Record.SuffixIndex, Record.SuffixLength);
    ArrayRef<PrototypeDescriptor> OverloadedSuffixProto = ProtoSeq2ArrayRef(
        K, Record.OverloadedSuffixIndex, Record.OverloadedSuffixSize);

    PolicyScheme UnMaskedPolicyScheme =
        static_cast<PolicyScheme>(Record.UnMaskedPolicyScheme);
    PolicyScheme MaskedPolicyScheme =
        static_cast<PolicyScheme>(Record.MaskedPolicyScheme);

    const Policy DefaultPolicy;

    llvm::SmallVector<PrototypeDescriptor> ProtoSeq =
        RVVIntrinsic::computeBuiltinTypes(
            BasicProtoSeq, /*IsMasked=*/false,
            /*HasMaskedOffOperand=*/false, Record.HasVL, Record.NF,
            UnMaskedPolicyScheme, DefaultPolicy, Record.IsTuple);

    llvm::SmallVector<PrototypeDescriptor> ProtoMaskSeq;
    if (Record.HasMasked)
      ProtoMaskSeq = RVVIntrinsic::computeBuiltinTypes(
          BasicProtoSeq, /*IsMasked=*/true, Record.HasMaskedOffOperand,
          Record.HasVL, Record.NF, MaskedPolicyScheme, DefaultPolicy,
          Record.IsTuple);

    bool UnMaskedHasPolicy = UnMaskedPolicyScheme != PolicyScheme::SchemeNone;
    bool MaskedHasPolicy = MaskedPolicyScheme != PolicyScheme::SchemeNone;
    SmallVector<Policy> SupportedUnMaskedPolicies =
        RVVIntrinsic::getSupportedUnMaskedPolicies();
    SmallVector<Policy> SupportedMaskedPolicies =
        RVVIntrinsic::getSupportedMaskedPolicies(Record.HasTailPolicy,
                                                 Record.HasMaskPolicy);

    for (unsigned int TypeRangeMaskShift = 0;
         TypeRangeMaskShift <= static_cast<unsigned int>(BasicType::MaxOffset);
         ++TypeRangeMaskShift) {
      unsigned int BaseTypeI = 1 << TypeRangeMaskShift;
      BaseType = static_cast<BasicType>(BaseTypeI);

      if ((BaseTypeI & Record.TypeRangeMask) != BaseTypeI)
        continue;

      if (BaseType == BasicType::Float16) {
        if ((Record.RequiredExtensions & RVV_REQ_ZvfhminOrZvfh) ==
            RVV_REQ_ZvfhminOrZvfh) {
          if (!TI.hasFeature("zvfh") && !TI.hasFeature("zvfhmin"))
            continue;
        } else if (!TI.hasFeature("zvfh")) {
          continue;
        }
      }

      // Expanded with different LMUL.
      for (int Log2LMUL = -3; Log2LMUL <= 3; Log2LMUL++) {
        if (!(Record.Log2LMULMask & (1 << (Log2LMUL + 3))))
          continue;

        std::optional<RVVTypes> Types =
            TypeCache.computeTypes(BaseType, Log2LMUL, Record.NF, ProtoSeq);

        // Ignored to create new intrinsic if there are any illegal types.
        if (!Types.has_value())
          continue;

        std::string SuffixStr = RVVIntrinsic::getSuffixStr(
            TypeCache, BaseType, Log2LMUL, SuffixProto);
        std::string OverloadedSuffixStr = RVVIntrinsic::getSuffixStr(
            TypeCache, BaseType, Log2LMUL, OverloadedSuffixProto);

        // Create non-masked intrinsic.
        InitRVVIntrinsic(Record, SuffixStr, OverloadedSuffixStr, false, *Types,
                         UnMaskedHasPolicy, DefaultPolicy);

        // Create non-masked policy intrinsic.
        if (Record.UnMaskedPolicyScheme != PolicyScheme::SchemeNone) {
          for (auto P : SupportedUnMaskedPolicies) {
            llvm::SmallVector<PrototypeDescriptor> PolicyPrototype =
                RVVIntrinsic::computeBuiltinTypes(
                    BasicProtoSeq, /*IsMasked=*/false,
                    /*HasMaskedOffOperand=*/false, Record.HasVL, Record.NF,
                    UnMaskedPolicyScheme, P, Record.IsTuple);
            std::optional<RVVTypes> PolicyTypes = TypeCache.computeTypes(
                BaseType, Log2LMUL, Record.NF, PolicyPrototype);
            InitRVVIntrinsic(Record, SuffixStr, OverloadedSuffixStr,
                             /*IsMask=*/false, *PolicyTypes, UnMaskedHasPolicy,
                             P);
          }
        }
        if (!Record.HasMasked)
          continue;
        // Create masked intrinsic.
        std::optional<RVVTypes> MaskTypes =
            TypeCache.computeTypes(BaseType, Log2LMUL, Record.NF, ProtoMaskSeq);
        InitRVVIntrinsic(Record, SuffixStr, OverloadedSuffixStr, true,
                         *MaskTypes, MaskedHasPolicy, DefaultPolicy);
        if (Record.MaskedPolicyScheme == PolicyScheme::SchemeNone)
          continue;
        // Create masked policy intrinsic.
        for (auto P : SupportedMaskedPolicies) {
          llvm::SmallVector<PrototypeDescriptor> PolicyPrototype =
              RVVIntrinsic::computeBuiltinTypes(
                  BasicProtoSeq, /*IsMasked=*/true, Record.HasMaskedOffOperand,
                  Record.HasVL, Record.NF, MaskedPolicyScheme, P,
                  Record.IsTuple);
          std::optional<RVVTypes> PolicyTypes = TypeCache.computeTypes(
              BaseType, Log2LMUL, Record.NF, PolicyPrototype);
          InitRVVIntrinsic(Record, SuffixStr, OverloadedSuffixStr,
                           /*IsMask=*/true, *PolicyTypes, MaskedHasPolicy, P);
        }
      } // End for different LMUL
    }   // End for different TypeRange
  }
}

void RISCVIntrinsicManagerImpl::InitIntrinsicList() {

  if (S.DeclareRISCVVBuiltins && !ConstructedRISCVVBuiltins) {
    ConstructedRISCVVBuiltins = true;
    ConstructRVVIntrinsics(RVVIntrinsicRecords,
                           IntrinsicKind::RVV);
  }
  if (S.DeclareRISCVSiFiveVectorBuiltins &&
      !ConstructedRISCVSiFiveVectorBuiltins) {
    ConstructedRISCVSiFiveVectorBuiltins = true;
    ConstructRVVIntrinsics(RVSiFiveVectorIntrinsicRecords,
                           IntrinsicKind::SIFIVE_VECTOR);
  }
}

// Compute name and signatures for intrinsic with practical types.
void RISCVIntrinsicManagerImpl::InitRVVIntrinsic(
    const RVVIntrinsicRecord &Record, StringRef SuffixStr,
    StringRef OverloadedSuffixStr, bool IsMasked, RVVTypes &Signature,
    bool HasPolicy, Policy PolicyAttrs) {
  // Function name, e.g. vadd_vv_i32m1.
  std::string Name = Record.Name;
  if (!SuffixStr.empty())
    Name += "_" + SuffixStr.str();

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

  RVVIntrinsic::updateNamesAndPolicy(IsMasked, HasPolicy, Name, BuiltinName,
                                     OverloadedName, PolicyAttrs,
                                     Record.HasFRMRoundModeOp);

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
