//===- RISCVVEmitter.cpp - Generate riscv_vector.h for use with clang -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting riscv_vector.h which
// includes a declaration and definition of each intrinsic functions specified
// in https://github.com/riscv/rvv-intrinsic-doc.
//
// See also the documentation in include/clang/Basic/riscv_vector.td.
//
//===----------------------------------------------------------------------===//

#include "clang/Support/RISCVVIntrinsicUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <numeric>

using namespace llvm;
using namespace clang::RISCV;

namespace {
class RVVEmitter {
private:
  RecordKeeper &Records;

public:
  RVVEmitter(RecordKeeper &R) : Records(R) {}

  /// Emit riscv_vector.h
  void createHeader(raw_ostream &o);

  /// Emit all the __builtin prototypes and code needed by Sema.
  void createBuiltins(raw_ostream &o);

  /// Emit all the information needed to map builtin -> LLVM IR intrinsic.
  void createCodeGen(raw_ostream &o);

private:
  /// Create all intrinsics and add them to \p Out
  void createRVVIntrinsics(std::vector<std::unique_ptr<RVVIntrinsic>> &Out);
  /// Print HeaderCode in RVVHeader Record to \p Out
  void printHeaderCode(raw_ostream &OS);

  /// Emit Acrh predecessor definitions and body, assume the element of Defs are
  /// sorted by extension.
  void emitArchMacroAndBody(
      std::vector<std::unique_ptr<RVVIntrinsic>> &Defs, raw_ostream &o,
      std::function<void(raw_ostream &, const RVVIntrinsic &)>);

  // Emit the architecture preprocessor definitions. Return true when emits
  // non-empty string.
  bool emitMacroRestrictionStr(RISCVPredefinedMacroT PredefinedMacros,
                               raw_ostream &o);
};

} // namespace

static BasicType ParseBasicType(char c) {
  switch (c) {
  case 'c':
    return BasicType::Int8;
    break;
  case 's':
    return BasicType::Int16;
    break;
  case 'i':
    return BasicType::Int32;
    break;
  case 'l':
    return BasicType::Int64;
    break;
  case 'x':
    return BasicType::Float16;
    break;
  case 'f':
    return BasicType::Float32;
    break;
  case 'd':
    return BasicType::Float64;
    break;

  default:
    return BasicType::Unknown;
  }
}

void emitCodeGenSwitchBody(const RVVIntrinsic *RVVI, raw_ostream &OS) {
  if (!RVVI->getIRName().empty())
    OS << "  ID = Intrinsic::riscv_" + RVVI->getIRName() + ";\n";
  if (RVVI->getNF() >= 2)
    OS << "  NF = " + utostr(RVVI->getNF()) + ";\n";
  if (RVVI->hasManualCodegen()) {
    OS << RVVI->getManualCodegen();
    OS << "break;\n";
    return;
  }

  if (RVVI->isMasked()) {
    if (RVVI->hasVL()) {
      OS << "  std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);\n";
      if (RVVI->hasPolicyOperand())
        OS << "  Ops.push_back(ConstantInt::get(Ops.back()->getType(),"
              " TAIL_UNDISTURBED));\n";
    } else {
      OS << "  std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());\n";
    }
  } else {
    if (RVVI->hasPolicyOperand())
      OS << "  Ops.push_back(ConstantInt::get(Ops.back()->getType(), "
            "TAIL_UNDISTURBED));\n";
    else if (RVVI->hasPassthruOperand()) {
      OS << "  Ops.push_back(llvm::UndefValue::get(ResultType));\n";
      OS << "  std::rotate(Ops.rbegin(), Ops.rbegin() + 1,  Ops.rend());\n";
    }
  }

  OS << "  IntrinsicTypes = {";
  ListSeparator LS;
  for (const auto &Idx : RVVI->getIntrinsicTypes()) {
    if (Idx == -1)
      OS << LS << "ResultType";
    else
      OS << LS << "Ops[" << Idx << "]->getType()";
  }

  // VL could be i64 or i32, need to encode it in IntrinsicTypes. VL is
  // always last operand.
  if (RVVI->hasVL())
    OS << ", Ops.back()->getType()";
  OS << "};\n";
  OS << "  break;\n";
}

void emitIntrinsicFuncDef(const RVVIntrinsic &RVVI, raw_ostream &OS) {
  OS << "__attribute__((__clang_builtin_alias__(";
  OS << "__builtin_rvv_" << RVVI.getBuiltinName() << ")))\n";
  OS << RVVI.getOutputType()->getTypeStr() << " " << RVVI.getName() << "(";
  // Emit function arguments
  const RVVTypes &InputTypes = RVVI.getInputTypes();
  if (!InputTypes.empty()) {
    ListSeparator LS;
    for (unsigned i = 0; i < InputTypes.size(); ++i)
      OS << LS << InputTypes[i]->getTypeStr();
  }
  OS << ");\n";
}

void emitMangledFuncDef(const RVVIntrinsic &RVVI, raw_ostream &OS) {
  OS << "__attribute__((__clang_builtin_alias__(";
  OS << "__builtin_rvv_" << RVVI.getBuiltinName() << ")))\n";
  OS << RVVI.getOutputType()->getTypeStr() << " " << RVVI.getMangledName()
     << "(";
  // Emit function arguments
  const RVVTypes &InputTypes = RVVI.getInputTypes();
  if (!InputTypes.empty()) {
    ListSeparator LS;
    for (unsigned i = 0; i < InputTypes.size(); ++i)
      OS << LS << InputTypes[i]->getTypeStr();
  }
  OS << ");\n";
}

//===----------------------------------------------------------------------===//
// RVVEmitter implementation
//===----------------------------------------------------------------------===//
void RVVEmitter::createHeader(raw_ostream &OS) {

  OS << "/*===---- riscv_vector.h - RISC-V V-extension RVVIntrinsics "
        "-------------------===\n"
        " *\n"
        " *\n"
        " * Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n"
        " * See https://llvm.org/LICENSE.txt for license information.\n"
        " * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
        " *\n"
        " *===-----------------------------------------------------------------"
        "------===\n"
        " */\n\n";

  OS << "#ifndef __RISCV_VECTOR_H\n";
  OS << "#define __RISCV_VECTOR_H\n\n";

  OS << "#include <stdint.h>\n";
  OS << "#include <stddef.h>\n\n";

  OS << "#ifndef __riscv_vector\n";
  OS << "#error \"Vector intrinsics require the vector extension.\"\n";
  OS << "#endif\n\n";

  OS << "#ifdef __cplusplus\n";
  OS << "extern \"C\" {\n";
  OS << "#endif\n\n";

  printHeaderCode(OS);

  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  auto printType = [&](auto T) {
    OS << "typedef " << T->getClangBuiltinStr() << " " << T->getTypeStr()
       << ";\n";
  };

  constexpr int Log2LMULs[] = {-3, -2, -1, 0, 1, 2, 3};
  // Print RVV boolean types.
  for (int Log2LMUL : Log2LMULs) {
    auto T = RVVType::computeType(BasicType::Int8, Log2LMUL,
                                  PrototypeDescriptor::Mask);
    if (T.hasValue())
      printType(T.getValue());
  }
  // Print RVV int/float types.
  for (char I : StringRef("csil")) {
    BasicType BT = ParseBasicType(I);
    for (int Log2LMUL : Log2LMULs) {
      auto T = RVVType::computeType(BT, Log2LMUL, PrototypeDescriptor::Vector);
      if (T.hasValue()) {
        printType(T.getValue());
        auto UT = RVVType::computeType(
            BT, Log2LMUL,
            PrototypeDescriptor(BaseTypeModifier::Vector,
                                VectorTypeModifier::NoModifier,
                                TypeModifier::UnsignedInteger));
        printType(UT.getValue());
      }
    }
  }
  OS << "#if defined(__riscv_zvfh)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T = RVVType::computeType(BasicType::Float16, Log2LMUL,
                                  PrototypeDescriptor::Vector);
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n";

  OS << "#if defined(__riscv_f)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T = RVVType::computeType(BasicType::Float32, Log2LMUL,
                                  PrototypeDescriptor::Vector);
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n";

  OS << "#if defined(__riscv_d)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T = RVVType::computeType(BasicType::Float64, Log2LMUL,
                                  PrototypeDescriptor::Vector);
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n\n";

  // The same extension include in the same arch guard marco.
  llvm::stable_sort(Defs, [](const std::unique_ptr<RVVIntrinsic> &A,
                             const std::unique_ptr<RVVIntrinsic> &B) {
    return A->getRISCVPredefinedMacros() < B->getRISCVPredefinedMacros();
  });

  OS << "#define __rvv_ai static __inline__\n";

  // Print intrinsic functions with macro
  emitArchMacroAndBody(Defs, OS, [](raw_ostream &OS, const RVVIntrinsic &Inst) {
    OS << "__rvv_ai ";
    emitIntrinsicFuncDef(Inst, OS);
  });

  OS << "#undef __rvv_ai\n\n";

  OS << "#define __riscv_v_intrinsic_overloading 1\n";

  // Print Overloaded APIs
  OS << "#define __rvv_aio static __inline__ "
        "__attribute__((__overloadable__))\n";

  emitArchMacroAndBody(Defs, OS, [](raw_ostream &OS, const RVVIntrinsic &Inst) {
    if (!Inst.isMasked() && !Inst.hasUnMaskedOverloaded())
      return;
    OS << "__rvv_aio ";
    emitMangledFuncDef(Inst, OS);
  });

  OS << "#undef __rvv_aio\n";

  OS << "\n#ifdef __cplusplus\n";
  OS << "}\n";
  OS << "#endif // __cplusplus\n";
  OS << "#endif // __RISCV_VECTOR_H\n";
}

void RVVEmitter::createBuiltins(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  // Map to keep track of which builtin names have already been emitted.
  StringMap<RVVIntrinsic *> BuiltinMap;

  OS << "#if defined(TARGET_BUILTIN) && !defined(RISCVV_BUILTIN)\n";
  OS << "#define RISCVV_BUILTIN(ID, TYPE, ATTRS) TARGET_BUILTIN(ID, TYPE, "
        "ATTRS, \"zve32x\")\n";
  OS << "#endif\n";
  for (auto &Def : Defs) {
    auto P =
        BuiltinMap.insert(std::make_pair(Def->getBuiltinName(), Def.get()));
    if (!P.second) {
      // Verf that this would have produced the same builtin definition.
      if (P.first->second->hasBuiltinAlias() != Def->hasBuiltinAlias())
        PrintFatalError("Builtin with same name has different hasAutoDef");
      else if (!Def->hasBuiltinAlias() &&
               P.first->second->getBuiltinTypeStr() != Def->getBuiltinTypeStr())
        PrintFatalError("Builtin with same name has different type string");
      continue;
    }
    OS << "RISCVV_BUILTIN(__builtin_rvv_" << Def->getBuiltinName() << ",\"";
    if (!Def->hasBuiltinAlias())
      OS << Def->getBuiltinTypeStr();
    OS << "\", \"n\")\n";
  }
  OS << "#undef RISCVV_BUILTIN\n";
}

void RVVEmitter::createCodeGen(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);
  // IR name could be empty, use the stable sort preserves the relative order.
  llvm::stable_sort(Defs, [](const std::unique_ptr<RVVIntrinsic> &A,
                             const std::unique_ptr<RVVIntrinsic> &B) {
    return A->getIRName() < B->getIRName();
  });

  // Map to keep track of which builtin names have already been emitted.
  StringMap<RVVIntrinsic *> BuiltinMap;

  // Print switch body when the ir name or ManualCodegen changes from previous
  // iteration.
  RVVIntrinsic *PrevDef = Defs.begin()->get();
  for (auto &Def : Defs) {
    StringRef CurIRName = Def->getIRName();
    if (CurIRName != PrevDef->getIRName() ||
        (Def->getManualCodegen() != PrevDef->getManualCodegen())) {
      emitCodeGenSwitchBody(PrevDef, OS);
    }
    PrevDef = Def.get();

    auto P =
        BuiltinMap.insert(std::make_pair(Def->getBuiltinName(), Def.get()));
    if (P.second) {
      OS << "case RISCVVector::BI__builtin_rvv_" << Def->getBuiltinName()
         << ":\n";
      continue;
    }

    if (P.first->second->getIRName() != Def->getIRName())
      PrintFatalError("Builtin with same name has different IRName");
    else if (P.first->second->getManualCodegen() != Def->getManualCodegen())
      PrintFatalError("Builtin with same name has different ManualCodegen");
    else if (P.first->second->getNF() != Def->getNF())
      PrintFatalError("Builtin with same name has different NF");
    else if (P.first->second->isMasked() != Def->isMasked())
      PrintFatalError("Builtin with same name has different isMasked");
    else if (P.first->second->hasVL() != Def->hasVL())
      PrintFatalError("Builtin with same name has different hasVL");
    else if (P.first->second->getPolicyScheme() != Def->getPolicyScheme())
      PrintFatalError("Builtin with same name has different getPolicyScheme");
    else if (P.first->second->getIntrinsicTypes() != Def->getIntrinsicTypes())
      PrintFatalError("Builtin with same name has different IntrinsicTypes");
  }
  emitCodeGenSwitchBody(Defs.back().get(), OS);
  OS << "\n";
}

void RVVEmitter::createRVVIntrinsics(
    std::vector<std::unique_ptr<RVVIntrinsic>> &Out) {
  std::vector<Record *> RV = Records.getAllDerivedDefinitions("RVVBuiltin");
  for (auto *R : RV) {
    StringRef Name = R->getValueAsString("Name");
    StringRef SuffixProto = R->getValueAsString("Suffix");
    StringRef OverloadedName = R->getValueAsString("MangledName");
    StringRef OverloadedSuffixProto = R->getValueAsString("MangledSuffix");
    StringRef Prototypes = R->getValueAsString("Prototype");
    StringRef TypeRange = R->getValueAsString("TypeRange");
    bool HasMasked = R->getValueAsBit("HasMasked");
    bool HasMaskedOffOperand = R->getValueAsBit("HasMaskedOffOperand");
    bool HasVL = R->getValueAsBit("HasVL");
    Record *MaskedPolicyRecord = R->getValueAsDef("MaskedPolicy");
    PolicyScheme MaskedPolicy =
        static_cast<PolicyScheme>(MaskedPolicyRecord->getValueAsInt("Value"));
    Record *UnMaskedPolicyRecord = R->getValueAsDef("UnMaskedPolicy");
    PolicyScheme UnMaskedPolicy =
        static_cast<PolicyScheme>(UnMaskedPolicyRecord->getValueAsInt("Value"));
    bool HasUnMaskedOverloaded = R->getValueAsBit("HasUnMaskedOverloaded");
    std::vector<int64_t> Log2LMULList = R->getValueAsListOfInts("Log2LMUL");
    bool HasBuiltinAlias = R->getValueAsBit("HasBuiltinAlias");
    StringRef ManualCodegen = R->getValueAsString("ManualCodegen");
    StringRef MaskedManualCodegen = R->getValueAsString("MaskedManualCodegen");
    std::vector<int64_t> IntrinsicTypes =
        R->getValueAsListOfInts("IntrinsicTypes");
    std::vector<StringRef> RequiredFeatures =
        R->getValueAsListOfStrings("RequiredFeatures");
    StringRef IRName = R->getValueAsString("IRName");
    StringRef MaskedIRName = R->getValueAsString("MaskedIRName");
    unsigned NF = R->getValueAsInt("NF");

    // Parse prototype and create a list of primitive type with transformers
    // (operand) in Prototype. Prototype[0] is output operand.
    SmallVector<PrototypeDescriptor> Prototype = parsePrototypes(Prototypes);

    SmallVector<PrototypeDescriptor> SuffixDesc = parsePrototypes(SuffixProto);
    SmallVector<PrototypeDescriptor> OverloadedSuffixDesc =
        parsePrototypes(OverloadedSuffixProto);

    // Compute Builtin types
    SmallVector<PrototypeDescriptor> MaskedPrototype = Prototype;
    if (HasMasked) {
      // If HasMaskedOffOperand, insert result type as first input operand.
      if (HasMaskedOffOperand) {
        if (NF == 1) {
          MaskedPrototype.insert(MaskedPrototype.begin() + 1, Prototype[0]);
        } else {
          // Convert
          // (void, op0 address, op1 address, ...)
          // to
          // (void, op0 address, op1 address, ..., maskedoff0, maskedoff1, ...)
          PrototypeDescriptor MaskoffType = Prototype[1];
          MaskoffType.TM &= ~static_cast<uint8_t>(TypeModifier::Pointer);
          for (unsigned I = 0; I < NF; ++I)
            MaskedPrototype.insert(MaskedPrototype.begin() + NF + 1,
                                   MaskoffType);
        }
      }
      if (HasMaskedOffOperand && NF > 1) {
        // Convert
        // (void, op0 address, op1 address, ..., maskedoff0, maskedoff1, ...)
        // to
        // (void, op0 address, op1 address, ..., mask, maskedoff0, maskedoff1,
        // ...)
        MaskedPrototype.insert(MaskedPrototype.begin() + NF + 1,
                               PrototypeDescriptor::Mask);
      } else {
        // If HasMasked, insert PrototypeDescriptor:Mask as first input operand.
        MaskedPrototype.insert(MaskedPrototype.begin() + 1,
                               PrototypeDescriptor::Mask);
      }
    }
    // If HasVL, append PrototypeDescriptor:VL to last operand
    if (HasVL) {
      Prototype.push_back(PrototypeDescriptor::VL);
      MaskedPrototype.push_back(PrototypeDescriptor::VL);
    }

    // Create Intrinsics for each type and LMUL.
    for (char I : TypeRange) {
      for (int Log2LMUL : Log2LMULList) {
        BasicType BT = ParseBasicType(I);
        Optional<RVVTypes> Types =
            RVVType::computeTypes(BT, Log2LMUL, NF, Prototype);
        // Ignored to create new intrinsic if there are any illegal types.
        if (!Types.hasValue())
          continue;

        auto SuffixStr = RVVIntrinsic::getSuffixStr(BT, Log2LMUL, SuffixDesc);
        auto OverloadedSuffixStr =
            RVVIntrinsic::getSuffixStr(BT, Log2LMUL, OverloadedSuffixDesc);
        // Create a unmasked intrinsic
        Out.push_back(std::make_unique<RVVIntrinsic>(
            Name, SuffixStr, OverloadedName, OverloadedSuffixStr, IRName,
            /*IsMasked=*/false, /*HasMaskedOffOperand=*/false, HasVL,
            UnMaskedPolicy, HasUnMaskedOverloaded, HasBuiltinAlias,
            ManualCodegen, Types.getValue(), IntrinsicTypes, RequiredFeatures,
            NF));
        if (HasMasked) {
          // Create a masked intrinsic
          Optional<RVVTypes> MaskTypes =
              RVVType::computeTypes(BT, Log2LMUL, NF, MaskedPrototype);
          Out.push_back(std::make_unique<RVVIntrinsic>(
              Name, SuffixStr, OverloadedName, OverloadedSuffixStr,
              MaskedIRName,
              /*IsMasked=*/true, HasMaskedOffOperand, HasVL, MaskedPolicy,
              HasUnMaskedOverloaded, HasBuiltinAlias, MaskedManualCodegen,
              MaskTypes.getValue(), IntrinsicTypes, RequiredFeatures, NF));
        }
      } // end for Log2LMULList
    }   // end for TypeRange
  }
}

void RVVEmitter::printHeaderCode(raw_ostream &OS) {
  std::vector<Record *> RVVHeaders =
      Records.getAllDerivedDefinitions("RVVHeader");
  for (auto *R : RVVHeaders) {
    StringRef HeaderCodeStr = R->getValueAsString("HeaderCode");
    OS << HeaderCodeStr.str();
  }
}

void RVVEmitter::emitArchMacroAndBody(
    std::vector<std::unique_ptr<RVVIntrinsic>> &Defs, raw_ostream &OS,
    std::function<void(raw_ostream &, const RVVIntrinsic &)> PrintBody) {
  RISCVPredefinedMacroT PrevMacros =
      (*Defs.begin())->getRISCVPredefinedMacros();
  bool NeedEndif = emitMacroRestrictionStr(PrevMacros, OS);
  for (auto &Def : Defs) {
    RISCVPredefinedMacroT CurMacros = Def->getRISCVPredefinedMacros();
    if (CurMacros != PrevMacros) {
      if (NeedEndif)
        OS << "#endif\n\n";
      NeedEndif = emitMacroRestrictionStr(CurMacros, OS);
      PrevMacros = CurMacros;
    }
    if (Def->hasBuiltinAlias())
      PrintBody(OS, *Def);
  }
  if (NeedEndif)
    OS << "#endif\n\n";
}

bool RVVEmitter::emitMacroRestrictionStr(RISCVPredefinedMacroT PredefinedMacros,
                                         raw_ostream &OS) {
  if (PredefinedMacros == RISCVPredefinedMacro::Basic)
    return false;
  OS << "#if ";
  ListSeparator LS(" && ");
  if (PredefinedMacros & RISCVPredefinedMacro::V)
    OS << LS << "defined(__riscv_v)";
  if (PredefinedMacros & RISCVPredefinedMacro::Zvfh)
    OS << LS << "defined(__riscv_zvfh)";
  if (PredefinedMacros & RISCVPredefinedMacro::RV64)
    OS << LS << "(__riscv_xlen == 64)";
  if (PredefinedMacros & RISCVPredefinedMacro::VectorMaxELen64)
    OS << LS << "(__riscv_v_elen >= 64)";
  if (PredefinedMacros & RISCVPredefinedMacro::VectorMaxELenFp32)
    OS << LS << "(__riscv_v_elen_fp >= 32)";
  if (PredefinedMacros & RISCVPredefinedMacro::VectorMaxELenFp64)
    OS << LS << "(__riscv_v_elen_fp >= 64)";
  OS << "\n";
  return true;
}

namespace clang {
void EmitRVVHeader(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createHeader(OS);
}

void EmitRVVBuiltins(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createBuiltins(OS);
}

void EmitRVVBuiltinCG(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createCodeGen(OS);
}

} // End namespace clang
