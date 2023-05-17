//===- bolt/Utils/Utils.cpp - Common helper functions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common helper functions.
//
//===----------------------------------------------------------------------===//

#include "bolt/Utils/Utils.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace bolt {

void report_error(StringRef Message, std::error_code EC) {
  assert(EC);
  errs() << "BOLT-ERROR: '" << Message << "': " << EC.message() << ".\n";
  exit(1);
}

void report_error(StringRef Message, Error E) {
  assert(E);
  errs() << "BOLT-ERROR: '" << Message << "': " << toString(std::move(E))
         << ".\n";
  exit(1);
}

void check_error(std::error_code EC, StringRef Message) {
  if (!EC)
    return;
  report_error(Message, EC);
}

void check_error(Error E, Twine Message) {
  if (!E)
    return;
  handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
    llvm::errs() << "BOLT-ERROR: '" << Message << "': " << EIB.message()
                 << '\n';
    exit(1);
  });
}

std::string getEscapedName(const StringRef &Name) {
  std::string Output = Name.str();
  for (size_t I = 0; I < Output.size(); ++I)
    if (Output[I] == ' ' || Output[I] == '\\')
      Output.insert(I++, 1, '\\');

  return Output;
}

std::string getUnescapedName(const StringRef &Name) {
  std::string Output = Name.str();
  for (size_t I = 0; I < Output.size(); ++I)
    if (Output[I] == '\\')
      Output.erase(I++, 1);

  return Output;
}

std::optional<StringRef> getLTOCommonName(const StringRef Name) {
  for (StringRef Suffix : {".__uniq.", ".lto_priv.", ".constprop.", ".llvm."}) {
    size_t LTOSuffixPos = Name.find(Suffix);
    if (LTOSuffixPos != StringRef::npos)
      return Name.substr(0, LTOSuffixPos + Suffix.size());
  }
  return std::nullopt;
}

std::optional<uint8_t> readDWARFExpressionTargetReg(StringRef ExprBytes) {
  uint8_t Opcode = ExprBytes[0];
  if (Opcode == dwarf::DW_CFA_def_cfa_expression)
    return std::nullopt;
  assert((Opcode == dwarf::DW_CFA_expression ||
          Opcode == dwarf::DW_CFA_val_expression) &&
         "invalid DWARF expression CFI");
  assert(ExprBytes.size() > 1 && "DWARF expression CFI is too short");
  const uint8_t *const Start =
      reinterpret_cast<const uint8_t *>(ExprBytes.drop_front(1).data());
  const uint8_t *const End =
      reinterpret_cast<const uint8_t *>(Start + ExprBytes.size() - 1);
  uint8_t Reg = decodeULEB128(Start, nullptr, End);
  return Reg;
}

} // namespace bolt

bool operator==(const llvm::MCCFIInstruction &L,
                const llvm::MCCFIInstruction &R) {
  if (L.getOperation() != R.getOperation())
    return false;
  switch (L.getOperation()) {
  case MCCFIInstruction::OpRestore:
  case MCCFIInstruction::OpSameValue:
  case MCCFIInstruction::OpUndefined:
  case MCCFIInstruction::OpDefCfaRegister:
    return L.getRegister() == R.getRegister();
  case MCCFIInstruction::OpRegister:
    return L.getRegister() == R.getRegister() &&
           L.getRegister2() == R.getRegister2();
  case MCCFIInstruction::OpOffset:
  case MCCFIInstruction::OpRelOffset:
  case MCCFIInstruction::OpDefCfa:
    return L.getRegister() == R.getRegister() && L.getOffset() == R.getOffset();
  case MCCFIInstruction::OpEscape:
    return L.getValues() == R.getValues();
  case MCCFIInstruction::OpRememberState:
  case MCCFIInstruction::OpRestoreState:
    return true;
  case MCCFIInstruction::OpDefCfaOffset:
  case MCCFIInstruction::OpAdjustCfaOffset:
    return L.getOffset() == R.getOffset();
  default:
    return false;
  }
}

} // namespace llvm
