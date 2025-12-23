//===- comgr-disassembly.cpp - Disassemble instruction --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the internals for the amd_comgr_create_disassembly_info
/// and amd_comgr_disassemble_instruction APIs. They leverage the LLVM MC
/// (Machine Code Playground) implementation to disassemble individual
/// instructions.
///
//===----------------------------------------------------------------------===//

#include "comgr-disassembly.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;
using namespace COMGR;

amd_comgr_status_t
DisassemblyInfo::create(const TargetIdentifier &Ident,
                        ReadMemoryCallback ReadMemory,
                        PrintInstructionCallback PrintInstruction,
                        PrintAddressAnnotationCallback PrintAddressAnnotation,
                        amd_comgr_disassembly_info_t *DisassemblyInfoT) {
  std::string TT = (Twine(Ident.Arch) + "-" + Ident.Vendor + "-" + Ident.OS +
                    "-" + Ident.Environ)
                       .str();
  std::string Isa = TT + Twine("-" + Ident.Processor).str();
  SmallVector<std::string, 2> FeaturesVec;

  for (auto &Feature : Ident.Features) {
    FeaturesVec.push_back(
        Twine(Feature.take_back() + Feature.drop_back()).str());
  }

  std::string Features = join(FeaturesVec, ",");

  std::string Error;
  llvm::Triple TheTriple(TT);
  const Target *TheTarget = TargetRegistry::lookupTarget(TheTriple, Error);
  if (!TheTarget) {
    return AMD_COMGR_STATUS_ERROR;
  }

  std::unique_ptr<const MCRegisterInfo>
    MRI(TheTarget->createMCRegInfo(TheTriple));
  if (!MRI) {
    return AMD_COMGR_STATUS_ERROR;
  }

  llvm::MCTargetOptions MCOptions;
  std::unique_ptr<const MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TheTriple, MCOptions));
  if (!MAI) {
    return AMD_COMGR_STATUS_ERROR;
  }

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII) {
    return AMD_COMGR_STATUS_ERROR;
  }

  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TheTriple, Ident.Processor, Features));
  if (!STI) {
    return AMD_COMGR_STATUS_ERROR;
  }

  std::unique_ptr<MCContext> Ctx(new (std::nothrow) MCContext(
      Triple(TT), MAI.get(), MRI.get(), STI.get()));
  if (!Ctx) {
    return AMD_COMGR_STATUS_ERROR;
  }

  std::unique_ptr<const MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, *Ctx));
  if (!DisAsm) {
    return AMD_COMGR_STATUS_ERROR;
  }

  // Optional; currently AMDGPU does not implement this.
  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));

  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      Triple(TT), MAI->getAssemblerDialect(), *MAI, *MII, *MRI));
  if (!IP) {
    return AMD_COMGR_STATUS_ERROR;
  }

  DisassemblyInfo *DI = new (std::nothrow) DisassemblyInfo(
      ReadMemory, PrintInstruction, PrintAddressAnnotation, TheTarget,
      std::move(MAI), std::move(MRI), std::move(STI), std::move(MII),
      std::move(Ctx), std::move(DisAsm), std::move(MIA), std::move(IP));
  if (!DI) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *DisassemblyInfoT = DisassemblyInfo::convert(DI);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t DisassemblyInfo::disassembleInstruction(uint64_t Address,
                                                           void *UserData,
                                                           uint64_t &Size) {
  uint64_t ReadSize = MAI->getMaxInstLength();
  SmallVector<uint8_t, 16> Buffer(ReadSize);

  uint64_t ActualSize = ReadMemory(
      Address, reinterpret_cast<char *>(Buffer.data()), ReadSize, UserData);
  if (!ActualSize || ActualSize > ReadSize) {
    return AMD_COMGR_STATUS_ERROR;
  }

  Buffer.resize(ActualSize);

  MCInst Inst;
  std::string Annotations;
  raw_string_ostream AnnotationsStream(Annotations);
  if (DisAsm->getInstruction(Inst, Size, Buffer, Address, AnnotationsStream) !=
      MCDisassembler::Success) {
    return AMD_COMGR_STATUS_ERROR;
  }

  std::string InstStr;
  raw_string_ostream InstStream(InstStr);
  IP->printInst(&Inst, Address, AnnotationsStream.str(), *STI, InstStream);

  PrintInstruction(InstStream.str().c_str(), UserData);

  if (MIA && (MIA->isCall(Inst) || MIA->isUnconditionalBranch(Inst) ||
              MIA->isConditionalBranch(Inst))) {
    uint64_t Target;
    if (MIA->evaluateBranch(Inst, Address, Size, Target)) {
      PrintAddressAnnotation(Target, UserData);
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}
