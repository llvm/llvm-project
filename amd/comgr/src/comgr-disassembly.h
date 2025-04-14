//===- comgr-disassembly.h - Disassemble instruction ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_DISASSEMBLY_H
#define COMGR_DISASSEMBLY_H

#include "comgr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {
class Target;
} // namespace llvm

namespace COMGR {

typedef uint64_t (*ReadMemoryCallback)(uint64_t, char *, uint64_t, void *);
typedef void (*PrintInstructionCallback)(const char *, void *);
typedef void (*PrintAddressAnnotationCallback)(uint64_t, void *);

struct DisassemblyInfo {
  DisassemblyInfo(ReadMemoryCallback ReadMemory,
                  PrintInstructionCallback PrintInstruction,
                  PrintAddressAnnotationCallback PrintAddressAnnotation,
                  const llvm::Target *TheTarget,
                  std::unique_ptr<const llvm::MCAsmInfo> &&MAI,
                  std::unique_ptr<const llvm::MCRegisterInfo> &&MRI,
                  std::unique_ptr<const llvm::MCSubtargetInfo> &&STI,
                  std::unique_ptr<const llvm::MCInstrInfo> &&MII,
                  std::unique_ptr<const llvm::MCContext> &&Ctx,
                  std::unique_ptr<const llvm::MCDisassembler> &&DisAsm,
                  std::unique_ptr<const llvm::MCInstrAnalysis> &&MIA,
                  std::unique_ptr<llvm::MCInstPrinter> &&IP)
      : ReadMemory(ReadMemory), PrintInstruction(PrintInstruction),
        PrintAddressAnnotation(PrintAddressAnnotation), TheTarget(TheTarget),
        MAI(std::move(MAI)), MRI(std::move(MRI)), STI(std::move(STI)),
        MII(std::move(MII)), Ctx(std::move(Ctx)), DisAsm(std::move(DisAsm)),
        MIA(std::move(MIA)), IP(std::move(IP)) {}

  static amd_comgr_disassembly_info_t convert(DisassemblyInfo *DisasmInfo) {
    amd_comgr_disassembly_info_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DisasmInfo))};
    return Handle;
  }

  static const amd_comgr_disassembly_info_t
  convert(const DisassemblyInfo *DisasmInfo) {
    const amd_comgr_disassembly_info_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DisasmInfo))};
    return Handle;
  }

  static DisassemblyInfo *convert(amd_comgr_disassembly_info_t DisasmInfo) {
    return reinterpret_cast<DisassemblyInfo *>(DisasmInfo.handle);
  }

  static amd_comgr_status_t
  create(const TargetIdentifier &Ident, ReadMemoryCallback ReadMemory,
         PrintInstructionCallback PrintInstruction,
         PrintAddressAnnotationCallback PrintAddressAnnotation,
         amd_comgr_disassembly_info_t *DisassemblyInfoT);

  amd_comgr_status_t disassembleInstruction(uint64_t Address, void *UserData,
                                            uint64_t &Size);

  ReadMemoryCallback ReadMemory;
  PrintInstructionCallback PrintInstruction;
  PrintAddressAnnotationCallback PrintAddressAnnotation;
  const llvm::Target *TheTarget;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<const llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCSubtargetInfo> STI;
  std::unique_ptr<const llvm::MCInstrInfo> MII;
  std::unique_ptr<const llvm::MCContext> Ctx;
  std::unique_ptr<const llvm::MCDisassembler> DisAsm;
  std::unique_ptr<const llvm::MCInstrAnalysis> MIA;
  std::unique_ptr<llvm::MCInstPrinter> IP;
};

} // namespace COMGR

#endif // COMGR_DISASSEMBLY_H
