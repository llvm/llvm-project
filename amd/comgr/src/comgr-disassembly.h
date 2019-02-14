/*******************************************************************************
*
* University of Illinois/NCSA
* Open Source License
*
* Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* with the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimers.
*
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimers in the
*       documentation and/or other materials provided with the distribution.
*
*     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
*       contributors may be used to endorse or promote products derived from
*       this Software without specific prior written permission.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
* THE SOFTWARE.
*
*******************************************************************************/

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
}

namespace COMGR {

typedef size_t (*ReadMemoryCallback)(uint64_t, char *, size_t, void *);
typedef void (*PrintInstructionCallback)(const char *, void *);
typedef void (*PrintAddressAnnotationCallback)(uint64_t, void *);

struct DisassemblyInfo {
  DisassemblyInfo(ReadMemoryCallback ReadMemory,
                  PrintInstructionCallback PrintInstruction,
                  PrintAddressAnnotationCallback PrintAddressAnnotation,
                  const Target *TheTarget, const MCAsmInfo *MAI,
                  const MCRegisterInfo *MRI, const MCSubtargetInfo *STI,
                  const MCInstrInfo *MII, const MCContext *Ctx,
                  const MCDisassembler *DisAsm, const MCInstrAnalysis *MIA,
                  MCInstPrinter *IP)
      : ReadMemory(ReadMemory), PrintInstruction(PrintInstruction),
        PrintAddressAnnotation(PrintAddressAnnotation), TheTarget(TheTarget) {
    this->MAI.reset(MAI);
    this->MRI.reset(MRI);
    this->STI.reset(STI);
    this->MII.reset(MII);
    this->Ctx.reset(Ctx);
    this->DisAsm.reset(DisAsm);
    this->MIA.reset(MIA);
    this->IP.reset(IP);
  }

  static amd_comgr_disassembly_info_t
  Convert(DisassemblyInfo *disassembly_info) {
    amd_comgr_disassembly_info_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(disassembly_info))};
    return handle;
  }

  static const amd_comgr_disassembly_info_t
  Convert(const DisassemblyInfo *disassembly_info) {
    const amd_comgr_disassembly_info_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(disassembly_info))};
    return handle;
  }

  static DisassemblyInfo *
  Convert(amd_comgr_disassembly_info_t disassembly_info) {
    return reinterpret_cast<DisassemblyInfo *>(disassembly_info.handle);
  }

  static amd_comgr_status_t
  Create(const TargetIdentifier &Ident, ReadMemoryCallback ReadMemory,
         PrintInstructionCallback PrintInstruction,
         PrintAddressAnnotationCallback PrintAddressAnnotation,
         amd_comgr_disassembly_info_t *DisassemblyInfoT);

  amd_comgr_status_t DisassembleInstruction(uint64_t Address, void *UserData,
                                            size_t &Size);

  ReadMemoryCallback ReadMemory;
  PrintInstructionCallback PrintInstruction;
  PrintAddressAnnotationCallback PrintAddressAnnotation;
  const Target *TheTarget;
  std::unique_ptr<const MCAsmInfo> MAI;
  std::unique_ptr<const MCRegisterInfo> MRI;
  std::unique_ptr<const MCSubtargetInfo> STI;
  std::unique_ptr<const MCInstrInfo> MII;
  std::unique_ptr<const MCContext> Ctx;
  std::unique_ptr<const MCDisassembler> DisAsm;
  std::unique_ptr<const MCInstrAnalysis> MIA;
  std::unique_ptr<MCInstPrinter> IP;
};

} // namespace COMGR

#endif // COMGR_DISASSEMBLY_H
