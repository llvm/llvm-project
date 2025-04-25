//===- comgr-elfdump.cpp - Print ELF Headers ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a routine to print ELF program headers, which is used
/// by the Comgr disassembly Actions (see comgr-objdump.cpp).
///
//===----------------------------------------------------------------------===//

#include "comgr-objdump.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

template <class ELFT>
void printProgramHeaders(const ELFFile<ELFT> &ELF, raw_ostream &OS) {
  typedef ELFFile<ELFT> ELFO;
  OS << "Program Header:\n";
  auto ProgramHeaderOrError = ELF.program_headers();
  if (!ProgramHeaderOrError) {
    report_fatal_error(
        Twine(errorToErrorCode(ProgramHeaderOrError.takeError()).message()));
  }
  for (const typename ELFO::Elf_Phdr &Phdr : *ProgramHeaderOrError) {
    switch (Phdr.p_type) {
    case ELF::PT_DYNAMIC:
      OS << " DYNAMIC ";
      break;
    case ELF::PT_GNU_EH_FRAME:
      OS << "EH_FRAME ";
      break;
    case ELF::PT_GNU_RELRO:
      OS << "   RELRO ";
      break;
    case ELF::PT_GNU_STACK:
      OS << "   STACK ";
      break;
    case ELF::PT_INTERP:
      OS << "  INTERP ";
      break;
    case ELF::PT_LOAD:
      OS << "    LOAD ";
      break;
    case ELF::PT_NOTE:
      OS << "    NOTE ";
      break;
    case ELF::PT_OPENBSD_BOOTDATA:
      OS << "    OPENBSD_BOOTDATA ";
      break;
    case ELF::PT_OPENBSD_RANDOMIZE:
      OS << "    OPENBSD_RANDOMIZE ";
      break;
    case ELF::PT_OPENBSD_WXNEEDED:
      OS << "    OPENBSD_WXNEEDED ";
      break;
    case ELF::PT_PHDR:
      OS << "    PHDR ";
      break;
    case ELF::PT_TLS:
      OS << "    TLS ";
      break;
    default:
      OS << " UNKNOWN ";
    }

    const char *Fmt = ELFT::Is64Bits ? "0x%016" PRIx64 " " : "0x%08" PRIx64 " ";

    OS << "off    " << format(Fmt, (uint64_t)Phdr.p_offset) << "vaddr "
       << format(Fmt, (uint64_t)Phdr.p_vaddr) << "paddr "
       << format(Fmt, (uint64_t)Phdr.p_paddr)
       << format("align 2**%u\n", countr_zero<uint64_t>(Phdr.p_align))
       << "         filesz " << format(Fmt, (uint64_t)Phdr.p_filesz) << "memsz "
       << format(Fmt, (uint64_t)Phdr.p_memsz) << "flags "
       << ((Phdr.p_flags & ELF::PF_R) ? "r" : "-")
       << ((Phdr.p_flags & ELF::PF_W) ? "w" : "-")
       << ((Phdr.p_flags & ELF::PF_X) ? "x" : "-") << "\n";
  }
  OS << "\n";
}

void COMGR::DisassemHelper::printELFFileHeader(const object::ObjectFile *Obj) {
  // Little-endian 32-bit
  if (const ELF32LEObjectFile *ELFObj = dyn_cast<ELF32LEObjectFile>(Obj)) {
    printProgramHeaders(ELFObj->getELFFile(), OutS);
  }

  // Big-endian 32-bit
  if (const ELF32BEObjectFile *ELFObj = dyn_cast<ELF32BEObjectFile>(Obj)) {
    printProgramHeaders(ELFObj->getELFFile(), OutS);
  }

  // Little-endian 64-bit
  if (const ELF64LEObjectFile *ELFObj = dyn_cast<ELF64LEObjectFile>(Obj)) {
    printProgramHeaders(ELFObj->getELFFile(), OutS);
  }

  // Big-endian 64-bit
  if (const ELF64BEObjectFile *ELFObj = dyn_cast<ELF64BEObjectFile>(Obj)) {
    printProgramHeaders(ELFObj->getELFFile(), OutS);
  }
}
