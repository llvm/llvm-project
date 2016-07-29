//===- EhFrame.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// .eh_frame section contains information on how to unwind the stack when
// an exception is thrown. The section consists of sequence of CIE and FDE
// records. The linker needs to merge CIEs and associate FDEs to CIEs.
// That means the linker has to understand the format of the section.
//
// This file contains a few utility functions to read .eh_frame contents.
//
//===----------------------------------------------------------------------===//

#include "EhFrame.h"
#include "Error.h"

#include "llvm/Object/ELF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::dwarf;
using namespace llvm::object;
using namespace llvm::support::endian;

namespace lld {
namespace elf {

// .eh_frame section is a sequence of records. Each record starts with
// a 4 byte length field. This function reads the length.
template <class ELFT> size_t readEhRecordSize(ArrayRef<uint8_t> D) {
  const endianness E = ELFT::TargetEndianness;
  if (D.size() < 4)
    fatal("CIE/FDE too small");

  // First 4 bytes of CIE/FDE is the size of the record.
  // If it is 0xFFFFFFFF, the next 8 bytes contain the size instead,
  // but we do not support that format yet.
  uint64_t V = read32<E>(D.data());
  if (V == UINT32_MAX)
    fatal("CIE/FDE too large");
  uint64_t Size = V + 4;
  if (Size > D.size())
    fatal("CIE/FIE ends past the end of the section");
  return Size;
}

// Read a byte and advance D by one byte.
static uint8_t readByte(ArrayRef<uint8_t> &D) {
  if (D.empty())
    fatal("corrupted or unsupported CIE information");
  uint8_t B = D.front();
  D = D.slice(1);
  return B;
}

// Skip an integer encoded in the LEB128 format.
// Actual number is not of interest because only the runtime needs it.
// But we need to be at least able to skip it so that we can read
// the field that follows a LEB128 number.
static void skipLeb128(ArrayRef<uint8_t> &D) {
  while (!D.empty()) {
    uint8_t Val = D.front();
    D = D.slice(1);
    if ((Val & 0x80) == 0)
      return;
  }
  fatal("corrupted or unsupported CIE information");
}

template <class ELFT> static size_t getAugPSize(unsigned Enc) {
  switch (Enc & 0x0f) {
  case DW_EH_PE_absptr:
  case DW_EH_PE_signed:
    return ELFT::Is64Bits ? 8 : 4;
  case DW_EH_PE_udata2:
  case DW_EH_PE_sdata2:
    return 2;
  case DW_EH_PE_udata4:
  case DW_EH_PE_sdata4:
    return 4;
  case DW_EH_PE_udata8:
  case DW_EH_PE_sdata8:
    return 8;
  }
  fatal("unknown FDE encoding");
}

template <class ELFT> static void skipAugP(ArrayRef<uint8_t> &D) {
  uint8_t Enc = readByte(D);
  if ((Enc & 0xf0) == DW_EH_PE_aligned)
    fatal("DW_EH_PE_aligned encoding is not supported");
  size_t Size = getAugPSize<ELFT>(Enc);
  if (Size >= D.size())
    fatal("corrupted CIE");
  D = D.slice(Size);
}

template <class ELFT> uint8_t getFdeEncoding(ArrayRef<uint8_t> D) {
  if (D.size() < 8)
    fatal("CIE too small");
  D = D.slice(8);

  uint8_t Version = readByte(D);
  if (Version != 1 && Version != 3)
    fatal("FDE version 1 or 3 expected, but got " + Twine((unsigned)Version));

  const unsigned char *AugEnd = std::find(D.begin(), D.end(), '\0');
  if (AugEnd == D.end())
    fatal("corrupted CIE");
  StringRef Aug(reinterpret_cast<const char *>(D.begin()), AugEnd - D.begin());
  D = D.slice(Aug.size() + 1);

  // Skip code alignment factor.
  skipLeb128(D);

  // Skip data alignment factor.
  skipLeb128(D);

  // Skip the return address register. In CIE version 1 this is a single
  // byte. In CIE version 3 this is an unsigned LEB128.
  if (Version == 1)
    readByte(D);
  else
    skipLeb128(D);

  // We only care about an 'R' value, but other records may precede an 'R'
  // record. Unfortunately records are not in TLV (type-length-value) format,
  // so we need to teach the linker how to skip records for each type.
  for (char C : Aug) {
    if (C == 'R')
      return readByte(D);
    if (C == 'z') {
      skipLeb128(D);
      continue;
    }
    if (C == 'P') {
      skipAugP<ELFT>(D);
      continue;
    }
    if (C == 'L') {
      readByte(D);
      continue;
    }
    fatal("unknown .eh_frame augmentation string: " + Aug);
  }
  return DW_EH_PE_absptr;
}

template size_t readEhRecordSize<ELF32LE>(ArrayRef<uint8_t>);
template size_t readEhRecordSize<ELF32BE>(ArrayRef<uint8_t>);
template size_t readEhRecordSize<ELF64LE>(ArrayRef<uint8_t>);
template size_t readEhRecordSize<ELF64BE>(ArrayRef<uint8_t>);

template uint8_t getFdeEncoding<ELF32LE>(ArrayRef<uint8_t>);
template uint8_t getFdeEncoding<ELF32BE>(ArrayRef<uint8_t>);
template uint8_t getFdeEncoding<ELF64LE>(ArrayRef<uint8_t>);
template uint8_t getFdeEncoding<ELF64BE>(ArrayRef<uint8_t>);
}
}
