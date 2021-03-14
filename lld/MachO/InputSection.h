//===- InputSection.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_INPUT_SECTION_H
#define LLD_MACHO_INPUT_SECTION_H

#include "Relocations.h"

#include "lld/Common/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/MachO.h"

namespace lld {
namespace macho {

class InputFile;
class InputSection;
class OutputSection;
class Symbol;
class Defined;

class InputSection {
public:
  virtual ~InputSection() = default;
  virtual uint64_t getSize() const { return data.size(); }
  virtual uint64_t getFileSize() const;
  uint64_t getFileOffset() const;
  uint64_t getVA() const;

  virtual void writeTo(uint8_t *buf);

  InputFile *file = nullptr;
  StringRef name;
  StringRef segname;

  OutputSection *parent = nullptr;
  uint64_t outSecOff = 0;
  uint64_t outSecFileOff = 0;

  uint32_t align = 1;
  uint32_t flags = 0;

  ArrayRef<uint8_t> data;
  std::vector<Reloc> relocs;
};

inline uint8_t sectionType(uint32_t flags) {
  return flags & llvm::MachO::SECTION_TYPE;
}

inline bool isZeroFill(uint32_t flags) {
  return llvm::MachO::isVirtualSection(sectionType(flags));
}

inline bool isThreadLocalVariables(uint32_t flags) {
  return sectionType(flags) == llvm::MachO::S_THREAD_LOCAL_VARIABLES;
}

// These sections contain the data for initializing thread-local variables.
inline bool isThreadLocalData(uint32_t flags) {
  return sectionType(flags) == llvm::MachO::S_THREAD_LOCAL_REGULAR ||
         sectionType(flags) == llvm::MachO::S_THREAD_LOCAL_ZEROFILL;
}

inline bool isDebugSection(uint32_t flags) {
  return (flags & llvm::MachO::SECTION_ATTRIBUTES_USR) ==
         llvm::MachO::S_ATTR_DEBUG;
}

bool isCodeSection(InputSection *);

extern std::vector<InputSection *> inputSections;

namespace section_names {

constexpr const char pageZero[] = "__pagezero";
constexpr const char common[] = "__common";
constexpr const char header[] = "__mach_header";
constexpr const char rebase[] = "__rebase";
constexpr const char binding[] = "__binding";
constexpr const char weakBinding[] = "__weak_binding";
constexpr const char lazyBinding[] = "__lazy_binding";
constexpr const char export_[] = "__export";
constexpr const char functionStarts[] = "__func_starts";
constexpr const char symbolTable[] = "__symbol_table";
constexpr const char indirectSymbolTable[] = "__ind_sym_tab";
constexpr const char stringTable[] = "__string_table";
constexpr const char codeSignature[] = "__code_signature";
constexpr const char got[] = "__got";
constexpr const char threadPtrs[] = "__thread_ptrs";
constexpr const char unwindInfo[] = "__unwind_info";
constexpr const char compactUnwind[] = "__compact_unwind";
constexpr const char ehFrame[] = "__eh_frame";
constexpr const char text[] = "__text";
constexpr const char stubs[] = "__stubs";
constexpr const char stubHelper[] = "__stub_helper";
constexpr const char laSymbolPtr[] = "__la_symbol_ptr";
constexpr const char data[] = "__data";

} // namespace section_names

} // namespace macho

std::string toString(const macho::InputSection *);

} // namespace lld

#endif
