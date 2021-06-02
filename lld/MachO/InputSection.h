//===- InputSection.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_INPUT_SECTION_H
#define LLD_MACHO_INPUT_SECTION_H

#include "Config.h"
#include "Relocations.h"

#include "lld/Common/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/MachO.h"

namespace lld {
namespace macho {

class InputFile;
class OutputSection;

class InputSection {
public:
  virtual ~InputSection() = default;
  virtual uint64_t getSize() const { return data.size(); }
  uint64_t getFileSize() const;
  uint64_t getFileOffset() const;
  uint64_t getVA() const;

  void writeTo(uint8_t *buf);

  InputFile *file = nullptr;
  StringRef name;
  StringRef segname;

  OutputSection *parent = nullptr;
  uint64_t outSecOff = 0;
  uint64_t outSecFileOff = 0;

  uint32_t align = 1;
  uint32_t flags = 0;
  uint32_t callSiteCount = 0;
  bool isFinal = false; // is address assigned?

  // How many symbols refer to this InputSection.
  uint32_t numRefs = 0;

  // With subsections_via_symbols, most symbols have their own InputSection,
  // and for weak symbols (e.g. from inline functions), only the
  // InputSection from one translation unit will make it to the output,
  // while all copies in other translation units are coalesced into the
  // first and not copied to the output.
  bool wasCoalesced = false;

  bool isCoalescedWeak() const { return wasCoalesced && numRefs == 0; }
  bool shouldOmitFromOutput() const { return !live || isCoalescedWeak(); }

  bool live = !config->deadStrip;

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

bool isCodeSection(const InputSection *);

extern std::vector<InputSection *> inputSections;

namespace section_names {

constexpr const char authGot[] = "__auth_got";
constexpr const char authPtr[] = "__auth_ptr";
constexpr const char binding[] = "__binding";
constexpr const char bitcodeBundle[] = "__bundle";
constexpr const char cfString[] = "__cfstring";
constexpr const char codeSignature[] = "__code_signature";
constexpr const char common[] = "__common";
constexpr const char compactUnwind[] = "__compact_unwind";
constexpr const char data[] = "__data";
constexpr const char debugAbbrev[] = "__debug_abbrev";
constexpr const char debugInfo[] = "__debug_info";
constexpr const char debugStr[] = "__debug_str";
constexpr const char ehFrame[] = "__eh_frame";
constexpr const char export_[] = "__export";
constexpr const char functionStarts[] = "__func_starts";
constexpr const char got[] = "__got";
constexpr const char header[] = "__mach_header";
constexpr const char indirectSymbolTable[] = "__ind_sym_tab";
constexpr const char const_[] = "__const";
constexpr const char lazySymbolPtr[] = "__la_symbol_ptr";
constexpr const char lazyBinding[] = "__lazy_binding";
constexpr const char moduleInitFunc[] = "__mod_init_func";
constexpr const char moduleTermFunc[] = "__mod_term_func";
constexpr const char nonLazySymbolPtr[] = "__nl_symbol_ptr";
constexpr const char objcCatList[] = "__objc_catlist";
constexpr const char objcClassList[] = "__objc_classlist";
constexpr const char objcConst[] = "__objc_const";
constexpr const char objcImageInfo[] = "__objc_imageinfo";
constexpr const char objcNonLazyCatList[] = "__objc_nlcatlist";
constexpr const char objcNonLazyClassList[] = "__objc_nlclslist";
constexpr const char objcProtoList[] = "__objc_protolist";
constexpr const char pageZero[] = "__pagezero";
constexpr const char pointers[] = "__pointers";
constexpr const char rebase[] = "__rebase";
constexpr const char staticInit[] = "__StaticInit";
constexpr const char stringTable[] = "__string_table";
constexpr const char stubHelper[] = "__stub_helper";
constexpr const char stubs[] = "__stubs";
constexpr const char swift[] = "__swift";
constexpr const char symbolTable[] = "__symbol_table";
constexpr const char textCoalNt[] = "__textcoal_nt";
constexpr const char text[] = "__text";
constexpr const char threadPtrs[] = "__thread_ptrs";
constexpr const char threadVars[] = "__thread_vars";
constexpr const char unwindInfo[] = "__unwind_info";
constexpr const char weakBinding[] = "__weak_binding";
constexpr const char zeroFill[] = "__zerofill";

} // namespace section_names

} // namespace macho

std::string toString(const macho::InputSection *);

} // namespace lld

#endif
