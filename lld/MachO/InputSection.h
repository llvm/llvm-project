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
#include "llvm/ADT/CachedHashString.h"
#include "llvm/BinaryFormat/MachO.h"

namespace lld {
namespace macho {

class InputFile;
class OutputSection;

class InputSection {
public:
  enum Kind {
    ConcatKind,
    CStringLiteralKind,
  };

  Kind kind() const { return sectionKind; }
  virtual ~InputSection() = default;
  virtual uint64_t getSize() const { return data.size(); }
  uint64_t getFileSize() const;
  // Translates \p off -- an offset relative to this InputSection -- into an
  // offset from the beginning of its parent OutputSection.
  virtual uint64_t getOffset(uint64_t off) const = 0;
  // The offset from the beginning of the file.
  virtual uint64_t getFileOffset(uint64_t off) const = 0;
  uint64_t getVA(uint64_t off) const;

  void writeTo(uint8_t *buf);

  InputFile *file = nullptr;
  StringRef name;
  StringRef segname;

  OutputSection *parent = nullptr;

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

protected:
  explicit InputSection(Kind kind) : sectionKind(kind) {}

private:
  Kind sectionKind;
};

// ConcatInputSections are combined into (Concat)OutputSections through simple
// concatentation, in contrast with literal sections which may have their
// contents merged before output.
class ConcatInputSection : public InputSection {
public:
  ConcatInputSection() : InputSection(ConcatKind) {}
  uint64_t getFileOffset(uint64_t off) const override;
  uint64_t getOffset(uint64_t off) const override { return outSecOff + off; }
  uint64_t getVA() const { return InputSection::getVA(0); }

  static bool classof(const InputSection *isec) {
    return isec->kind() == ConcatKind;
  }

  uint64_t outSecOff = 0;
  uint64_t outSecFileOff = 0;
};

// We allocate a lot of these and binary search on them, so they should be as
// compact as possible. Hence the use of 32 rather than 64 bits for the hash.
struct StringPiece {
  // Offset from the start of the containing input section.
  uint32_t inSecOff;
  uint32_t hash;
  // Offset from the start of the containing output section.
  uint64_t outSecOff = 0;

  StringPiece(uint64_t off, uint32_t hash) : inSecOff(off), hash(hash) {}
};

// CStringInputSections are composed of multiple null-terminated string
// literals, which we represent using StringPieces. These literals can be
// deduplicated and tail-merged, so translating offsets between the input and
// outputs sections is more complicated.
//
// NOTE: One significant difference between LLD and ld64 is that we merge all
// cstring literals, even those referenced directly by non-private symbols.
// ld64 is more conservative and does not do that. This was mostly done for
// implementation simplicity; if we find programs that need the more
// conservative behavior we can certainly implement that.
class CStringInputSection : public InputSection {
public:
  CStringInputSection() : InputSection(CStringLiteralKind) {}
  uint64_t getFileOffset(uint64_t off) const override;
  uint64_t getOffset(uint64_t off) const override;
  // Find the StringPiece that contains this offset.
  const StringPiece &getStringPiece(uint64_t off) const;
  // Split at each null byte.
  void splitIntoPieces();

  // Returns i'th piece as a CachedHashStringRef. This function is very hot when
  // string merging is enabled, so we want to inline.
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  llvm::CachedHashStringRef getCachedHashStringRef(size_t i) const {
    size_t begin = pieces[i].inSecOff;
    size_t end =
        (pieces.size() - 1 == i) ? data.size() : pieces[i + 1].inSecOff;
    return {toStringRef(data.slice(begin, end - begin)), pieces[i].hash};
  }

  static bool classof(const InputSection *isec) {
    return isec->kind() == CStringLiteralKind;
  }

  std::vector<StringPiece> pieces;
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
constexpr const char cString[] = "__cstring";
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
