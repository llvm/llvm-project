//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_LINUX_ELF
#define _LIBCPP_STACKTRACE_LINUX_ELF

#include <__stacktrace/base.h>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <string_view>
#include <unistd.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace::elf {

// Includes ELF constants and structs copied from <elf.h>, with a few changes.

struct Header32 final {
  uint8_t ident[16];  /* Magic number and other info */
  uint16_t type;      /* Object file type */
  uint16_t machine;   /* Architecture */
  uint32_t version;   /* Object file version */
  uint32_t entry;     /* Entry point virtual address */
  uint32_t phoff;     /* Program header table file offset */
  uint32_t shoff;     /* Section header table file offset */
  uint32_t flags;     /* Processor-specific flags */
  uint16_t ehsize;    /* ELF header size in bytes */
  uint16_t phentsize; /* Program header table entry size */
  uint16_t phnum;     /* Program header table entry count */
  uint16_t shentsize; /* Section header table entry size */
  uint16_t shnum;     /* Section header table entry count */
  uint16_t shstrndx;  /* Section header string table index */
};

struct Section32 final {
  uint32_t name;      /* Section name (string tbl index) */
  uint32_t type;      /* Section type */
  uint32_t flags;     /* Section flags */
  uint32_t addr;      /* Section virtual addr at execution */
  uint32_t offset;    /* Section file offset */
  uint32_t size;      /* Section size in bytes */
  uint32_t link;      /* Link to another section */
  uint32_t info;      /* Additional section information */
  uint32_t addralign; /* Section alignment */
  uint32_t entsize;   /* Entry size if section holds table */
};

struct Symbol32 final {
  uint32_t name;  /* Symbol name (string tbl index) */
  uint32_t value; /* Symbol value */
  uint32_t size;  /* Symbol size */
  uint8_t info;   /* Symbol type and binding */
  uint8_t other;  /* Symbol visibility */
  uint16_t shndx; /* Section index */
};

struct Header64 final {
  uint8_t ident[16];  /* Magic number and other info */
  uint16_t type;      /* Object file type */
  uint16_t machine;   /* Architecture */
  uint32_t version;   /* Object file version */
  uint64_t entry;     /* Entry point virtual address */
  uint64_t phoff;     /* Program header table file offset */
  uint64_t shoff;     /* Section header table file offset */
  uint32_t flags;     /* Processor-specific flags */
  uint16_t ehsize;    /* ELF header size in bytes */
  uint16_t phentsize; /* Program header table entry size */
  uint16_t phnum;     /* Program header table entry count */
  uint16_t shentsize; /* Section header table entry size */
  uint16_t shnum;     /* Section header table entry count */
  uint16_t shstrndx;  /* Section header string table index */
};

struct Section64 final {
  uint32_t name;      /* Section name (string tbl index) */
  uint32_t type;      /* Section type */
  uint64_t flags;     /* Section flags */
  uint64_t addr;      /* Section virtual addr at execution */
  uint64_t offset;    /* Section file offset */
  uint64_t size;      /* Section size in bytes */
  uint32_t link;      /* Link to another section */
  uint32_t info;      /* Additional section information */
  uint64_t addralign; /* Section alignment */
  uint64_t entsize;   /* Entry size if section holds table */
};

struct Symbol64 final {
  uint32_t name;  /* Symbol name (string tbl index) */
  uint8_t info;   /* Symbol type and binding */
  uint8_t other;  /* Symbol visibility */
  uint16_t shndx; /* Section index */
  uint64_t value; /* Symbol value */
  uint64_t size;  /* Symbol size */
};

/** Represents an ELF header.  Supports the minimum needed to navigate an ELF file's sections and get at the symbol and
 * string tables. */
struct Header final {
  std::byte const* ptr_{};
  uintptr_t shoff_{};
  size_t shnum_{};
  size_t shstrndx_{};

  Header()              = default;
  Header(Header const&) = default;
  Header& operator=(Header const& rhs) { return *new (this) Header(rhs); }

  operator bool() { return ptr_; }

  template <class H>
  explicit Header(H* h)
      : ptr_((std::byte const*)h),
        shoff_(uintptr_t(h->shoff)),
        shnum_(size_t(h->shnum)),
        shstrndx_(size_t(h->shstrndx)) {}
};

struct ELF;
struct StringTable;

struct Section final {
  constexpr static uint32_t kSymTab = 2; // symbol table
  constexpr static uint32_t kStrTab = 3; // name table for symbols or sections

  ELF* elf_{};
  std::byte const* ptr_{};
  uintptr_t nameIndex_{};
  uint32_t type_{};
  uintptr_t offset_{};
  size_t size_{};

  Section() = default;

  template <class S>
  Section(ELF* elf, S* sec)
      : elf_(elf),
        ptr_((std::byte const*)sec),
        nameIndex_(sec->name),
        type_(sec->type),
        offset_(sec->offset),
        size_(sec->size) {}

  operator bool() const { return ptr_; }

  template <class T = std::byte>
  T const* data() const {
    return (T const*)(elfBase() + offset_);
  }

  std::byte const* elfBase() const;
  std::string_view name() const;
};

struct Symbol final {
  constexpr static uint8_t kFunc = 0x02; // STT_FUNC (code object)

  ELF* elf_{};
  std::byte const* ptr_{};
  uintptr_t nameIndex_{};
  uint32_t type_{};
  uintptr_t value_{};

  Symbol()              = default;
  Symbol(Symbol const&) = default;
  Symbol& operator=(Symbol const& rhs) { return *new (this) Symbol(rhs); }

  operator bool() { return ptr_; }

  bool isCode() const { return type_ == kFunc; }

  template <class S>
  Symbol(ELF* elf, S* sym)
      : elf_(elf), ptr_((std::byte const*)sym), nameIndex_(sym->name), type_(0x0f & sym->info), value_(sym->value) {}

  std::byte const* elfBase() const;
  std::string_view name() const;
};

/** Represents one of the ELF's `strtab`s.  This is a block of string data, with strings appended one after another, and
 * NUL-terminated.  Strings are indexed according to their starting offset.  At offset 0 is typically an empty string.
 */
struct StringTable {
  std::string_view data_{};

  StringTable() = default;

  /* implicit */ StringTable(Section const& sec) : data_(sec.data<char>(), sec.size_) {}

  operator bool() { return data_.size(); }

  std::string_view at(size_t index) {
    auto* ret = data_.data() + index;
    return {ret, strlen(ret)};
  }
};

/** Encapsulates an ELF image specified by byte-address (e.g. from an mmapped file or a program image or shared object
 * in memory).  If given a supported ELF image, this will test true with `operator bool` to indicate it is supported and
 * was able to parse some basic information from the header. */
struct ELF {
  Header header_{};
  Section (*makeSection_)(ELF*, std::byte const*){};
  Symbol (*makeSymbol_)(ELF*, std::byte const*){};
  size_t secSize_{};
  size_t symSize_{};
  StringTable nametab_{};
  Section symtab_{};
  StringTable strtab_{};
  size_t symCount_{};

  static Section makeSection32(ELF* elf, std::byte const* ptr) { return Section(elf, (Section32 const*)ptr); }
  static Section makeSection64(ELF* elf, std::byte const* ptr) { return Section(elf, (Section64 const*)ptr); }
  static Symbol makeSymbol32(ELF* elf, std::byte const* ptr) { return Symbol(elf, (Symbol32 const*)ptr); }
  static Symbol makeSymbol64(ELF* elf, std::byte const* ptr) { return Symbol(elf, (Symbol64 const*)ptr); }

  operator bool() { return header_; }

  explicit ELF(std::byte const* image);

  Section section(size_t index);
  Symbol symbol(size_t index);

  template <class T>
  using CB = std::function<bool(T const&)>;

  void eachSection(CB<Section> cb);
  void eachSymbol(CB<Symbol> cb);

  Symbol getSym(uintptr_t addr);
};

inline std::byte const* Section::elfBase() const { return elf_->header_.ptr_; }
inline std::byte const* Symbol::elfBase() const { return elf_->header_.ptr_; }

inline std::string_view Section::name() const { return elf_->nametab_.at(nameIndex_); }
inline std::string_view Symbol::name() const { return elf_->strtab_.at(nameIndex_); }

} // namespace __stacktrace::elf
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_LINUX_ELF
