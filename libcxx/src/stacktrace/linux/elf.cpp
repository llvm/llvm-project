//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__linux__)

#  include "stacktrace/linux/elf.h"

#  include <__stacktrace/base.h>
#  include <cassert>
#  include <cstddef>
#  include <cstdlib>
#  include <functional>
#  include <unistd.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace::elf {

ELF::ELF(std::byte const* image) {
  auto* p = (uint8_t const*)image;
  // Bytes 0..3: magic bytes: 0x7F, 'E', 'L', 'F'
  if (*p++ == 0x7f && *p++ == 0x45 && *p++ == 0x4c && *p++ == 0x46) {
    auto klass       = *p++; // Byte 4 (EI_CLASS): ELF class, 32- or 64-bit (0x01 or 0x02)
    auto dataFormat  = *p++; // Byte 5 (EI_DATA): (0x01) little- or (0x02) big-endian
    auto fileVersion = *p++; // Byte 6 (EI_VERSION): ELF version: expect 1 (latest ELF version)
    constexpr static uint16_t kEndianTestWord{0x0201};
    auto hostEndianness = *(uint8_t const*)&kEndianTestWord;
    if (dataFormat == hostEndianness && fileVersion == 1) {
      if (klass == 0x01) {
        header_      = Header((Header32 const*)image);
        makeSection_ = makeSection32;
        makeSymbol_  = makeSymbol32;
        secSize_     = sizeof(Section32);
        symSize_     = sizeof(Symbol32);
      } else if (klass == 0x02) {
        header_      = Header((Header64 const*)image);
        makeSection_ = makeSection64;
        makeSymbol_  = makeSymbol64;
        secSize_     = sizeof(Section64);
        symSize_     = sizeof(Symbol64);
      }
    }
  }
  if (*this) {
    nametab_ = section(header_.shstrndx_);
    eachSection([&](auto& sec) mutable -> bool {
      if (sec.type_ == Section::kSymTab && sec.name() == ".symtab") {
        symtab_ = sec;
      } else if (sec.type_ == Section::kStrTab && sec.name() == ".strtab") {
        strtab_ = sec;
      }
      return !symtab_ || !strtab_;
    });
  }
  if (symtab_) {
    symCount_ = symtab_.size_ / symSize_;
  }
}

Section ELF::section(size_t index) {
  auto* addr = header_.ptr_ + header_.shoff_ + (index * secSize_);
  return makeSection_(this, addr);
}

Symbol ELF::symbol(size_t index) {
  auto* addr = symtab_.data() + (index * symSize_);
  return makeSymbol_(this, addr);
}

void ELF::eachSection(CB<Section> cb) {
  for (size_t i = 0; i < header_.shnum_ && cb(section(i)); i++)
    ;
}

void ELF::eachSymbol(CB<Symbol> cb) {
  for (size_t i = 0; i < symCount_ && cb(symbol(i)); i++)
    ;
}

Symbol ELF::getSym(uintptr_t addr) {
  Symbol ret{};
  eachSymbol([&](auto& sym) -> bool {
    if (sym.value_ <= addr && sym.value_ > ret.value_) {
      ret = sym;
    }
    return true;
  });
  return ret;
}

} // namespace __stacktrace::elf
_LIBCPP_END_NAMESPACE_STD

#endif // defined(__linux__)
