//===-- Header file of gnu_property_section -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_STARTUP_LINUX_GNU_PROPERTY_SECTION_H
#define LLVM_LIBC_STARTUP_LINUX_GNU_PROPERTY_SECTION_H

#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

struct GnuPropertyFeatures {
  // Set if the binary was compiled with SHSTK enabled and declares support.
  bool shstk_supported = false;
};

// This class parses the .note.gnu.property section within the ELF binary.
// Currently it only extracts the bit representing SHSTK support but can easily
// be expanded to other features included in it.
// The layout of the .note.gnu.property section and the program property is
// described in "System V Application Binary Interface - Linux Extensions"
// (https://github.com/hjl-tools/linux-abi/wiki).
class GnuPropertySection {
private:
  [[maybe_unused]] GnuPropertyFeatures features_;

public:
  LIBC_INLINE GnuPropertySection() = default;

  bool parse(const ElfW(Phdr) * gnu_property_phdr, const ElfW(Addr) base);

  LIBC_INLINE bool is_shstk_supported() const {
    return features_.shstk_supported;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_STARTUP_LINUX_GNU_PROPERTY_SECTION_H
