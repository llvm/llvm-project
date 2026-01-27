//===-- Implementation file of gnu_property_section -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "startup/linux/gnu_property_section.h"

#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/utils.h"

namespace LIBC_NAMESPACE_DECL {

// The program property note is basically a note (Elf64_Nhdr) prepended with:
// * n_name[4]: should always be "GNU\0"
// * n_desc: an array of n_descsz bytes with program property entries
// Since we are casting a memory address into this struct, the layout needs to
// *exactly* match.
struct Elf64_ProgramPropertyNote {
  Elf64_Nhdr nhdr;
  unsigned char n_name[4];
  unsigned char n_desc[0]; // the size of 'n_desc' depends on n_descsz and is
                           // not known statically.
};

// 32-bit variant for ProgramPropertyNote.
struct Elf32_ProgramPropertyNote {
  Elf32_Nhdr nhdr;
  unsigned char n_name[4];
  unsigned char n_desc[0]; // the size of 'n_desc' depends on n_descsz and is
                           // not known statically.
};

// A program property consists of a type, the data size, followed by the actual
// data and potential padding (aligning it to 64bit).
// Since we are casting a memory address into this struct, the layout needs to
// *exactly* match (The padding is ommited since it doesn't have actual
// content).
// pr_data needs to be ElfW_Word aligned, therefore the whole struct needs to be
// aligned.
struct Elf64_ProgramProperty {
  Elf64_Word pr_type;
  Elf64_Word pr_datasz;
  unsigned char pr_data[0];
};

// 32-bit variant for ProgramProperty.
struct Elf32_ProgramProperty {
  Elf32_Word pr_type;
  Elf32_Word pr_datasz;
  unsigned char pr_data[0];
};

bool GnuPropertySection::parse(const ElfW(Phdr) * gnu_property_phdr,
                               const ElfW(Addr) base) {
  if (!gnu_property_phdr)
    return false;

  const auto note_nhdr_size = gnu_property_phdr->p_memsz;
  // Sanity check we are using the correct phdr and the memory size is large
  // enough to fit the program property note.
  if (gnu_property_phdr->p_type != PT_GNU_PROPERTY ||
      note_nhdr_size < sizeof(ElfW(ProgramPropertyNote)))
    return false;

  const ElfW(ProgramPropertyNote) *note_nhdr =
      reinterpret_cast<ElfW(ProgramPropertyNote) *>(base +
                                                    gnu_property_phdr->p_vaddr);
  if (!note_nhdr)
    return false;

  const ElfW(Word) nhdr_desc_size = note_nhdr->nhdr.n_descsz;

  // sizeof(*note_nhdr) does not include the size of n_desc,
  // since it is not known at compile time.
  // The size of it combined with n_descsz cannot exceed the total size of the
  // program property note.
  if ((sizeof(*note_nhdr) + nhdr_desc_size) > note_nhdr_size)
    return false;

  if (note_nhdr->nhdr.n_namesz != 4 ||
      note_nhdr->nhdr.n_type != NT_GNU_PROPERTY_TYPE_0 ||
      cpp::string_view(reinterpret_cast<const char *>(note_nhdr->n_name), 4) !=
          cpp::string_view("GNU", 4))
    return false;

  // program property note is valid, we can parse the program property array.
  ElfW(Word) offset = 0;
  // Process properties until we can no longer even fit the statically known
  // size of ProgramProperty.
  while (offset + sizeof(ElfW(ProgramProperty)) <= nhdr_desc_size) {
    const ElfW(ProgramProperty) *property =
        reinterpret_cast<const ElfW(ProgramProperty) *>(
            &note_nhdr->n_desc[offset]);

    // Sanity check that property is correctly aligned.
    if (distance_to_align_up<sizeof(ElfW(Word))>(property) > 0)
      return false;

    const ElfW(Xword) property_size = sizeof(*property) + property->pr_datasz;
    // Also check that pr_data does not reach out of bounds.
    if ((offset + property_size) > nhdr_desc_size)
      return false;

    switch (property->pr_type) {
#ifdef LIBC_TARGET_ARCH_IS_X86_64
    case GNU_PROPERTY_X86_FEATURE_1_AND: {
      // PR_DATASZ should always be 4 bytes, for both 32bit and 64bit.
      if (property->pr_datasz != 4)
        return false;

      const uint32_t feature_bitmap =
          *reinterpret_cast<const uint32_t *>(&property->pr_data[0]);
      features_.shstk_supported =
          (feature_bitmap & GNU_PROPERTY_X86_FEATURE_1_SHSTK) != 0;
      break;
    }
#endif
    default:
      break;
    }

    offset += static_cast<ElfW(Word)>(property_size);
  }

  return true;
}

} // namespace LIBC_NAMESPACE_DECL
