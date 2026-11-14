//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for GnuPropertySection ELF PT_GNU_PROPERTY note parsing.
///
//===----------------------------------------------------------------------===//

#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/macros/properties/architectures.h"
#include "startup/linux/gnu_property_section.h"
#include "test/UnitTest/Test.h"

namespace {

struct alignas(ElfW(Addr)) SinglePropertyNote {
  ElfW(Nhdr) nhdr;
  unsigned char name[4];
  ElfW(Word) pr_type;
  ElfW(Word) pr_datasz;
  uint32_t pr_data;
#if UINTPTR_MAX == UINT64_MAX
  uint32_t padding;
#endif
};

struct alignas(ElfW(Addr)) TwoPropertyNote {
  ElfW(Nhdr) nhdr;
  unsigned char name[4];
  ElfW(Word) first_type;
  ElfW(Word) first_datasz;
  uint32_t first_data;
#if UINTPTR_MAX == UINT64_MAX
  uint32_t first_padding;
#endif
  ElfW(Word) second_type;
  ElfW(Word) second_datasz;
  uint32_t second_data;
#if UINTPTR_MAX == UINT64_MAX
  uint32_t second_padding;
#endif
};

SinglePropertyNote make_x86_feature_note(uint32_t feature_bits) {
  SinglePropertyNote note{};
  note.nhdr.n_namesz = 4;
  note.nhdr.n_descsz = sizeof(SinglePropertyNote) - sizeof(ElfW(Nhdr)) - 4;
  note.nhdr.n_type = NT_GNU_PROPERTY_TYPE_0;
  note.name[0] = 'G';
  note.name[1] = 'N';
  note.name[2] = 'U';
  note.name[3] = '\0';
  note.pr_type = GNU_PROPERTY_X86_FEATURE_1_AND;
  note.pr_datasz = sizeof(uint32_t);
  note.pr_data = feature_bits;
  return note;
}

} // namespace

TEST(LlvmLibcGnuPropertySectionTest, RejectsNullOrInvalidPhdr) {
  LIBC_NAMESPACE::GnuPropertySection section;
  EXPECT_FALSE(section.parse(nullptr, 0));
  EXPECT_FALSE(section.is_shstk_supported());

  SinglePropertyNote note =
      make_x86_feature_note(GNU_PROPERTY_X86_FEATURE_1_SHSTK);
  ElfW(Phdr) phdr{};
  phdr.p_type = PT_LOAD;
  phdr.p_vaddr = reinterpret_cast<ElfW(Addr)>(&note);
  phdr.p_memsz = sizeof(note);
  EXPECT_FALSE(section.parse(&phdr, 0));

  phdr.p_type = PT_GNU_PROPERTY;
  phdr.p_memsz = sizeof(ElfW(Nhdr));
  EXPECT_FALSE(section.parse(&phdr, 0));
}

TEST(LlvmLibcGnuPropertySectionTest, RejectsMalformedNoteAndBounds) {
  SinglePropertyNote bad_name =
      make_x86_feature_note(GNU_PROPERTY_X86_FEATURE_1_SHSTK);
  bad_name.name[2] = 'X';

  ElfW(Phdr) phdr{};
  phdr.p_type = PT_GNU_PROPERTY;
  phdr.p_vaddr = reinterpret_cast<ElfW(Addr)>(&bad_name);
  phdr.p_memsz = sizeof(bad_name);

  LIBC_NAMESPACE::GnuPropertySection section;
  EXPECT_FALSE(section.parse(&phdr, 0));

  SinglePropertyNote bad_descsz =
      make_x86_feature_note(GNU_PROPERTY_X86_FEATURE_1_SHSTK);
  bad_descsz.nhdr.n_descsz = static_cast<ElfW(Word)>(sizeof(bad_descsz));
  phdr.p_vaddr = reinterpret_cast<ElfW(Addr)>(&bad_descsz);
  EXPECT_FALSE(section.parse(&phdr, 0));

  SinglePropertyNote truncated_prop =
      make_x86_feature_note(GNU_PROPERTY_X86_FEATURE_1_SHSTK);
  truncated_prop.pr_datasz = 32;
  phdr.p_vaddr = reinterpret_cast<ElfW(Addr)>(&truncated_prop);
  EXPECT_FALSE(section.parse(&phdr, 0));
}

TEST(LlvmLibcGnuPropertySectionTest, ParsesShstkBitAndNonZeroBase) {
  constexpr ElfW(Addr) FAKE_VADDR = 0x2000;
  SinglePropertyNote with_shstk =
      make_x86_feature_note(GNU_PROPERTY_X86_FEATURE_1_SHSTK);

  ElfW(Phdr) phdr{};
  phdr.p_type = PT_GNU_PROPERTY;
  phdr.p_vaddr = FAKE_VADDR;
  phdr.p_memsz = sizeof(with_shstk);
  const ElfW(Addr) BASE =
      reinterpret_cast<ElfW(Addr)>(&with_shstk) - FAKE_VADDR;

  LIBC_NAMESPACE::GnuPropertySection section;
  ASSERT_TRUE(section.parse(&phdr, BASE));
#if defined(LIBC_TARGET_ARCH_IS_X86_64)
  EXPECT_TRUE(section.is_shstk_supported());
#else
  EXPECT_FALSE(section.is_shstk_supported());
#endif

  SinglePropertyNote without_shstk = make_x86_feature_note(0);
  phdr.p_vaddr = reinterpret_cast<ElfW(Addr)>(&without_shstk);
  LIBC_NAMESPACE::GnuPropertySection section_no_shstk;
  ASSERT_TRUE(section_no_shstk.parse(&phdr, 0));
  EXPECT_FALSE(section_no_shstk.is_shstk_supported());
}

TEST(LlvmLibcGnuPropertySectionTest, ParsesMultiplePaddedProperties) {
  constexpr ElfW(Word) ISA_NEEDED_PROP = 0xc0008002U;
  TwoPropertyNote note{};
  note.nhdr.n_namesz = 4;
  note.nhdr.n_descsz = sizeof(TwoPropertyNote) - sizeof(ElfW(Nhdr)) - 4;
  note.nhdr.n_type = NT_GNU_PROPERTY_TYPE_0;
  note.name[0] = 'G';
  note.name[1] = 'N';
  note.name[2] = 'U';
  note.name[3] = '\0';
  note.first_type = ISA_NEEDED_PROP;
  note.first_datasz = sizeof(uint32_t);
  note.first_data = 0x1U;
  note.second_type = GNU_PROPERTY_X86_FEATURE_1_AND;
  note.second_datasz = sizeof(uint32_t);
  note.second_data = GNU_PROPERTY_X86_FEATURE_1_SHSTK;

  ElfW(Phdr) phdr{};
  phdr.p_type = PT_GNU_PROPERTY;
  phdr.p_vaddr = reinterpret_cast<ElfW(Addr)>(&note);
  phdr.p_memsz = sizeof(note);

  LIBC_NAMESPACE::GnuPropertySection section;
  ASSERT_TRUE(section.parse(&phdr, 0));
#if defined(LIBC_TARGET_ARCH_IS_X86_64)
  EXPECT_TRUE(section.is_shstk_supported());
#else
  EXPECT_FALSE(section.is_shstk_supported());
#endif
}
