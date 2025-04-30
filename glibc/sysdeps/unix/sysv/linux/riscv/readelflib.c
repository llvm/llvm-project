/* Support for reading ELF files.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */


int process_elf32_file (const char *file_name, const char *lib,
			int *flag, unsigned int *osversion,
			unsigned int *isa_level, char **soname,
			void *file_contents, size_t file_length);
int process_elf64_file (const char *file_name, const char *lib,
			int *flag, unsigned int *osversion,
			unsigned int *isa_level, char **soname,
			void *file_contents, size_t file_length);

/* The ELF flags supported by our current glibc port:
   - EF_RISCV_FLOAT_ABI: We support the soft and double ABIs.
   - EF_RISCV_RVC: While the Linux ABI mandates the presence of the C
     extension, we can still support libraries compiled without that extension
     so we just ignore this flag.
   - EF_RISCV_RVE: glibc (and Linux) don't support RV32E based systems.
   - EF_RISCV_TSO: The TSO extension isn't supported, as doing so would require
     some mechanism to ensure that the TSO extension is enabled which doesn't
     currently exist.  */
#define SUPPORTED_ELF_FLAGS (EF_RISCV_FLOAT_ABI | EF_RISCV_RVC)

/* Returns 0 if everything is ok, != 0 in case of error.  */
int
process_elf_file (const char *file_name, const char *lib, int *flag,
		  unsigned int *osversion, unsigned int *isa_level,
		  char **soname, void *file_contents, size_t file_length)
{
  ElfW(Ehdr) *elf_header = (ElfW(Ehdr) *) file_contents;
  Elf32_Ehdr *elf32_header = (Elf32_Ehdr *) elf_header;
  Elf64_Ehdr *elf64_header = (Elf64_Ehdr *) elf_header;
  int ret;
  long flags;

  /* RISC-V libraries are always libc.so.6+.  */
  *flag = FLAG_ELF_LIBC6;

  if (elf_header->e_ident [EI_CLASS] == ELFCLASS32)
    {
      ret = process_elf32_file (file_name, lib, flag, osversion, isa_level,
				soname, file_contents, file_length);
      flags = elf32_header->e_flags;
    }
  else
    {
      ret = process_elf64_file (file_name, lib, flag, osversion, isa_level,
				soname, file_contents, file_length);
      flags = elf64_header->e_flags;
    }

  /* RISC-V linkers encode the floating point ABI as part of the ELF headers.  */
  switch (flags & EF_RISCV_FLOAT_ABI)
    {
      case EF_RISCV_FLOAT_ABI_SOFT:
        *flag |= FLAG_RISCV_FLOAT_ABI_SOFT;
	break;
      case EF_RISCV_FLOAT_ABI_DOUBLE:
        *flag |= FLAG_RISCV_FLOAT_ABI_DOUBLE;
	break;
      default:
        return 1;
    }

  /* If there are any other ELF flags set then glibc doesn't support this
     library.  */
  if (flags & ~SUPPORTED_ELF_FLAGS)
    return 1;

  return ret;
}

#undef __ELF_NATIVE_CLASS
#undef process_elf_file
#define process_elf_file process_elf32_file
#define __ELF_NATIVE_CLASS 32
#include "elf/readelflib.c"

#undef __ELF_NATIVE_CLASS
#undef process_elf_file
#define process_elf_file process_elf64_file
#define __ELF_NATIVE_CLASS 64
#include "elf/readelflib.c"
