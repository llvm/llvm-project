/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 1999 and
		  Jakub Jelinek <jakub@redhat.com>, 1999.

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

/* Returns 0 if everything is ok, != 0 in case of error.  */
int
process_elf_file (const char *file_name, const char *lib, int *flag,
		  unsigned int *osversion, unsigned int *isa_level,
		  char **soname, void *file_contents, size_t file_length)
{
  ElfW(Ehdr) *elf_header = (ElfW(Ehdr) *) file_contents;
  int ret;

  if (elf_header->e_ident [EI_CLASS] == ELFCLASS32)
    {
      Elf32_Ehdr *elf32_header = (Elf32_Ehdr *) elf_header;

      ret = process_elf32_file (file_name, lib, flag, osversion, isa_level,
				soname, file_contents, file_length);

      if (!ret && EF_ARM_EABI_VERSION (elf32_header->e_flags) == EF_ARM_EABI_VER5)
	{
	  if (elf32_header->e_flags & EF_ARM_ABI_FLOAT_HARD)
	    *flag = FLAG_ARM_LIBHF|FLAG_ELF_LIBC6;
	  else if (elf32_header->e_flags & EF_ARM_ABI_FLOAT_SOFT)
	    *flag = FLAG_ARM_LIBSF|FLAG_ELF_LIBC6;
	  else
	    /* We must assume the unmarked objects are compatible
	       with all ABI variants. Such objects may have been
	       generated in a transitional period when the ABI
	       tags were not added to all objects.  */
	    *flag = FLAG_ELF_LIBC6;
	}
    }
  else
    {
      ret = process_elf64_file (file_name, lib, flag, osversion, isa_level,
				soname, file_contents, file_length);
      /* AArch64 libraries are always libc.so.6+.  */
      if (!ret)
	*flag = FLAG_AARCH64_LIB64|FLAG_ELF_LIBC6;
    }
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
