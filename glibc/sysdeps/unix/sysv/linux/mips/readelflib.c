/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Alexandre Oliva <aoliva@redhat.com>
   Based on work ../x86_64/readelflib.c,
   contributed by Andreas Jaeger <aj@suse.de>, 1999 and
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
   License along with the GNU C Library.  If not, see
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
  union
    {
      Elf64_Ehdr *eh64;
      Elf32_Ehdr *eh32;
      ElfW(Ehdr) *eh;
    }
  elf_header;
  int ret;

  elf_header.eh = file_contents;
  if (elf_header.eh->e_ident [EI_CLASS] == ELFCLASS32)
    {
      ret = process_elf32_file (file_name, lib, flag, osversion, isa_level,
				soname, file_contents, file_length);
      if (!ret)
	{
	  Elf32_Word flags = elf_header.eh32->e_flags;
	  int nan2008 = (flags & EF_MIPS_NAN2008) != 0;

	  /* n32 libraries are always libc.so.6+, o32 only if 2008 NaN.  */
	  if ((flags & EF_MIPS_ABI2) != 0)
	    *flag = (nan2008 ? FLAG_MIPS64_LIBN32_NAN2008
		     : FLAG_MIPS64_LIBN32) | FLAG_ELF_LIBC6;
	  else if (nan2008)
	    *flag = FLAG_MIPS_LIB32_NAN2008 | FLAG_ELF_LIBC6;
	}
    }
  else
    {
      ret = process_elf64_file (file_name, lib, flag, osversion, isa_level,
				soname, file_contents, file_length);
      /* n64 libraries are always libc.so.6+.  */
      if (!ret)
	{
	  Elf64_Word flags = elf_header.eh64->e_flags;
	  int nan2008 = (flags & EF_MIPS_NAN2008) != 0;

	  *flag = (nan2008 ? FLAG_MIPS64_LIBN64_NAN2008
		   : FLAG_MIPS64_LIBN64) | FLAG_ELF_LIBC6;
	}
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
