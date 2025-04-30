/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 1999 and
		  Jakub Jelinek <jakub@redhat.com>, 2000.

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
  int ret, file_flag = 0;

  switch (elf_header->e_machine)
    {
    case EM_X86_64:
      if (elf_header->e_ident[EI_CLASS] == ELFCLASS64)
	/* X86-64 64bit libraries are always libc.so.6+.  */
	file_flag = FLAG_X8664_LIB64|FLAG_ELF_LIBC6;
      else
	/* X32 libraries are always libc.so.6+.  */
	file_flag = FLAG_X8664_LIBX32|FLAG_ELF_LIBC6;
      break;
#ifndef __x86_64__
    case EM_IA_64:
      if (elf_header->e_ident[EI_CLASS] == ELFCLASS64)
	{
	  /* IA64 64bit libraries are always libc.so.6+.  */
	  file_flag = FLAG_IA64_LIB64|FLAG_ELF_LIBC6;
	  break;
	}
      goto failed;
#endif
    case EM_386:
      if (elf_header->e_ident[EI_CLASS] == ELFCLASS32)
	break;
      /* Fall through.  */
    default:
#ifndef __x86_64__
failed:
#endif
      error (0, 0, _("%s is for unknown machine %d.\n"),
	     file_name, elf_header->e_machine);
      return 1;
    }

  if (elf_header->e_ident[EI_CLASS] == ELFCLASS32)
    ret = process_elf32_file (file_name, lib, flag, osversion, isa_level,
			      soname, file_contents, file_length);
  else
    ret = process_elf64_file (file_name, lib, flag, osversion, isa_level,
			      soname, file_contents, file_length);

  if (!ret && file_flag)
    *flag = file_flag;

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
