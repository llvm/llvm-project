/* Print kernel diagnostics data in ld.so.  Linux version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <dl-diagnostics.h>
#include <ldsodefs.h>
#include <sys/utsname.h>

/* Dump the auxiliary vector to standard output.  */
static void
print_auxv (void)
{
  /* See _dl_show_auxv.  The code below follows the general output
     format for diagnostic dumps.  */
  unsigned int index = 0;
  for (ElfW(auxv_t) *av = GLRO(dl_auxv); av->a_type != AT_NULL; ++av)
    {
      _dl_printf ("auxv[0x%x].a_type=0x%lx\n"
                  "auxv[0x%x].a_val=",
                  index, (unsigned long int) av->a_type, index);
      if (av->a_type == AT_EXECFN
          || av->a_type == AT_PLATFORM
          || av->a_type == AT_BASE_PLATFORM)
        /* The address of the strings is not useful at all, so print
           the strings themselvs.  */
        _dl_diagnostics_print_string ((const char *) av->a_un.a_val);
      else
        _dl_printf ("0x%lx", (unsigned long int) av->a_un.a_val);
      _dl_printf ("\n");
      ++index;
    }
}

/* Print one uname entry.  */
static void
print_utsname_entry (const char *field, const char *value)
{
  _dl_printf ("uname.");
  _dl_diagnostics_print_labeled_string (field, value);
}

/* Print information from uname, including the kernel version.  */
static void
print_uname (void)
{
  struct utsname uts;
  if (__uname (&uts) == 0)
    {
      print_utsname_entry ("sysname", uts.sysname);
      print_utsname_entry ("nodename", uts.nodename);
      print_utsname_entry ("release", uts.release);
      print_utsname_entry ("version", uts.version);
      print_utsname_entry ("machine", uts.machine);
      print_utsname_entry ("domainname", uts.domainname);
    }
}

void
_dl_diagnostics_kernel (void)
{
  print_auxv ();
  print_uname ();
}
