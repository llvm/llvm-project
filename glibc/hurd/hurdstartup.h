/* Data from initial program startup for running under the GNU Hurd.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#ifndef _HURDSTARTUP_H
#define _HURDSTARTUP_H 1

# include <stdint.h>

/* Interesting data saved from the exec_startup reply.
   The DATA argument to *MAIN (see below) points to:

    int argc;
    char *argv[argc];
    char *argv_terminator = NULL;
    char *envp[?];
    char *envp_terminator = NULL;
    struct hurd_startup_data data;

*/

struct hurd_startup_data
  {
    int flags;
    mach_port_t *dtable;
    mach_msg_type_number_t dtablesize;
    mach_port_t *portarray;
    mach_msg_type_number_t portarraysize;
    int *intarray;
    mach_msg_type_number_t intarraysize;
    vm_address_t stack_base;
    vm_size_t stack_size;
    vm_address_t phdr;
    vm_size_t phdrsz;
    vm_address_t user_entry;
  };


/* Initialize Mach RPCs; do initial handshake with the exec server (or
   extract the arguments from the stack in the case of the bootstrap task);
   finally, call *MAIN with the information gleaned.  That function is not
   expected to return.  ARGPTR should be the address of the first argument
   of the entry point function that is called with the stack exactly as the
   exec server or kernel sets it.  */

extern void _hurd_startup (void **argptr, void (*main) (intptr_t *data));


#endif	/* hurdstartup.h */
