/* Machine-dependent ELF startup code.  PowerPC version.
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

#include <sysdep.h>

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.	*/
ENTRY(_start)
/* We start with the following on the stack, from top:
   argc (4 bytes);
   arguments for program (terminated by NULL);
   environment variables (terminated by NULL);
   arguments for the program loader. */

/* Call _dl_start with one parameter pointing at argc */
	mr	r3,r1
/* (we have to frob the stack pointer a bit to allow room for
   _dl_start to save the link register).  */
	li	r4,0
	addi	r1,r1,-16
	cfi_adjust_cfa_offset (16)
/* Mark lr as undefined to stop unwinding.  */
	cfi_undefined (lr)
	stw	r4,0(r1)
	bl	_dl_start@local

	/* FALLTHRU */
_dl_start_user:
/* Now, we do our main work of calling initialisation procedures.
   The ELF ABI doesn't say anything about parameters for these,
   so we just pass argc, argv, and the environment.
   Changing these is strongly discouraged (not least because argc is
   passed by value!).  */

/*  Put our GOT pointer in r31, */
	SETUP_GOT_ACCESS(r31,got_label)
	addis	r31,r31,_GLOBAL_OFFSET_TABLE_-got_label@ha
	addi	r31,r31,_GLOBAL_OFFSET_TABLE_-got_label@l
/*  the address of _start in r30, */
	mr	r30,r3
/*  &_dl_argc in 29, &_dl_argv in 27, and _dl_loaded in 28.  */
	lwz	r28,_rtld_local@got(r31)
	lwz	r29,_dl_argc@got(r31)
	lwz	r27,__GI__dl_argv@got(r31)

/* Call _dl_init (_dl_loaded, _dl_argc, _dl_argv, _dl_argv+_dl_argc+1). */
	lwz	r3,0(r28)
	lwz	r4,0(r29)
	lwz	r5,0(r27)
	slwi	r6,r4,2
	add	r6,r5,r6
	addi	r6,r6,4
	bl	_dl_init@local

/* Now, to conform to the ELF ABI, we have to: */
/* Pass argc (actually _dl_argc) in r3; */
	lwz	r3,0(r29)
/* pass argv (actually _dl_argv) in r4; */
	lwz	r4,0(r27)
/* pass envp (actually _dl_argv+_dl_argc+1) in r5; */
	slwi	r5,r3,2
	add	r6,r4,r5
	addi	r5,r6,4
/* pass the auxiliary vector in r6. This is passed to us just after _envp.  */
2:	lwzu	r0,4(r6)
	cmpwi	r0,0
	bne	2b
	addi	r6,r6,4
/* Pass a termination function pointer (in this case _dl_fini) in r7.  */
	lwz	r7,_dl_fini@got(r31)
/* Now, call the start function in r30... */
	mtctr	r30
/* Pass the stack pointer in r1 (so far so good), pointing to a NULL value.
   (This lets our startup code distinguish between a program linked statically,
   which linux will call with argc on top of the stack which will hopefully
   never be zero, and a dynamically linked program which will always have
   a NULL on the top of the stack).
   Take the opportunity to clear LR, so anyone who accidentally returns
   from _start gets SEGV.  Also clear the next few words of the stack.  */

_dl_main_dispatch:
	li	r31,0
	stw	r31,0(r1)
	mtlr	r31
	stw	r31,4(r1)
	stw	r31,8(r1)
	stw	r31,12(r1)
/* Go do it!  */
	bctr
END(_start)
