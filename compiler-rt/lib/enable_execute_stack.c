/* ===-- enable_execute_stack.c - Implement __enable_execute_stack ---------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 */

#include <stdint.h>
#include <sys/mman.h>

/* #include "config.h"
 * FIXME: CMake - include when cmake system is ready.
 * Remove #define HAVE_SYSCONF 1 line.
 */
#define HAVE_SYSCONF 1

#ifndef __APPLE__
#include <unistd.h>
#endif /* __APPLE__ */

#if __LP64__
	#define TRAMPOLINE_SIZE 48
#else
	#define TRAMPOLINE_SIZE 40
#endif

/*
 * The compiler generates calls to __enable_execute_stack() when creating 
 * trampoline functions on the stack for use with nested functions.
 * It is expected to mark the page(s) containing the address 
 * and the next 48 bytes as executable.  Since the stack is normally rw-
 * that means changing the protection on those page(s) to rwx. 
 */

void __enable_execute_stack(void* addr)
{

#if __APPLE__
	/* On Darwin, pagesize is always 4096 bytes */
	const uintptr_t pageSize = 4096;
#elif !defined(HAVE_SYSCONF)
#error "HAVE_SYSCONF not defined! See enable_execute_stack.c"
#else
        const uintptr_t pageSize = sysconf(_SC_PAGESIZE);
#endif /* __APPLE__ */

	const uintptr_t pageAlignMask = ~(pageSize-1);
	uintptr_t p = (uintptr_t)addr;
	unsigned char* startPage = (unsigned char*)(p & pageAlignMask);
	unsigned char* endPage = (unsigned char*)((p+TRAMPOLINE_SIZE+pageSize) & pageAlignMask);
	size_t length = endPage - startPage;
	(void) mprotect((void *)startPage, length, PROT_READ | PROT_WRITE | PROT_EXEC);
}


