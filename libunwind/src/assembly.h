/* ===-- assembly.h - libUnwind assembler support macros -------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file defines macros for use in libUnwind assembler source.
 * This file is not part of the interface of this library.
 *
 * ===----------------------------------------------------------------------===
 */

#ifndef UNWIND_ASSEMBLY_H
#define UNWIND_ASSEMBLY_H

#if defined(__POWERPC__) || defined(__powerpc__) || defined(__ppc__)
#define SEPARATOR @
#elif defined(__arm64__)
#define SEPARATOR %%
#else
#define SEPARATOR ;
#endif

#if defined(__APPLE__)
#define HIDDEN_DIRECTIVE .private_extern
#else
#define HIDDEN_DIRECTIVE .hidden
#endif

#define GLUE2(a, b) a ## b
#define GLUE(a, b) GLUE2(a, b)
#define SYMBOL_NAME(name) GLUE(__USER_LABEL_PREFIX__, name)

#if defined(__APPLE__)

#define SYMBOL_IS_FUNC(name)
#define NO_EXEC_STACK_DIRECTIVE

#elif defined(__ELF__)

#if defined(__arm__)
#define SYMBOL_IS_FUNC(name) .type name,%function
#else
#define SYMBOL_IS_FUNC(name) .type name,@function
#endif

#if defined(__GNU__) || defined(__ANDROID__) || defined(__FreeBSD__)
#define NO_EXEC_STACK_DIRECTIVE .section .note.GNU-stack,"",%progbits
#else
#define NO_EXEC_STACK_DIRECTIVE
#endif

#else

#define SYMBOL_IS_FUNC(name)                                                   \
  .def name SEPARATOR                                                          \
    .scl 2 SEPARATOR                                                           \
    .type 32 SEPARATOR                                                         \
  .endef

#define NO_EXEC_STACK_DIRECTIVE

#endif

#define DEFINE_LIBUNWIND_FUNCTION(name)                   \
  .globl SYMBOL_NAME(name) SEPARATOR                      \
  SYMBOL_IS_FUNC(SYMBOL_NAME(name)) SEPARATOR             \
  SYMBOL_NAME(name):

#define DEFINE_LIBUNWIND_PRIVATE_FUNCTION(name)           \
  .globl SYMBOL_NAME(name) SEPARATOR                      \
  HIDDEN_DIRECTIVE SYMBOL_NAME(name) SEPARATOR            \
  SYMBOL_IS_FUNC(SYMBOL_NAME(name)) SEPARATOR             \
  SYMBOL_NAME(name):

#if defined(__arm__)
#if !defined(__ARM_ARCH)
#define __ARM_ARCH 4
#endif

#if defined(__ARM_ARCH_4T__) || __ARM_ARCH >= 5
#define ARM_HAS_BX
#endif

#ifdef ARM_HAS_BX
#define JMP(r) bx r
#else
#define JMP(r) mov pc, r
#endif
#endif /* __arm__ */

#endif /* UNWIND_ASSEMBLY_H */
