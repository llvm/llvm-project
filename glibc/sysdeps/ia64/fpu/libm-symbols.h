#ifndef _ISOMAC
# include <sysdep.h>
# undef ret	/* get rid of the stupid "ret" macro; it breaks br.ret */

# include <libm-alias-float.h>
# include <libm-alias-double.h>
# include <libm-alias-ldouble.h>

/* Support for compatible assembler handling.  */

# define ASM_SIZE_DIRECTIVE(name) .size name,.-name

# define LOCAL_LIBM_ENTRY(name)			\
	.proc name;				\
 name:

# define LOCAL_LIBM_END(name)			\
	.endp name;				\
 ASM_SIZE_DIRECTIVE(name)


# define RODATA		.rodata
# define LOCAL_OBJECT_START(name)		\
   name:;					\
   .type name, @object
# define LOCAL_OBJECT_END(name)			\
   ASM_SIZE_DIRECTIVE(name)

# define GLOBAL_LIBM_ENTRY(name)		\
	LOCAL_LIBM_ENTRY(name);			\
	.global name
# define GLOBAL_LIBM_END(name)		LOCAL_LIBM_END(name)

# define INTERNAL_LIBM_ENTRY(name)		\
	GLOBAL_LIBM_ENTRY(__libm_##name);	\
	.global __libm_##name
# define INTERNAL_LIBM_END(name)	GLOBAL_LIBM_END(__libm_##name)

# define WEAK_LIBM_ENTRY(name)			\
	.align 32;				\
	LOCAL_LIBM_ENTRY(__##name);		\
	.global __##name;			\
 __##name:
# define WEAK_LIBM_END(name)			\
 weak_alias (__##name, name);			\
 .hidden __##name;				\
	LOCAL_LIBM_END(__##name);		\
 ASM_SIZE_DIRECTIVE(__##name);			\
 .type __##name, @function

# define GLOBAL_IEEE754_ENTRY(name)		\
	WEAK_LIBM_ENTRY(name);			\
	.global __ieee754_##name;		\
	.hidden __ieee754_##name;		\
 __ieee754_##name:
# define GLOBAL_IEEE754_END(name)			\
	WEAK_LIBM_END(name);				\
 ASM_SIZE_DIRECTIVE(__ieee754_##name);			\
 .type __ieee754_##name, @function

# if defined ASSEMBLER && IS_IN (libc)
#  define __libm_error_support	HIDDEN_JUMPTARGET(__libm_error_support)
# endif
#endif
