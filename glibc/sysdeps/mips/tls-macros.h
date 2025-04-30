/* Macros to support TLS testing in times of missing compiler support.  */

#include <sys/cdefs.h>
#include <sys/asm.h>
#include <sysdep.h>

#define __STRING2(X) __STRING(X)
#define ADDU __STRING2(PTR_ADDU)
#define ADDIU __STRING2(PTR_ADDIU)
#define LW __STRING2(PTR_L)

/* Load the GOT pointer, which may not be in $28 in a non-PIC
   (abicalls pic0) function.  */
#ifndef __PIC__
# if _MIPS_SIM != _ABI64
#  ifndef __mips16
#   define LOAD_GP "move %[tmp], $28\n\tla $28, __gnu_local_gp\n\t"
#  else
#   define LOAD_GP					\
           "li %[tmp], %%hi(__gnu_local_gp)\n\t"	\
           "sll %[tmp], 16\n\t"				\
           "addiu %[tmp], %%lo(__gnu_local_gp)\n\t"
#  endif
# else
#  define LOAD_GP "move %[tmp], $28\n\tdla $28, __gnu_local_gp\n\t"
# endif
# define UNLOAD_GP "\n\tmove $28, %[tmp]"
#else
/* MIPS16 (re)creates the GP value using PC-relative instructions.  */
# ifdef __mips16
#  define LOAD_GP					\
           "li %[tmp], %%hi(_gp_disp)\n\t"		\
           "addiu %0, $pc, %%lo(_gp_disp)\n\t"		\
           "sll %[tmp], 16\n\t"				\
           "addu %[tmp], %0\n\t"
# else
#  define LOAD_GP
# endif
# define UNLOAD_GP
#endif

# if __mips_isa_rev >= 2
#  define TLS_RDHWR "rdhwr\t%0,$29"
# else
#  define TLS_RDHWR 					\
	  ".set push\n\t.set mips32r2\n\t"		\
	  "rdhwr\t%0,$29\n\t.set pop"
#endif

#ifndef __mips16
# define TLS_GD(x)					\
  ({ void *__result, *__tmp;				\
     extern void *__tls_get_addr (void *);		\
     asm (LOAD_GP ADDIU " %0, $28, %%tlsgd(" #x ")"	\
	  UNLOAD_GP					\
	  : "=r" (__result), [tmp] "=&r" (__tmp));	\
     (int *)__tls_get_addr (__result); })
# define TLS_LD(x)					\
  ({ void *__result, *__tmp;				\
     extern void *__tls_get_addr (void *);		\
     asm (LOAD_GP ADDIU " %0, $28, %%tlsldm(" #x ")"	\
	  UNLOAD_GP					\
	  : "=r" (__result), [tmp] "=&r" (__tmp));	\
     __result = __tls_get_addr (__result);		\
     asm ("lui $3,%%dtprel_hi(" #x ")\n\t"		\
	  "addiu $3,$3,%%dtprel_lo(" #x ")\n\t"		\
	  ADDU " %0,%0,$3"				\
	  : "+r" (__result) : : "$3");			\
     __result; })
# define TLS_IE(x)					\
  ({ void *__result, *__tmp;				\
     asm (TLS_RDHWR					\
	  : "=v" (__result));				\
     asm (LOAD_GP LW " $3,%%gottprel(" #x ")($28)\n\t"	\
	  ADDU " %0,%0,$3"				\
	  UNLOAD_GP					\
	  : "+r" (__result), [tmp] "=&r" (__tmp)	\
	  : : "$3");					\
     __result; })
# define TLS_LE(x)					\
  ({ void *__result;					\
     asm (TLS_RDHWR					\
	  : "=v" (__result));				\
     asm ("lui $3,%%tprel_hi(" #x ")\n\t"		\
	  "addiu $3,$3,%%tprel_lo(" #x ")\n\t"		\
	  ADDU " %0,%0,$3"				\
	  : "+r" (__result) : : "$3");			\
     __result; })

#else /* __mips16 */
/* MIPS16 version.  */
# define TLS_GD(x)					\
  ({ void *__result, *__tmp;				\
     extern void *__tls_get_addr (void *);		\
     asm (LOAD_GP ADDIU " %1, %%tlsgd(" #x ")"		\
	  "\n\tmove %0, %1"				\
	  : "=d" (__result), [tmp] "=&d" (__tmp));	\
     (int *) __tls_get_addr (__result); })
# define TLS_LD(x)					\
  ({ void *__result, *__tmp;				\
     extern void *__tls_get_addr (void *);		\
     asm (LOAD_GP ADDIU " %1, %%tlsldm(" #x ")"		\
	  "\n\tmove %0, %1"				\
	  : "=d" (__result), [tmp] "=&d" (__tmp));	\
     __result = __tls_get_addr (__result);		\
     asm ("li $3,%%dtprel_hi(" #x ")\n\t"		\
	  "sll $3,16\n\t"				\
	  "addiu $3,%%dtprel_lo(" #x ")\n\t"		\
	  ADDU " %0,%0,$3"				\
	  : "+d" (__result) : : "$3");			\
     __result; })
# define TLS_IE(x)					\
  ({ void *__result, *__tmp, *__tp;			\
     __tp = __builtin_thread_pointer ();		\
     asm (LOAD_GP LW " $3,%%gottprel(" #x ")(%1)\n\t"	\
	  ADDU " %0,%[tp],$3"				\
	  : "=&d" (__result), [tmp] "=&d" (__tmp)	\
	  : [tp] "d" (__tp) : "$3");			\
     __result; })
# define TLS_LE(x)					\
  ({ void *__result, *__tp;				\
     __tp = __builtin_thread_pointer ();		\
     asm ("li $3,%%tprel_hi(" #x ")\n\t"		\
	  "sll $3,16\n\t"				\
	  "addiu $3,%%tprel_lo(" #x ")\n\t"		\
	  ADDU " %0,%[tp],$3"				\
	  : "=d" (__result) : [tp] "d" (__tp) : "$3");	\
     __result; })

#endif /* __mips16 */
