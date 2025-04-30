#include <sysdep.h>                     /* For ARCH_HAS_T2.  */

#ifdef __thumb2__
# define ARM_PC_OFFSET "4"
#else
# define ARM_PC_OFFSET "8"
#endif

/* Returns the address of data containing ".word SYMBOL(RELOC)".  */
#if defined (ARCH_HAS_T2) && !defined (PIC)
# define GET_SPECIAL_RELOC(symbol, reloc)			\
  ({								\
    int *__##symbol##_rodata;					\
    asm ("movw %0, #:lower16:1f\n"				\
	 "movt %0, #:upper16:1f\n"				\
	 ".pushsection .rodata.cst4, \"aM\", %%progbits, 4\n"	\
	 ".balign 4\n"						\
	 "1: .word " #symbol "(" #reloc ")\n"			\
	 ".popsection"						\
	 : "=r" (__##symbol##_rodata));				\
    __##symbol##_rodata;					\
  })
#elif defined (ARCH_HAS_T2) && defined (PIC) && ARM_PCREL_MOVW_OK
# define GET_SPECIAL_RELOC(symbol, reloc)			\
  ({								\
    int *__##symbol##_rodata;					\
    asm ("movw %0, #:lower16:1f - 2f - " ARM_PC_OFFSET "\n"	\
	 "movt %0, #:upper16:1f - 2f - " ARM_PC_OFFSET "\n"	\
	 ".pushsection .rodata.cst4, \"aM\", %%progbits, 4\n"	\
	 ".balign 4\n"						\
	 "1: .word " #symbol "(" #reloc ")\n"			\
	 ".popsection\n"					\
	 "2: add %0, %0, pc"					\
	 : "=r" (__##symbol##_rodata));				\
    __##symbol##_rodata;					\
  })
#else
# define GET_SPECIAL_RELOC(symbol, reloc)			\
  ({								\
    int *__##symbol##_rodata;					\
    asm ("adr %0, 1f\n"						\
	 "b 2f\n"						\
	 ".balign 4\n"						\
	 "1: .word " #symbol "(" #reloc ")\n"			\
	 "2:"							\
	 : "=r" (__##symbol##_rodata));				\
    __##symbol##_rodata;					\
  })
#endif

/* Returns the pointer value (SYMBOL(RELOC) + pc - PC_OFS).  */
#define GET_SPECIAL_PCREL(symbol, reloc)				\
  ({									\
    int *__##symbol##_rodata = GET_SPECIAL_RELOC (symbol, reloc);	\
    (void *) ((int) __##symbol##_rodata + *__##symbol##_rodata);	\
  })

#define TLS_LE(x)						\
  (__builtin_thread_pointer () + *GET_SPECIAL_RELOC (x, tpoff))

#define TLS_IE(x)						\
  ((int *) (__builtin_thread_pointer ()				\
	    + *(int *) GET_SPECIAL_PCREL (x, gottpoff)))

extern void *__tls_get_addr (void *);

#define TLS_LD(x)						\
  ((int *) (__tls_get_addr (GET_SPECIAL_PCREL (x, tlsldm))	\
	    + *GET_SPECIAL_RELOC (x, tlsldo)))

#define TLS_GD(x)						\
  ((int *) __tls_get_addr (GET_SPECIAL_PCREL (x, tlsgd)))
