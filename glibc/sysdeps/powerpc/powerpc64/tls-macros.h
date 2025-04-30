/* Include sysdeps/powerpc/tls-macros.h for __TLS_CALL_CLOBBERS  */
#include_next "tls-macros.h"

/* PowerPC64 Local Exec TLS access.  */
#define TLS_LE(x)							      \
  ({ int * __result;							      \
     asm ("addis %0,13," #x "@tprel@ha\n\t"				      \
	  "addi  %0,%0," #x "@tprel@l"					      \
	  : "=b" (__result) );						      \
     __result;								      \
  })
/* PowerPC64 Initial Exec TLS access.  */
#define TLS_IE(x)							      \
  ({ int * __result;							      \
     asm ("ld  %0," #x "@got@tprel(2)\n\t"				      \
	  "add %0,%0," #x "@tls"					      \
	  : "=r" (__result) );						      \
     __result;								      \
  })

/* PowerPC64 Local Dynamic TLS access.  */
#define TLS_LD(x)							      \
  ({ int * __result;							      \
     asm ("addi  3,2," #x "@got@tlsld\n\t"				      \
	  "bl    __tls_get_addr\n\t"					      \
	  "nop   \n\t"							      \
	  "addis %0,3," #x "@dtprel@ha\n\t"				      \
	  "addi  %0,%0," #x "@dtprel@l"					      \
	  : "=b" (__result) :						      \
	  : "3", __TLS_CALL_CLOBBERS);					      \
     __result;								      \
  })
/* PowerPC64 General Dynamic TLS access.  */
#define TLS_GD(x)							      \
  ({ register int *__result __asm__ ("r3");				      \
     asm ("addi  3,2," #x "@got@tlsgd\n\t"				      \
	  "bl    __tls_get_addr\n\t"					      \
	  "nop   "							      \
	  : "=r" (__result) :						      \
	  : __TLS_CALL_CLOBBERS);					      \
     __result;								      \
  })
