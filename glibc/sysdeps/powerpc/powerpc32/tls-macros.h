/* Include sysdeps/powerpc/tls-macros.h for __TLS_CALL_CLOBBERS  */
#include_next "tls-macros.h"

/* PowerPC32 Local Exec TLS access.  */
#define TLS_LE(x)							      \
  ({ int *__result;							      \
     asm ("addi %0,2," #x "@tprel"					      \
	  : "=r" (__result));						      \
     __result; })

/* PowerPC32 Initial Exec TLS access.  */
#define TLS_IE(x)							      \
  ({ int *__result;							      \
     asm ("bcl 20,31,1f\n1:\t"						      \
	  "mflr %0\n\t"							      \
	  "addis %0,%0,_GLOBAL_OFFSET_TABLE_-1b@ha\n\t"			      \
	  "addi %0,%0,_GLOBAL_OFFSET_TABLE_-1b@l\n\t"			      \
	  "lwz %0," #x "@got@tprel(%0)\n\t"				      \
	  "add %0,%0," #x "@tls"					      \
	  : "=b" (__result) :						      \
	  : "lr");							      \
     __result; })

/* PowerPC32 Local Dynamic TLS access.  */
#define TLS_LD(x)							      \
  ({ int *__result;							      \
     asm ("bcl 20,31,1f\n1:\t"						      \
	  "mflr 3\n\t"							      \
	  "addis 3,3,_GLOBAL_OFFSET_TABLE_-1b@ha\n\t"			      \
	  "addi 3,3,_GLOBAL_OFFSET_TABLE_-1b@l\n\t"			      \
	  "addi 3,3," #x "@got@tlsld\n\t"				      \
	  "bl __tls_get_addr@plt\n\t"					      \
	  "addi %0,3," #x "@dtprel"					      \
	  : "=r" (__result) :						      \
	  : "3", __TLS_CALL_CLOBBERS);					      \
     __result; })

/* PowerPC32 General Dynamic TLS access.  */
#define TLS_GD(x)							      \
  ({ register int *__result __asm__ ("r3");				      \
     asm ("bcl 20,31,1f\n1:\t"						      \
	  "mflr 3\n\t"							      \
	  "addis 3,3,_GLOBAL_OFFSET_TABLE_-1b@ha\n\t"			      \
	  "addi 3,3,_GLOBAL_OFFSET_TABLE_-1b@l\n\t"			      \
	  "addi 3,3," #x "@got@tlsgd\n\t"				      \
	  "bl __tls_get_addr@plt"					      \
	  : "=r" (__result) :						      \
	  : __TLS_CALL_CLOBBERS);					      \
     __result; })
