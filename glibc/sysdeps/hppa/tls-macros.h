/* TLS Access Macros for HP PARISC Linux */

/* HPPA Local Exec TLS access.  */
#define TLS_LE(x) \
  ({  int * __result;  \
      unsigned long __tmp; \
      asm ( \
	"  mfctl %%cr27, %1\n" \
	"  addil LR'" #x "-$tls_leoff$, %1\n" \
	"  ldo RR'" #x "-$tls_leoff$(%%r1), %0\n" \
        : "=r" (__result), "=r" (__tmp) \
	: \
	: "r1" );  \
      __result;  \
  })

/* HPPA Initial Exec TLS access.  */
#ifdef PIC
#  define TLS_IE(x) \
  ({  int * __result;  \
      unsigned long __tmp, __tmp2; \
      asm ( \
	"  mfctl %%cr27, %1\n" \
	"  addil LT'" #x "-$tls_ieoff$, %%r19\n" \
	"  ldw RT'" #x "-$tls_ieoff$(%%r1), %2\n" \
	"  add %1, %2, %0\n" \
	: "=r" (__result), "=r" (__tmp), "=r" (__tmp2) \
	: \
	: "r1" ); \
      __result;  \
  })
#else
#  define TLS_IE(x) \
  ({  int * __result;  \
      unsigned long __tmp, __tmp2; \
      asm ( \
	"  mfctl %%cr27, %1\n" \
	"  addil LR'" #x "-$tls_ieoff$, %%r27\n" \
	"  ldw RR'" #x "-$tls_ieoff$(%%r1), %2\n" \
	"  add %1, %2, %0\n" \
	: "=r" (__result), "=r" (__tmp), "=r" (__tmp2) \
	: \
	: "r1" ); \
      __result;  \
  })
#endif

#ifdef PIC
/* HPPA Local Dynamic TLS access.  */
#  define TLS_LD(x) \
  ({  int * __result;  \
      asm (  \
	"  copy %%r19, %%r4\n" \
	"  addil LT'" #x "-$tls_ldidx$, %%r19\n" \
	"  bl __tls_get_addr, %%r2\n" \
	"  ldo RT'" #x "-$tls_ldidx$(%%r1), %%r26\n" \
	"  addil LR'" #x "-$tls_dtpoff$, %%r28\n" \
	"  ldo RR'" #x "-$tls_dtpoff$(%%r1), %0\n" \
	"  copy %%r4, %%r19\n" \
	: "=r" (__result) \
	: \
	: "r1", "r2", "r4", "r20", "r21", "r22", "r23", "r24", \
	  "r25", "r26", "r28", "r29", "r31" ); \
      __result;  \
  })
#else
#  define TLS_LD(x) \
  ({  int * __result;  \
      asm (  \
	"  addil LR'" #x "-$tls_ldidx$, %%r27\n" \
	"  bl __tls_get_addr, %%r2\n" \
	"  ldo RR'" #x "-$tls_ldidx$(%%r1), %%r26\n" \
	"  addil LR'" #x "-$tls_dtpoff$, %%r28\n" \
	"  ldo RR'" #x "-$tls_dtpoff$(%%r1), %0\n" \
	: "=r" (__result) \
	: \
	: "r1", "r2", "r20", "r21", "r22", "r23", "r24", \
	  "r25", "r26", "r28", "r29", "r31" ); \
      __result;  \
  })
#endif

/* HPPA General Dynamic TLS access.  */
#ifdef PIC
#  define TLS_GD(x) \
  ({  int * __result;  \
      asm (  \
	"  copy %%r19, %%r4\n" \
        "  addil LT'" #x "-$tls_gdidx$, %%r19\n" \
	"  bl __tls_get_addr, %%r2\n" \
	"  ldo RT'" #x "-$tls_gdidx$(%%r1), %%r26\n" \
	"  copy %%r28, %0\n" \
	"  copy %%r4, %%r19\n" \
	: "=r" (__result) \
	: \
	: "r1", "r2", "r4", "r20", "r21", "r22", "r23", "r24", \
	  "r25", "r26", "r28", "r29", "r31" ); \
      __result;  \
  })
#else
#  define TLS_GD(x) \
  ({  int * __result;  \
      asm (  \
        "  addil LR'" #x "-$tls_gdidx$, %%r27\n" \
	"  bl __tls_get_addr, %%r2\n" \
	"  ldo RR'" #x "-$tls_gdidx$(%%r1), %%r26\n" \
	"  copy %%r28, %0\n" \
	: "=r" (__result) \
	: \
	: "r1", "r2", "r20", "r21", "r22", "r23", "r24", \
	  "r25", "r26", "r28", "r29", "r31" ); \
      __result;  \
  })
#endif
