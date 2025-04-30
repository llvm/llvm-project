/* Macros to support TLS testing in times of missing compiler support.  */

extern void *__tls_get_addr (void *);

# define TLS_GD(x)							\
  ({ register void *__gp asm ("$29"); void *__result;			\
     asm ("lda %0, " #x "($gp) !tlsgd" : "=r" (__result) : "r"(__gp));	\
     __tls_get_addr (__result); })

# define TLS_LD(x)							\
  ({ register void *__gp asm ("$29"); void *__result;			\
     asm ("lda %0, " #x "($gp) !tlsldm" : "=r" (__result) : "r"(__gp));	\
     __result = __tls_get_addr (__result);				\
     asm ("lda %0, " #x "(%0) !dtprel" : "+r" (__result));		\
     __result; })

# define TLS_IE(x)							\
  ({ register void *__gp asm ("$29"); long ofs;				\
     asm ("ldq %0, " #x "($gp) !gottprel" : "=r"(ofs) : "r"(__gp));	\
     __builtin_thread_pointer () + ofs; })

# define TLS_LE(x)						\
  ({ void *__result = __builtin_thread_pointer ();		\
     asm ("lda %0, " #x "(%0) !tprel" : "+r" (__result));	\
     __result; })
