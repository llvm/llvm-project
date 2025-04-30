#define TLS_LE(x)					\
  ({ int *__result;					\
     asm ("addi %0, r23, %%tls_le(" #x ")"		\
	  : "=r" (__result));		                \
     __result; })

#define TLS_IE(x)					\
  ({ int *__result;					\
     int __tmp;                                         \
     asm ("nextpc %0 ; "                                \
          "1: movhi %1, %%hiadj(_gp_got - 1b) ; "	\
          "addi %1, %1, %%lo(_gp_got - 1b) ; "		\
          "add %0, %0, %1 ; "                           \
          "ldw %1, %%tls_ie(" #x ")(%0) ; "        	\
	  "add %1, r23, %1"              		\
          : "=&r" (__tmp), "=&r" (__result));           \
     __result; })

#define TLS_LD(x)					\
  ({ char *__result;					\
     char *__result2;                                   \
     int *__result3;                                    \
     int __tmp;                                         \
     extern void *__tls_get_addr (void *);		\
     asm ("nextpc %0 ; "                                \
          "1: movhi %1, %%hiadj(_gp_got - 1b) ; "	\
          "addi %1, %1, %%lo(_gp_got - 1b) ; "		\
          "add %0, %0, %1 ; "                           \
          "addi %0, %0, %%tls_ldm(" #x ")"              \
          : "=r" (__result), "=r" (__tmp));             \
     __result2 = (char *)__tls_get_addr (__result);	\
     asm ("addi %0, %1, %%tls_ldo(" #x ")"              \
	  : "=r" (__result3) : "r" (__result2));        \
     __result3; })

#define TLS_GD(x)					\
  ({ int *__result;					\
     int __tmp;                                         \
     extern void *__tls_get_addr (void *);		\
     asm ("nextpc %0 ; "                                \
          "1: movhi %1, %%hiadj(_gp_got - 1b) ; "	\
          "addi %1, %1, %%lo(_gp_got - 1b) ; "		\
          "add %0, %0, %1 ; "                           \
          "addi %0, %0, %%tls_gd(" #x ")"		\
	  : "=r" (__result), "=r" (__tmp));		\
     (int *)__tls_get_addr (__result); })
