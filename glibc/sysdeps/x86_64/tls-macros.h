#define TLS_LE(x) \
  ({ int *__l;								      \
     asm ("mov %%fs:0,%0\n\t"						      \
	  "lea " #x "@tpoff(%0), %0"					      \
	  : "=r" (__l));						      \
     __l; })

#define TLS_IE(x) \
  ({ int *__l;								      \
     asm ("mov %%fs:0,%0\n\t"						      \
	  "add " #x "@gottpoff(%%rip),%0"				      \
	  : "=r" (__l));						      \
     __l; })

#define TLS_LD(x) \
  ({ int *__l, __c, __d;						      \
     asm ("leaq " #x "@tlsld(%%rip),%%rdi\n\t"				      \
	  "call __tls_get_addr@plt\n\t"					      \
	  "leaq " #x "@dtpoff(%%rax), %%rax"				      \
	  : "=a" (__l), "=&c" (__c), "=&d" (__d)			      \
	  : : "rdi", "rsi", "r8", "r9", "r10", "r11"); 			      \
     __l; })

#ifdef __ILP32__
# define TLS_GD_PREFIX
#else
# define TLS_GD_PREFIX	".byte 0x66\n\t"
#endif

#define TLS_GD(x) \
  ({ int *__l, __c, __d;						      \
     asm (TLS_GD_PREFIX							      \
	  "leaq " #x "@tlsgd(%%rip),%%rdi\n\t"				      \
	  ".word 0x6666\n\t"						      \
	  "rex64\n\t"							      \
	  "call __tls_get_addr@plt"					      \
	  : "=a" (__l), "=&c" (__c), "=&d" (__d)			      \
	  : : "rdi", "rsi", "r8", "r9", "r10", "r11"); 			      \
     __l; })
