#define TLS_LE(x) \
  ({ int *__l;								      \
     asm ("movl %%gs:0,%0\n\t"						      \
	  "subl $" #x "@tpoff,%0"					      \
	  : "=r" (__l));						      \
     __l; })

#define TLS_IE(x) \
  ({ int *__l, __b;							      \
     asm ("call 1f\n\t"							      \
	  ".subsection 1\n"						      \
	  "1:\tmovl (%%esp), %%ebx\n\t"					      \
	  "ret\n\t"							      \
	  ".previous\n\t"						      \
	  "addl $_GLOBAL_OFFSET_TABLE_, %%ebx\n\t"			      \
	  "movl %%gs:0,%0\n\t"						      \
	  "subl " #x "@gottpoff(%%ebx),%0"				      \
	  : "=r" (__l), "=&b" (__b));					      \
     __l; })

#define TLS_LD(x) \
  ({ int *__l, __b, __c, __d;						      \
     asm ("call 1f\n\t"							      \
	  ".subsection 1\n"						      \
	  "1:\tmovl (%%esp), %%ebx\n\t"					      \
	  "ret\n\t"							      \
	  ".previous\n\t"						      \
	  "addl $_GLOBAL_OFFSET_TABLE_, %%ebx\n\t"			      \
	  "leal " #x "@tlsldm(%%ebx),%%eax\n\t"				      \
	  "call ___tls_get_addr@plt\n\t"				      \
	  "leal " #x "@dtpoff(%%eax), %%eax"				      \
	  : "=a" (__l), "=&b" (__b), "=&c" (__c), "=&d" (__d));		      \
     __l; })

#define TLS_GD(x) \
  ({ int *__l, __b, __c, __d;						      \
     asm ("call 1f\n\t"							      \
	  ".subsection 1\n"						      \
	  "1:\tmovl (%%esp), %%ebx\n\t"					      \
	  "ret\n\t"							      \
	  ".previous\n\t"						      \
	  "addl $_GLOBAL_OFFSET_TABLE_, %%ebx\n\t"			      \
	  "leal " #x "@tlsgd(%%ebx),%%eax\n\t"				      \
	  "call ___tls_get_addr@plt\n\t"				      \
	  "nop"								      \
	  : "=a" (__l), "=&b" (__b), "=&c" (__c), "=&d" (__d));		      \
     __l; })
