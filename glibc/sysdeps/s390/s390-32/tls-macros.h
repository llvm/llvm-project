#define TLS_LE(x) \
  ({ unsigned long __offset;						      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.long " #x "@ntpoff\n"				      \
	      "1:\tl %0,0(%0)"						      \
	      : "=a" (__offset) : : "cc" );				      \
     (int *) (__builtin_thread_pointer() + __offset); })

#ifdef PIC
# define TLS_IE(x) \
  ({ unsigned long __offset, __save12;					      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.long _GLOBAL_OFFSET_TABLE_-0b\n\t"			      \
	      ".long " #x "@gotntpoff\n"				      \
	      "1:\tlr %1,%%r12\n\t"					      \
	      "l %%r12,0(%0)\n\t"					      \
	      "la %%r12,0(%0,%%r12)\n\t"				      \
	      "l %0,4(%0)\n\t"						      \
	      "l %0,0(%0,%%r12):tls_load:" #x "\n\t"			      \
	      "lr %%r12,%1\n"						      \
	      : "=&a" (__offset), "=&a" (__save12) : : "cc" );		      \
     (int *) (__builtin_thread_pointer() + __offset); })
#else
# define TLS_IE(x) \
  ({ unsigned long  __offset;						      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.long " #x "@indntpoff\n"				      \
	      "1:\t l %0,0(%0)\n\t"					      \
	      "l %0,0(%0):tls_load:" #x					      \
	      : "=&a" (__offset) : : "cc" );				      \
     (int *) (__builtin_thread_pointer() + __offset); })
#endif

#ifdef PIC
# define TLS_LD(x) \
  ({ unsigned long __offset, __save12;					      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.long _GLOBAL_OFFSET_TABLE_-0b\n\t"			      \
	      ".long __tls_get_offset@plt-0b\n\t"			      \
	      ".long " #x "@tlsldm\n\t"					      \
	      ".long " #x "@dtpoff\n"					      \
	      "1:\tlr %1,%%r12\n\t"					      \
	      "l %%r12,0(%0)\n\t"					      \
	      "la %%r12,0(%%r12,%0)\n\t"				      \
	      "l %%r1,4(%0)\n\t"					      \
	      "l %%r2,8(%0)\n\t"					      \
	      "bas %%r14,0(%%r1,%0):tls_ldcall:" #x "\n\t"		      \
	      "l %0,12(%0)\n\t"						      \
	      "alr %0,%%r2\n\t"						      \
	      "lr %%r12,%1"						      \
	      : "=&a" (__offset), "=&a" (__save12)			      \
	      : : "cc", "0", "1", "2", "3", "4", "5", "14");		      \
     (int *) (__builtin_thread_pointer() + __offset); })
#else
# define TLS_LD(x) \
  ({ unsigned long __offset;						      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.long _GLOBAL_OFFSET_TABLE_\n\t"			      \
	      ".long __tls_get_offset@plt\n\t"				      \
	      ".long " #x "@tlsldm\n\t"					      \
	      ".long " #x "@dtpoff\n"					      \
	      "1:\tl %%r12,0(%0)\n\t"					      \
	      "l %%r1,4(%0)\n\t"					      \
	      "l %%r2,8(%0)\n\t"					      \
	      "bas %%r14,0(%%r1):tls_ldcall:" #x "\n\t"			      \
	      "l %0,12(%0)\n\t"						      \
	      "alr %0,%%r2"						      \
	      : "=&a" (__offset)					      \
	      : : "cc", "0", "1", "2", "3", "4", "5", "12", "14");	      \
     (int *) (__builtin_thread_pointer() + __offset); })
#endif

#ifdef PIC
# define TLS_GD(x) \
  ({ unsigned long __offset, __save12;					      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.long _GLOBAL_OFFSET_TABLE_-0b\n\t"			      \
	      ".long __tls_get_offset@plt-0b\n\t"			      \
	      ".long " #x "@tlsgd\n"					      \
	      "1:\tlr %1,%%r12\n\t"					      \
	      "l %%r12,0(%0)\n\t"					      \
	      "la %%r12,0(%%r12,%0)\n\t"				      \
	      "l %%r1,4(%0)\n\t"					      \
	      "l %%r2,8(%0)\n\t"					      \
	      "bas %%r14,0(%%r1,%0):tls_gdcall:" #x "\n\t"		      \
	      "lr %0,%%r2\n\t"						      \
	      "lr %%r12,%1"						      \
	      : "=&a" (__offset), "=&a" (__save12)			      \
	      : : "cc", "0", "1", "2", "3", "4", "5", "14");		      \
     (int *) (__builtin_thread_pointer() + __offset); })
#else
# define TLS_GD(x) \
  ({ unsigned long __offset;						      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.long _GLOBAL_OFFSET_TABLE_\n\t"			      \
	      ".long __tls_get_offset@plt\n\t"				      \
	      ".long " #x "@tlsgd\n"					      \
	      "1:\tl %%r12,0(%0)\n\t"					      \
	      "l %%r1,4(%0)\n\t"					      \
	      "l %%r2,8(%0)\n\t"					      \
	      "bas %%r14,0(%%r1):tls_gdcall:" #x "\n\t"			      \
	      "lr %0,%%r2"						      \
	      : "=&a" (__offset)					      \
	      : : "cc", "0", "1", "2", "3", "4", "5", "12", "14");	      \
     (int *) (__builtin_thread_pointer() + __offset); })
#endif
