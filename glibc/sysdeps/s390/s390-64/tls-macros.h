#define TLS_LE(x) \
  ({ unsigned long __offset;						      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.quad " #x "@ntpoff\n"				      \
	      "1:\tlg %0,0(%0)"						      \
	      : "=a" (__offset) : : "cc" );				      \
     (int *) (__builtin_thread_pointer() + __offset); })

#ifdef PIC
# define TLS_IE(x) \
  ({ unsigned long __offset, __save12;					      \
     __asm__ ("bras %0,0f\n\t"						      \
	      ".quad " #x "@gotntpoff\n"				      \
	      "0:\tlgr %1,%%r12\n\t"					      \
	      "larl %%r12,_GLOBAL_OFFSET_TABLE_\n\t"			      \
	      "lg %0,0(%0)\n\t"						      \
	      "lg %0,0(%0,%%r12):tls_load:" #x	"\n\t"			      \
	      "lgr %%r12,%1\n"						      \
	      : "=&a" (__offset), "=&a" (__save12) : : "cc" );		      \
     (int *) (__builtin_thread_pointer() + __offset); })
#else
# define TLS_IE(x) \
  ({ unsigned long  __offset;						      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.quad " #x "@indntpoff\n"				      \
	      "1:\t lg %0,0(%0)\n\t"					      \
	      "lg %0,0(%0):tls_load:" #x				      \
	      : "=&a" (__offset) : : "cc" );				      \
     (int *) (__builtin_thread_pointer() + __offset); })
#endif

#ifdef PIC
# define TLS_LD(x) \
  ({ unsigned long __offset, __save12;					      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.quad " #x "@tlsldm\n\t"				      \
	      ".quad " #x "@dtpoff\n"					      \
	      "1:\tlgr %1,%%r12\n\t"					      \
	      "larl %%r12,_GLOBAL_OFFSET_TABLE_\n\t"			      \
	      "lg %%r2,0(%0)\n\t"					      \
	      "brasl %%r14,__tls_get_offset@plt:tls_ldcall:" #x "\n\t"	      \
	      "lg %0,8(%0)\n\t"						      \
	      "algr %0,%%r2\n\t"					      \
	      "lgr %%r12,%1"						      \
	      : "=&a" (__offset), "=&a" (__save12)			      \
	      : : "cc", "0", "1", "2", "3", "4", "5", "14" );		      \
     (int *) (__builtin_thread_pointer() + __offset); })
#else
# define TLS_LD(x) \
  ({ unsigned long __offset;						      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.quad " #x "@tlsldm\n\t"				      \
	      ".quad " #x "@dtpoff\n"					      \
	      "1:\tlarl %%r12,_GLOBAL_OFFSET_TABLE_\n\t"		      \
	      "lg %%r2,0(%0)\n\t"					      \
	      "brasl %%r14,__tls_get_offset@plt:tls_ldcall:" #x "\n\t"	      \
	      "lg %0,8(%0)\n\t"						      \
	      "algr %0,%%r2"						      \
	      : "=&a" (__offset)					      \
	      : : "cc", "0", "1", "2", "3", "4", "5", "12", "14" );	      \
     (int *) (__builtin_thread_pointer() + __offset); })
#endif

#ifdef PIC
# define TLS_GD(x) \
  ({ unsigned long __offset, __save12;					      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.quad " #x "@tlsgd\n"				      \
	      "1:\tlgr %1,%%r12\n\t"					      \
	      "larl %%r12,_GLOBAL_OFFSET_TABLE_\n\t"			      \
	      "lg %%r2,0(%0)\n\t"					      \
	      "brasl %%r14,__tls_get_offset@plt:tls_gdcall:" #x "\n\t"	      \
	      "lgr %0,%%r2\n\t"						      \
	      "lgr %%r12,%1"						      \
	      : "=&a" (__offset), "=&a" (__save12)			      \
	      : : "cc", "0", "1", "2", "3", "4", "5", "14" );		      \
     (int *) (__builtin_thread_pointer() + __offset); })
#else
# define TLS_GD(x) \
  ({ unsigned long __offset;						      \
     __asm__ ("bras %0,1f\n"						      \
	      "0:\t.quad " #x "@tlsgd\n"				      \
	      "1:\tlarl %%r12,_GLOBAL_OFFSET_TABLE_\n\t"		      \
	      "lg %%r2,0(%0)\n\t"					      \
	      "brasl %%r14,__tls_get_offset@plt:tls_gdcall:" #x "\n\t"	      \
	      "lgr %0,%%r2"						      \
	      : "=&a" (__offset)					      \
	      : : "cc", "0", "1", "2", "3", "4", "5", "12", "14" );	      \
     (int *) (__builtin_thread_pointer() + __offset); })
#endif
