#define TLS_LE(x) \
  ({ int *__l; void *__tp;						      \
     asm ("stc gbr,%1\n\t"						      \
	  "mov.l 1f,%0\n\t"						      \
	  "bra 2f\n\t"							      \
	  " add %1,%0\n\t"						      \
	  ".align 2\n\t"						      \
	  "1: .long " #x "@tpoff\n\t"					      \
	  "2:"								      \
	  : "=r" (__l), "=r" (__tp));					      \
     __l; })

#ifdef PIC
# define TLS_IE(x) \
  ({ int *__l; void *__tp;						      \
     register void *__gp __asm__("r12");				      \
     asm ("mov.l 1f,r0\n\t"						      \
	  "stc gbr,%1\n\t"						      \
	  "mov.l @(r0,r12),%0\n\t"					      \
	  "bra 2f\n\t"							      \
	  " add %1,%0\n\t"						      \
	  ".align 2\n\t"						      \
	  "1: .long " #x "@gottpoff\n\t"				      \
	  "2:"								      \
	  : "=r" (__l), "=r" (__tp) : "r" (__gp) : "r0");		      \
     __l; })
#else
# define TLS_IE(x) \
  ({ int *__l; void *__tp;						      \
     asm ("mov.l r12,@-r15\n\t"						      \
	  "mova 0f,r0\n\t"						      \
	  "mov.l 0f,r12\n\t"						      \
	  "add r0,r12\n\t"						      \
	  "mov.l 1f,r0\n\t"						      \
	  "stc gbr,%1\n\t"						      \
	  "mov.l @(r0,r12),%0\n\t"					      \
	  "bra 2f\n\t"							      \
	  " add %1,%0\n\t"						      \
	  ".align 2\n\t"						      \
	  "1: .long " #x "@gottpoff\n\t"				      \
	  "0: .long _GLOBAL_OFFSET_TABLE_\n\t"				      \
	  "2: mov.l @r15+,r12"						      \
	  : "=r" (__l), "=r" (__tp) : : "r0");				      \
     __l; })
#endif

#ifdef PIC
# define TLS_LD(x) \
  ({ int *__l;								      \
     register void *__gp __asm__("r12");				      \
     asm ("mov.l 1f,r4\n\t"						      \
	  "mova 2f,r0\n\t"						      \
	  "mov.l 2f,r1\n\t"						      \
	  "add r0,r1\n\t"						      \
	  "jsr @r1\n\t"							      \
	  " add r12,r4\n\t"						      \
	  "bra 4f\n\t"							      \
	  " nop\n\t"							      \
	  ".align 2\n\t"						      \
	  "1: .long " #x "@tlsldm\n\t"					      \
	  "2: .long __tls_get_addr@plt\n\t"				      \
	  "4: mov.l 3f,%0\n\t"						      \
	  "bra 5f\n\t"							      \
	  " add r0,%0\n\t"						      \
	  ".align 2\n\t"						      \
	  "3: .long " #x "@dtpoff\n\t"					      \
	  "5:"								      \
	  : "=r" (__l) : "r" (__gp) : "r0", "r1", "r2", "r3", "r4", "r5",     \
				      "r6", "r7", "pr", "t");		      \
     __l; })
#else
# define TLS_LD(x) \
  ({ int *__l;								      \
     asm ("mov.l r12,@-r15\n\t"						      \
	  "mova 0f,r0\n\t"						      \
	  "mov.l 0f,r12\n\t"						      \
	  "add r0,r12\n\t"						      \
	  "mov.l 1f,r4\n\t"						      \
	  "mova 2f,r0\n\t"						      \
	  "mov.l 2f,r1\n\t"						      \
	  "add r0,r1\n\t"						      \
	  "jsr @r1\n\t"							      \
	  " add r12,r4\n\t"						      \
	  "bra 4f\n\t"							      \
	  " nop\n\t"							      \
	  ".align 2\n\t"						      \
	  "1: .long " #x "@tlsldm\n\t"					      \
	  "2: .long __tls_get_addr@plt\n\t"				      \
	  "0: .long _GLOBAL_OFFSET_TABLE_\n\t"				      \
	  "4: mov.l 3f,%0\n\t"						      \
	  "bra 5f\n\t"							      \
	  " add r0,%0\n\t"						      \
	  ".align 2\n\t"						      \
	  "3: .long " #x "@dtpoff\n\t"					      \
	  "5: mov.l @r15+,r12"						      \
	  : "=r" (__l) : : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",    \
			   "pr", "t");					      \
     __l; })
#endif

#ifdef PIC
# define TLS_GD(x) \
  ({ int *__l;								      \
     register void *__gp __asm__("r12");				      \
     asm ("mov.l 1f,r4\n\t"						      \
	  "mova 2f,r0\n\t"						      \
	  "mov.l 2f,r1\n\t"						      \
	  "add r0,r1\n\t"						      \
	  "jsr @r1\n\t"							      \
	  " add r12,r4\n\t"						      \
	  "bra 3f\n\t"							      \
	  " mov r0,%0\n\t"						      \
	  ".align 2\n\t"						      \
	  "1: .long " #x "@tlsgd\n\t"					      \
	  "2: .long __tls_get_addr@plt\n\t"				      \
	  "3:"								      \
	  : "=r" (__l) : "r" (__gp) : "r0", "r1", "r2", "r3", "r4", "r5",     \
				      "r6", "r7", "pr", "t");		      \
     __l; })
#else
# define TLS_GD(x) \
  ({ int *__l;								      \
     asm ("mov.l r12,@-r15\n\t"						      \
	  "mova 0f,r0\n\t"						      \
	  "mov.l 0f,r12\n\t"						      \
	  "add r0,r12\n\t"						      \
	  "mov.l 1f,r4\n\t"						      \
	  "mova 2f,r0\n\t"						      \
	  "mov.l 2f,r1\n\t"						      \
	  "add r0,r1\n\t"						      \
	  "jsr @r1\n\t"							      \
	  " add r12,r4\n\t"						      \
	  "bra 3f\n\t"							      \
	  " mov r0,%0\n\t"						      \
	  ".align 2\n\t"						      \
	  "1: .long " #x "@tlsgd\n\t"					      \
	  "2: .long __tls_get_addr@plt\n\t"				      \
	  "0: .long _GLOBAL_OFFSET_TABLE_\n\t"				      \
	  "3: mov.l @r15+,r12"						      \
	  : "=r" (__l) : : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",    \
			   "pr", "t");					      \
     __l; })
#endif
