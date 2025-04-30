/* Macros to support TLS testing in times of missing compiler support.  */

extern void *__tls_get_addr (void *);

# define TLS_LE(x) \
  ({ void *__l;								      \
     asm ("mov r2=r13\n\t"						      \
         ";;\n\t"							      \
         "addl %0=@tprel(" #x "),r2\n\t"				      \
         : "=r" (__l) : : "r2"  ); __l; })

# define TLS_IE(x) \
  ({ void *__l;								      \
     register long __gp asm ("gp");					      \
     asm (";;\n\t"							      \
	 "addl r16=@ltoff(@tprel(" #x ")),gp\n\t"			      \
         ";;\n\t"							      \
         "ld8 r17=[r16]\n\t"						      \
         ";;\n\t"							      \
         "add %0=r13,r17\n\t"						      \
         ";;\n\t"							      \
         : "=r" (__l) : "r" (__gp) : "r16", "r17" ); __l; })

# define __TLS_CALL_CLOBBERS \
  "r2", "r3", "r8", "r9", "r10", "r11", "r14", "r15", "r16", "r17",	      \
  "r18", "r19", "r20", "r21", "r22", "r23", "r24", "r25", "r26",	      \
  "r27", "r28", "r29", "r30", "r31",					      \
  "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15",	      \
  "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",	      \
  "b6", "b7",								      \
  "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7"

# define TLS_LD(x) \
  ({ void *__l;								      \
     register long __gp asm ("gp");					      \
     asm (";;\n\t"							      \
	 "mov loc0=gp\n\t"						      \
         "addl r16=@ltoff(@dtpmod(" #x ")),gp\n\t"			      \
         "addl out1=@dtprel(" #x "),r0\n\t"				      \
         ";;\n\t"							      \
         "ld8 out0=[r16]\n\t"						      \
         "br.call.sptk.many b0=__tls_get_addr"				      \
         ";;\n\t"							      \
         "mov gp=loc0\n\t"						      \
         "mov %0=r8\n\t"						      \
         ";;\n\t"							      \
         : "=r" (__l) : "r" (__gp) : "loc0", __TLS_CALL_CLOBBERS);	      \
     __l; })

# define TLS_GD(x) \
  ({ void *__l;								      \
     register long __gp asm ("gp");					      \
     asm (";;\n\t"							      \
	 "mov loc0=gp\n\t"						      \
         "addl r16=@ltoff(@dtpmod(" #x ")),gp\n\t"			      \
         "addl r17=@ltoff(@dtprel(" #x ")),gp\n\t"			      \
         ";;\n\t"							      \
         "ld8 out0=[r16]\n\t"						      \
         "ld8 out1=[r17]\n\t"						      \
         "br.call.sptk.many b0=__tls_get_addr"				      \
         ";;\n\t"							      \
         "mov gp=loc0\n\t"						      \
         "mov %0=r8\n\t"						      \
         ";;\n\t"							      \
          : "=r" (__l) : "r" (__gp) : "loc0", __TLS_CALL_CLOBBERS);	      \
     __l; })
