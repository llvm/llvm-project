/* Macros to support TLS testing in times of missing compiler support.  */

#define COMMON_INT_DEF(x) \
  __thread int x;
/* XXX Until we get compiler support we don't need declarations.  */
#define COMMON_INT_DECL(x)

/* XXX This definition will probably be machine specific, too.  */
#define VAR_INT_DEF(x) \
  asm (".section .tdata\n\t"						      \
       ".globl " #x "\n"						      \
       ".balign 4\n"							      \
       #x ":\t.long 0\n\t"						      \
       ".size " #x ",4\n\t"						      \
       ".previous")
/* XXX Until we get compiler support we don't need declarations.  */
#define VAR_INT_DECL(x)

#include_next <tls-macros.h>

  /* XXX Each architecture must have its own asm for now.  */
#if !defined TLS_LE || !defined TLS_IE \
      || !defined TLS_LD || !defined TLS_GD
# error "No support for this architecture so far."
#endif
