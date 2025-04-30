/* The libm assembly code wants to include <machine/asm.h> to define the
   ENTRY macro.  We define assembly-related macros in sysdep.h and
   asm-syntax.h.  */

#include <sysdep.h>
#include <asm-syntax.h>

/* The libm assembly code uses this macro for RCSid strings.
   We don't put RCSid strings into object files.  */
#define RCSID(id) /* ignore them */
