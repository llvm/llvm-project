#ifndef _FPU_CONTROL_H
#include_next <fpu_control.h>

# ifndef _ISOMAC

/* Called at startup.  It can be used to manipulate fpu control register.  */
extern void __setfpucw (fpu_control_t) attribute_hidden;

# endif /* !_ISOMAC */
#endif /* fpu_control.h */
