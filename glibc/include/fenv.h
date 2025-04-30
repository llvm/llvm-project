#ifndef _FENV_H
#include <math/fenv.h>

#ifndef _ISOMAC
# include <stdbool.h>
/* Now define the internal interfaces.  */

extern int __feclearexcept (int __excepts);
extern int __fegetexcept (void);
extern int __fegetexceptflag (fexcept_t *__flagp, int __excepts);
extern int __feraiseexcept (int __excepts);
extern int __fesetexceptflag (const fexcept_t *__flagp, int __excepts);
extern int __fegetenv (fenv_t *__envp);
extern int __fesetenv (const fenv_t *__envp);
extern int __feupdateenv (const fenv_t *__envp);
extern __typeof (fegetround) __fegetround __attribute_pure__;
extern __typeof (feholdexcept) __feholdexcept;
extern __typeof (fesetround) __fesetround;

libm_hidden_proto (feraiseexcept)
libm_hidden_proto (__feraiseexcept)
libm_hidden_proto (fegetenv)
libm_hidden_proto (__fegetenv)
libm_hidden_proto (fegetround)
libm_hidden_proto (__fegetround)
libm_hidden_proto (fesetenv)
libm_hidden_proto (__fesetenv)
libm_hidden_proto (fesetround)
libm_hidden_proto (__fesetround)
libm_hidden_proto (feholdexcept)
libm_hidden_proto (__feholdexcept)
libm_hidden_proto (feupdateenv)
libm_hidden_proto (__feupdateenv)
libm_hidden_proto (fetestexcept)
libm_hidden_proto (feclearexcept)

/* Rounding mode context.  This allows functions to set/restore rounding mode
   only when the desired rounding mode is different from the current rounding
   mode.  */
struct rm_ctx
{
  fenv_t env;
  bool updated_status;
};

/* Track whether rounding mode macros were defined, since
   get-rounding-mode.h may define default versions if they weren't.
   FE_TONEAREST must always be defined (even if no changes of rounding
   mode are supported, glibc requires it to be defined to represent
   the default rounding mode).  */
# ifndef FE_TONEAREST
#  error "FE_TONEAREST not defined"
# endif
# if defined FE_DOWNWARD || defined FE_TOWARDZERO || defined FE_UPWARD
#  define FE_HAVE_ROUNDING_MODES 1
# else
#  define FE_HAVE_ROUNDING_MODES 0
# endif

/* When no floating-point exceptions are defined in <fenv.h>, make
   feraiseexcept ignore its argument so that unconditional
   feraiseexcept calls do not cause errors for undefined exceptions.
   Define it to expand to a void expression so that any calls testing
   the result of feraiseexcept do produce errors.  */
# if FE_ALL_EXCEPT == 0
#  define feraiseexcept(excepts) ((void) 0)
#  define __feraiseexcept(excepts) ((void) 0)
# endif

/* Similarly, most <fenv.h> functions have trivial implementations in
   the absence of support for floating-point exceptions and rounding
   modes.  */

# if !FE_HAVE_ROUNDING_MODES
#  if FE_ALL_EXCEPT == 0
extern inline int
fegetenv (fenv_t *__e)
{
  return 0;
}

extern inline int
__fegetenv (fenv_t *__e)
{
  return 0;
}

extern inline int
feholdexcept (fenv_t *__e)
{
  return 0;
}

extern inline int
__feholdexcept (fenv_t *__e)
{
  return 0;
}

extern inline int
fesetenv (const fenv_t *__e)
{
  return 0;
}

extern inline int
__fesetenv (const fenv_t *__e)
{
  return 0;
}

extern inline int
feupdateenv (const fenv_t *__e)
{
  return 0;
}

extern inline int
__feupdateenv (const fenv_t *__e)
{
  return 0;
}
#  endif

extern inline int
fegetround (void)
{
  return FE_TONEAREST;
}

extern inline int
__fegetround (void)
{
  return FE_TONEAREST;
}

extern inline int
fesetround (int __d)
{
  return 0;
}

extern inline int
__fesetround (int __d)
{
  return 0;
}
# endif

#endif

#endif
