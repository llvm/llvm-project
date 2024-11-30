#ifndef FORTRAN_RUNTIME_CONFIG_H
#define FORTRAN_RUNTIME_CONFIG_H

/* Define to 1 if you have the `strerror_r' function. */
#cmakedefine01 HAVE_STRERROR_R

/* Define to 1 if you have the declaration of `strerror_s', and to 0 if you
   don't. */
#cmakedefine01 HAVE_DECL_STRERROR_S

/* Define to 1 if you have the `backtrace' function. */
#cmakedefine HAVE_BACKTRACE ${HAVE_BACKTRACE}

#define BACKTRACE_HEADER <${BACKTRACE_HEADER}>

#endif
