// RUN: %clang_cc1 -triple i586-pc-haiku -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple i686-pc-linux -fsyntax-only -verify %s

// expected-no-diagnostics

#ifdef __HAIKU__
typedef signed long pid_t;
#else
typedef signed int pid_t;
#endif
pid_t	 vfork(void);