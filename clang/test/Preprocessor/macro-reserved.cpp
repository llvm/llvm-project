// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -std=c++98 %s

#define for 0    // expected-warning {{keyword is hidden by macro definition}}
#define final 1
#define __HAVE_X 0
#define _HAVE_X 0
#define X__Y

#undef for
#undef final
#undef __HAVE_X
#undef _HAVE_X
#undef X__Y

#undef __cplusplus
#define __cplusplus

// whitelisted definitions
#define while while
#define const
#define static
#define extern
#define inline

#undef while
#undef const
#undef static
#undef extern
#undef inline

#define inline __inline
#undef  inline
#define inline __inline__
#undef  inline

#define inline inline__  // expected-warning {{keyword is hidden by macro definition}}
#undef  inline
#define extern __inline  // expected-warning {{keyword is hidden by macro definition}}
#undef  extern
#define extern __extern	 // expected-warning {{keyword is hidden by macro definition}}
#undef  extern
#define extern __extern__ // expected-warning {{keyword is hidden by macro definition}}
#undef  extern

#define inline _inline   // expected-warning {{keyword is hidden by macro definition}}
#undef  inline
#define volatile   // expected-warning {{keyword is hidden by macro definition}}
#undef  volatile

#pragma clang diagnostic warning "-Wreserved-macro-identifier"

#define switch if  // expected-warning {{keyword is hidden by macro definition}}
#define final 1
#define __HAVE_X 0 // expected-warning {{macro name is a reserved identifier}}
#define _HAVE_X 0  // expected-warning {{macro name is a reserved identifier}}
#define X__Y       // expected-warning {{macro name is a reserved identifier}}

#undef __cplusplus // expected-warning {{macro name is a reserved identifier}}
#undef _HAVE_X     // expected-warning {{macro name is a reserved identifier}}
#undef X__Y        // expected-warning {{macro name is a reserved identifier}}

int x;
