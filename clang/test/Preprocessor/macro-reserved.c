// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

#define for 0    // expected-warning {{keyword is hidden by macro definition}}
#define final 1
#define __HAVE_X 0
#define __cplusplus
#define _HAVE_X 0
#define X__Y
#define __STDC__ 1 // expected-warning {{redefining builtin macro}}
#define __clang__ 1

#undef for // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#undef final
#undef __HAVE_X
#undef __cplusplus
#undef _HAVE_X
#undef X__Y
#undef __STDC_HOSTED__ // expected-warning {{undefining builtin macro}}
#undef __INT32_TYPE__
#undef __UINT32_TYPE__
#undef __UINTPTR_TYPE__
#undef __UINT64_TYPE__
#undef __INT64_TYPE__
#undef __OPTIMIZE__

// allowlisted definitions
#define while while
#define const
#define static
#define extern
#define inline

#undef while // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#undef const // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#undef static // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#undef extern // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#undef inline // expected-warning {{keyword or identifier with special meaning is used as a macro name}}

#define inline __inline
#undef  inline // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#define inline __inline__
#undef  inline // expected-warning {{keyword or identifier with special meaning is used as a macro name}}

#define inline inline__  // expected-warning {{keyword is hidden by macro definition}}
#undef  inline // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#define extern __inline  // expected-warning {{keyword is hidden by macro definition}}
#undef  extern // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#define extern __extern	 // expected-warning {{keyword is hidden by macro definition}}
#undef  extern // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#define extern __extern__ // expected-warning {{keyword is hidden by macro definition}}
#undef  extern // expected-warning {{keyword or identifier with special meaning is used as a macro name}}

#define inline _inline   // expected-warning {{keyword is hidden by macro definition}}
#undef  inline // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#define volatile   // expected-warning {{keyword is hidden by macro definition}}
#undef  volatile // expected-warning {{keyword or identifier with special meaning is used as a macro name}}

#pragma clang diagnostic warning "-Wreserved-macro-identifier"

#define switch if  // expected-warning {{keyword is hidden by macro definition}}
#define final 1
#define __clusplus // expected-warning {{macro name is a reserved identifier}}
#define __HAVE_X 0 // expected-warning {{macro name is a reserved identifier}}
#define _HAVE_X 0  // expected-warning {{macro name is a reserved identifier}}
#define X__Y

#undef switch // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#undef final
#undef __cplusplus // expected-warning {{macro name is a reserved identifier}}
#undef _HAVE_X     // expected-warning {{macro name is a reserved identifier}}
#undef X__Y

int x;

#define _GNU_SOURCE          // no-warning
#define __STDC_FORMAT_MACROS // no-warning
