// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN:  -target-feature -sve -fexceptions -DNO_THROW -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN:  -target-feature -sve -fexceptions -DNO_EXCEPT -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN:  -target-feature -sve -DNO_EXCEPT_FLAG -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include "arm_sme.h"

int non_streaming_decl(void);
int streaming_decl(void) __arm_streaming;
int streaming_compatible_decl(void) __arm_streaming_compatible;

#ifdef NO_THROW
#define NOTHROW_ATTR __attribute__((__nothrow__))
#else
#define NOTHROW_ATTR
#endif

#ifdef NO_EXCEPT
#define NOEXCEPT_ATTR noexcept
#else
#define NOEXCEPT_ATTR
#endif

#ifdef NO_EXCEPT_FLAG
  // expected-no-diagnostics
#endif

NOTHROW_ATTR int nothrow_non_streaming_decl(void) NOEXCEPT_ATTR;
NOTHROW_ATTR int nothrow_streaming_decl(void) NOEXCEPT_ATTR;
NOTHROW_ATTR int nothrow_streaming_compatible_decl(void) NOEXCEPT_ATTR;

// Streaming-mode changes which would require spilling VG if unwinding is possible, unsupported without SVE

int streaming_caller_no_sve(void) __arm_streaming {
#ifndef NO_EXCEPT_FLAG
  // expected-error@+2 {{function requires a streaming-mode change, unwinding is not possible without 'sve'. Consider marking this function as 'noexcept' or '__attribute__((nothrow))'}}
#endif
  return non_streaming_decl();
}

int sc_caller_non_streaming_callee(void) __arm_streaming_compatible {
#ifndef NO_EXCEPT_FLAG
  // expected-error@+2 {{function requires a streaming-mode change, unwinding is not possible without 'sve'. Consider marking this function as 'noexcept' or '__attribute__((nothrow))'}}
#endif
  return non_streaming_decl();
}

__arm_locally_streaming int locally_streaming_no_sve(void) {
#ifndef NO_EXCEPT_FLAG
  // expected-error@+2 {{unwinding is not possible for locally-streaming functions without 'sve'. Consider marking this function as 'noexcept' or '__attribute__((nothrow))'}}
#endif
  return streaming_decl();
}

// Nothrow / noexcept attribute on callee - warnings not expected

int nothrow_streaming_caller_no_sve(void) __arm_streaming {
  return nothrow_non_streaming_decl();
}

int nothrow_sc_caller_non_streaming_callee(void) __arm_streaming_compatible {
  return nothrow_non_streaming_decl();
}

__arm_locally_streaming int nothrow_locally_streaming_no_sve(void) {
  return nothrow_streaming_decl();
}

// No warnings expected, even if unwinding is possible

int normal_caller_streaming_callee(void) {
  return streaming_decl();
}

int normal_caller_streaming_compatible_callee(void) {
  return streaming_compatible_decl();
}

int sc_caller_streaming_callee(void) __arm_streaming_compatible {
  return streaming_decl();
}

int sc_caller_sc_callee(void) __arm_streaming_compatible {
  return streaming_compatible_decl();
}
