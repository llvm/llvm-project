// RUN: %clang_cc1  -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN:  -target-feature -sve -Waarch64-sme-attributes -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include "arm_sme.h"

int non_streaming_decl(void);
int streaming_decl(void) __arm_streaming;
int streaming_compatible_decl(void) __arm_streaming_compatible;

// Streaming-mode changes which would require spilling VG, unsupported without SVE

int streaming_caller_no_sve(void) __arm_streaming {
  // expected-warning@+1 {{function requires a streaming-mode change, unwinding is not possible without 'sve'}}
  return non_streaming_decl();
}

int sc_caller_non_streaming_callee(void) __arm_streaming_compatible {
  // expected-warning@+1 {{function requires a streaming-mode change, unwinding is not possible without 'sve'}}
  return non_streaming_decl();
}

__arm_locally_streaming int locally_streaming_no_sve(void) {
  // expected-warning@+1 {{unwinding is not possible for locally-streaming functions without 'sve'}}
  return streaming_decl();
}

// No warnings expected

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
