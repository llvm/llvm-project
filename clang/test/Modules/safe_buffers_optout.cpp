// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodule-name=safe_buffers_test_base -x c++ %S/Inputs/SafeBuffers/safe_buffers_test.modulemap -std=c++20\
// RUN:     -o %t/safe_buffers_test_base.pcm -Wunsafe-buffer-usage
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodule-name=safe_buffers_test_optout -x c++ %S/Inputs/SafeBuffers/safe_buffers_test.modulemap -std=c++20\
// RUN:     -o %t/safe_buffers_test_optout.pcm -fmodule-file=%t/safe_buffers_test_base.pcm -Wunsafe-buffer-usage
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodule-file=%t/safe_buffers_test_optout.pcm -I %S/Inputs/SafeBuffers %s\
// RUN:     -std=c++20 -Wunsafe-buffer-usage

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -fmodule-map-file=%S/Inputs/SafeBuffers/safe_buffers_test.modulemap -I %S/Inputs/SafeBuffers %s\
// RUN:     -x c++ -std=c++20 -Wunsafe-buffer-usage

#include "test_sub1.h"
#include "test_sub2.h"

// Testing safe buffers opt-out region serialization with modules: this
// file loads 2 submodules from top-level module
// `safe_buffers_test_optout`, which uses another top-level module
// `safe_buffers_test_base`. (So the module dependencies form a DAG.)

// expected-warning@base.h:3{{unsafe buffer access}}
// expected-note@base.h:3{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@test_sub1.h:5{{unsafe buffer access}}
// expected-note@test_sub1.h:5{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@test_sub1.h:14{{unsafe buffer access}}
// expected-note@test_sub1.h:14{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// expected-warning@test_sub2.h:5{{unsafe buffer access}}
// expected-note@test_sub2.h:5{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
int foo(int * p) {
  int x = p[5]; // expected-warning{{unsafe buffer access}} expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
#pragma clang unsafe_buffer_usage begin
  int y = p[5];
#pragma clang unsafe_buffer_usage end
  sub1_T(p); // instantiate template
  return base(p) + sub1(p) + sub2(p);
}
