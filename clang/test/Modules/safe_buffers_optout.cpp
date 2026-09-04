// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t


// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodule-name=safe_buffers_test_base -x c++\
// RUN:            %t/safe_buffers_test.modulemap -std=c++20 -o %t/safe_buffers_test_base.pcm -Wunsafe-buffer-usage -verify
//--- safe_buffers_test.modulemap
module safe_buffers_test_base {
 header "base.h"
}

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodule-name=safe_buffers_test_textual -x c++ \
// RUN:            %t/safe_buffers_test.modulemap -std=c++20 -o %t/safe_buffers_test_textual.pcm \
// RUN:            -Wunsafe-buffer-usage
module safe_buffers_test_textual {
 textual header "textual.h"
}

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -emit-module -fmodule-name=safe_buffers_test_optout -x c++ \
// RUN:            %t/safe_buffers_test.modulemap -std=c++20 -fmodule-file=%t/safe_buffers_test_base.pcm \
// RUN:            -fmodule-file=%t/safe_buffers_test_textual.pcm -o %t/safe_buffers_test_optout.pcm -Wunsafe-buffer-usage\
// RUN:            -verify
module safe_buffers_test_optout {
  explicit module test_sub1 {  header "test_sub1.h" }
  explicit module test_sub2 {  header "test_sub2.h" }
  use safe_buffers_test_base
}


//--- base.h
#ifdef __cplusplus
int base(int *p) {
  int x = p[5]; // expected-warning{{unsafe buffer access}}\
		   expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
#pragma clang unsafe_buffer_usage begin
  int y = p[5];
#pragma clang unsafe_buffer_usage end
  return x + y;
}
#endif

//--- test_sub1.h
#include "base.h"

#ifdef __cplusplus
int sub1(int *p) {
  int x = p[5]; // expected-warning{{unsafe buffer access}}\
		   expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
#pragma clang unsafe_buffer_usage begin
  int y = p[5];
#pragma clang unsafe_buffer_usage end
  return x + y + base(p);
}

template <typename T>
T sub1_T(T *p) {
  T x = p[5];
#pragma clang unsafe_buffer_usage begin
  T y = p[5];
#pragma clang unsafe_buffer_usage end
  return x + y;
}
#endif

//--- test_sub2.h
#include "base.h"

#ifdef __cplusplus
int sub2(int *p) {
  int x = p[5]; // expected-warning{{unsafe buffer access}}\
		   expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
#pragma clang unsafe_buffer_usage begin
  int y = p[5];
#pragma clang unsafe_buffer_usage end
  return x + y + base(p);
}
#endif

//--- textual.h
#ifdef __cplusplus
int textual(int *p) {
  int x = p[5];
  int y = p[5];
  return x + y;
}
#endif

// Specify modules explicitly:
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodule-file=%t/safe_buffers_test_optout.pcm -I %t \
// RUN:            -std=c++20 -Wunsafe-buffer-usage -verify=main %t/safe_buffers_optout_main.cpp

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodule-file=%t/safe_buffers_test_optout.pcm -I %t \
// RUN:            -std=c++20 -Wunsafe-buffer-usage -verify=main-fixit %t/safe_buffers_optout_main.cpp \
// RUN:            -fsafe-buffer-usage-suggestions

// Specify modules implicitly:
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify=main -fmodules-cache-path=%t \
// RUN:            -fmodule-map-file=%t/safe_buffers_test.modulemap -I%t\
// RUN:            -x c++ -std=c++20 -Wunsafe-buffer-usage %t/safe_buffers_optout_main.cpp

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify=main-fixit -fmodules-cache-path=%t \
// RUN:            -fmodule-map-file=%t/safe_buffers_test.modulemap -I%t\
// RUN:            -x c++ -std=c++20 -Wunsafe-buffer-usage %t/safe_buffers_optout_main.cpp \
// RUN:            -fsafe-buffer-usage-suggestions

//--- safe_buffers_optout_main.cpp
#include "test_sub1.h"
#include "test_sub2.h"

// Testing safe buffers opt-out region serialization with modules: this
// file loads 2 submodules from top-level module
// `safe_buffers_test_optout`, which uses another top-level module
// `safe_buffers_test_base`. (So the module dependencies form a DAG.)

int foo(int * p) { // main-fixit-warning{{'p' is an unsafe pointer used for buffer access}} \
                      main-fixit-note{{change type of 'p' to 'std::span' to preserve bounds information}}
  int x = p[5]; // main-warning{{unsafe buffer access}} \
		   main-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}} \
                   main-fixit-note{{used in buffer access here}}
#pragma clang unsafe_buffer_usage begin
  int y = p[5];
#pragma clang unsafe_buffer_usage end
  sub1_T(p); // instantiate template
  return sub1(p) + sub2(p);
}
// main-warning@test_sub1.h:15 {{unsafe buffer access}}
// main-note@test_sub1.h:15 {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
// main-note@-5{{in instantiation of function template specialization 'sub1_T<int>' requested here}}

#pragma clang unsafe_buffer_usage begin
#include "textual.h"         // This header is textually included (i.e., it is in the same TU as %s), so warnings are suppressed
#pragma clang unsafe_buffer_usage end
