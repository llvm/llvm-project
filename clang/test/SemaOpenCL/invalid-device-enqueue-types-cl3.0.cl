// RUN: %clang_cc1 -verify -fblocks -cl-std=CL3.0 -cl-ext=-all,+__opencl_c_device_enqueue,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables %s
// RUN: %clang_cc1 -verify -fblocks -cl-std=CL3.0 -cl-ext=-__opencl_c_device_enqueue %s

void f() {
  clk_event_t e;
  queue_t q;
#ifndef __opencl_c_device_enqueue
// expected-error@-3 {{use of undeclared identifier 'clk_event_t'}}
// expected-error@-3 {{use of undeclared identifier 'queue_t'}}
#else
// expected-no-diagnostics
#endif
}
