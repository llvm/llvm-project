// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -std=c++23 -fsyntax-only -verify %s

// This test is testing the diagnostics that Clang emits when compiling without '+sme'.

void streaming_compatible_def() __arm_streaming_compatible {} // OK
void streaming_def() __arm_streaming { } // expected-error {{function executed in streaming-SVE mode requires 'sme'}}
void shared_za_def() __arm_inout("za") { } // expected-error {{function using ZA state requires 'sme'}}
__arm_new("za") void new_za_def() { } // expected-error {{function using ZA state requires 'sme'}}
__arm_locally_streaming void locally_streaming_def() { } // expected-error {{function executed in streaming-SVE mode requires 'sme'}}
void streaming_shared_za_def() __arm_streaming __arm_inout("za") { } // expected-error {{function executed in streaming-SVE mode requires 'sme'}}
void inout_za_def() __arm_inout("za") { } // expected-error {{function using ZA state requires 'sme'}}
void inout_zt0_def() __arm_inout("zt0") { } // expected-error {{function using ZT0 state requires 'sme2'}}

// It should work fine when we explicitly add the target("sme") attribute.
__attribute__((target("sme"))) void streaming_compatible_def_sme_attr() __arm_streaming_compatible {} // OK
__attribute__((target("sme"))) void streaming_def_sme_attr() __arm_streaming { } // OK
__attribute__((target("sme"))) void shared_za_def_sme_attr() __arm_inout("za") { } // OK
__arm_new("za") __attribute__((target("sme"))) void new_za_def_sme_attr() {} // OK
__arm_locally_streaming __attribute__((target("sme"))) void locally_streaming_def_sme_attr() {} // OK

// Test that it also works with the target("sme2") attribute.
__attribute__((target("sme2"))) void streaming_def_sme2_attr() __arm_streaming { } // OK

// No code is generated for declarations, so it should be fine to declare using the attribute.
void streaming_compatible_decl() __arm_streaming_compatible; // OK
void streaming_decl() __arm_streaming; // OK
void shared_za_decl() __arm_inout("za"); // OK

void non_streaming_decl();
void non_streaming_def(void (*streaming_fn_ptr)(void) __arm_streaming,
                       void (*streaming_compatible_fn_ptr)(void) __arm_streaming_compatible) {
  streaming_compatible_decl(); // OK
  streaming_compatible_fn_ptr(); // OK
  streaming_decl(); // expected-error {{call to a streaming function requires 'sme'}}
  streaming_fn_ptr(); // expected-error {{call to a streaming function requires 'sme'}}
}

void streaming_compatible_def2(void (*streaming_fn_ptr)(void) __arm_streaming,
                               void (*streaming_compatible_fn_ptr)(void) __arm_streaming_compatible)
                                __arm_streaming_compatible {
  non_streaming_decl(); // OK
  streaming_compatible_decl(); // OK
  streaming_compatible_fn_ptr(); // OK
  streaming_decl(); // expected-error {{call to a streaming function requires 'sme'}}
  streaming_fn_ptr(); // expected-error {{call to a streaming function requires 'sme'}}
}

// Also test when call-site is not a function.
int streaming_decl_ret_int() __arm_streaming;
int x = streaming_decl_ret_int(); // expected-error {{call to a streaming function requires 'sme'}}

void sme_attrs_lambdas() {
  [] __arm_locally_streaming () { return; }();  // expected-error {{function executed in streaming-SVE mode requires 'sme'}}
  [] __arm_new("za") () { return; }();  // expected-error {{function using ZA state requires 'sme'}}
  [] __arm_new("zt0") () { return; }();  // expected-error {{function using ZT0 state requires 'sme2'}}
}
