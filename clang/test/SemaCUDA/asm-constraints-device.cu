// Verify that we do check for constraints in device-side inline
// assembly. Passing an illegal input/output constraint and look 
// for corresponding error
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fsyntax-only -fcuda-is-device -verify %s

__attribute__((device)) void df() {
  short h;
  int a;
  // asm with PTX constraints. Some of them are PTX-specific.
  __asm__("output constraints"
          : "=h"(h), // .u16 reg, OK
            "=a"(a)  // expected-error {{invalid output constraint '=a' in asm}}
          :          // None
          );
  __asm__("input constraints"
          :           // None
          : "f"(0.0), // .f32 reg, OK
            "d"(0.0), // .f64 reg, OK
            "h"(0),   // .u16 reg, OK
            "r"(0),   // .u32 reg, OK
            "l"(0),   // .u64 reg, OK
            "a"(0)    // expected-error {{invalid input constraint 'a' in asm}}
          );
}
