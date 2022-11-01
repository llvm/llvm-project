// REQUIRES: nvptx-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 "-triple" "x86_64-unknown-linux-gnu" "-aux-triple" "nvptx64-nvidia-cuda" \
// RUN:    "-target-cpu" "x86-64" -fsyntax-only -verify=scalar %s
// RUN: %clang_cc1 "-aux-triple" "x86_64-unknown-linux-gnu" "-triple" "nvptx64-nvidia-cuda" \
// RUN:    -fcuda-is-device "-aux-target-cpu" "x86-64" -fsyntax-only -verify=scalar %s

#include "Inputs/cuda.h"

__device__ void test(bool b, __bf16 *out, __bf16 in) {
  __bf16 bf16 = in; // No error on using the type itself.

  bf16 + bf16; // scalar-error {{invalid operands to binary expression ('__bf16' and '__bf16')}}
  bf16 - bf16; // scalar-error {{invalid operands to binary expression ('__bf16' and '__bf16')}}
  bf16 * bf16; // scalar-error {{invalid operands to binary expression ('__bf16' and '__bf16')}}
  bf16 / bf16; // scalar-error {{invalid operands to binary expression ('__bf16' and '__bf16')}}

  __fp16 fp16;

  bf16 + fp16; // scalar-error {{invalid operands to binary expression ('__bf16' and '__fp16')}}
  fp16 + bf16; // scalar-error {{invalid operands to binary expression ('__fp16' and '__bf16')}}
  bf16 - fp16; // scalar-error {{invalid operands to binary expression ('__bf16' and '__fp16')}}
  fp16 - bf16; // scalar-error {{invalid operands to binary expression ('__fp16' and '__bf16')}}
  bf16 * fp16; // scalar-error {{invalid operands to binary expression ('__bf16' and '__fp16')}}
  fp16 * bf16; // scalar-error {{invalid operands to binary expression ('__fp16' and '__bf16')}}
  bf16 / fp16; // scalar-error {{invalid operands to binary expression ('__bf16' and '__fp16')}}
  fp16 / bf16; // scalar-error {{invalid operands to binary expression ('__fp16' and '__bf16')}}
  bf16 = fp16; // scalar-error {{assigning to '__bf16' from incompatible type '__fp16'}}
  fp16 = bf16; // scalar-error {{assigning to '__fp16' from incompatible type '__bf16'}}
  bf16 + (b ? fp16 : bf16); // scalar-error {{incompatible operand types ('__fp16' and '__bf16')}}
  *out = bf16;
}
