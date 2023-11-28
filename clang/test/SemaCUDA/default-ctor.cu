// RUN: %clang_cc1 -std=c++11 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -fcuda-is-device -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:            -verify -verify-ignore-unexpected=note %s

#include "Inputs/cuda.h"

struct In { In() = default; };
struct InD { __device__ InD() = default; };
struct InH { __host__ InH() = default; };
struct InHD { __host__ __device__ InHD() = default; };

struct Out { Out(); };
struct OutD { __device__ OutD(); };
struct OutH { __host__ OutH(); };
struct OutHD { __host__ __device__ OutHD(); };

Out::Out() = default;
__device__ OutD::OutD() = default;
__host__ OutH::OutH() = default;
__host__ __device__ OutHD::OutHD() = default;

__device__ void fd() {
  In in;
  InD ind;
  InH inh; // expected-error{{no matching constructor for initialization of 'InH'}}
  InHD inhd;
  Out out; // expected-error{{no matching constructor for initialization of 'Out'}}
  OutD outd;
  OutH outh; // expected-error{{no matching constructor for initialization of 'OutH'}}
  OutHD outhd;
}

__host__ void fh() {
  In in;
  InD ind; // expected-error{{no matching constructor for initialization of 'InD'}}
  InH inh;
  InHD inhd;
  Out out;
  OutD outd; // expected-error{{no matching constructor for initialization of 'OutD'}}
  OutH outh;
  OutHD outhd;
}
