// RUN: %clang_cc1 -fsyntax-only -verify=expected,host %s
// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify=expected,dev %s

#include "Inputs/cuda.h"

// Implicit deduction guide for host.
template <typename T>
struct HGuideImp {       // expected-note {{candidate template ignored: could not match 'HGuideImp<T>' against 'int'}}
   HGuideImp(T value) {} // expected-note {{candidate function not viable: call to __host__ function from __device__ function}}
                         // dev-note@-1 {{'<deduction guide for HGuideImp><int>' declared here}}
                         // dev-note@-2 {{'HGuideImp' declared here}}
};

// Explicit deduction guide for host.
template <typename T>
struct HGuideExp {       // expected-note {{candidate template ignored: could not match 'HGuideExp<T>' against 'int'}}
   HGuideExp(T value) {} // expected-note {{candidate function not viable: call to __host__ function from __device__ function}}
                         // dev-note@-1 {{'HGuideExp' declared here}}
};
template<typename T>
HGuideExp(T) -> HGuideExp<T>; // expected-note {{candidate function not viable: call to __host__ function from __device__ function}}
                              // dev-note@-1 {{'<deduction guide for HGuideExp><int>' declared here}}

// Implicit deduction guide for device.
template <typename T>
struct DGuideImp {                  // expected-note {{candidate template ignored: could not match 'DGuideImp<T>' against 'int'}}
   __device__ DGuideImp(T value) {} // expected-note {{candidate function not viable: call to __device__ function from __host__ function}}
                                    // host-note@-1 {{'<deduction guide for DGuideImp><int>' declared here}}
                                    // host-note@-2 {{'DGuideImp' declared here}}
};

// Explicit deduction guide for device.
template <typename T>
struct DGuideExp {                   // expected-note {{candidate template ignored: could not match 'DGuideExp<T>' against 'int'}}
   __device__ DGuideExp(T value) {}  // expected-note {{candidate function not viable: call to __device__ function from __host__ function}}
                                     // host-note@-1 {{'DGuideExp' declared here}}
};

template<typename T>
__device__ DGuideExp(T) -> DGuideExp<T>; // expected-note {{candidate function not viable: call to __device__ function from __host__ function}}
                                         // host-note@-1 {{'<deduction guide for DGuideExp><int>' declared here}}

template <typename T>
struct HDGuide {
   __device__ HDGuide(T value) {}
   HDGuide(T value) {}
};

template<typename T>
HDGuide(T) -> HDGuide<T>;

template<typename T>
__device__ HDGuide(T) -> HDGuide<T>;

void hfun() {
    HGuideImp hgi = 10;
    HGuideExp hge = 10;
    DGuideImp dgi = 10; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'DGuideImp'}}
    DGuideExp dge = 10; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'DGuideExp'}}
    HDGuide hdg = 10;
}

__device__ void dfun() {
    HGuideImp hgi = 10; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'HGuideImp'}}
    HGuideExp hge = 10; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'HGuideExp'}}
    DGuideImp dgi = 10;
    DGuideExp dge = 10;
    HDGuide hdg = 10;
}

__host__ __device__ void hdfun() {
    HGuideImp hgi = 10; // dev-error {{reference to __host__ function '<deduction guide for HGuideImp><int>' in __host__ __device__ function}}
                        // dev-error@-1 {{reference to __host__ function 'HGuideImp' in __host__ __device__ function}}
    HGuideExp hge = 10; // dev-error {{reference to __host__ function '<deduction guide for HGuideExp><int>' in __host__ __device__ function}}
                        // dev-error@-1 {{reference to __host__ function 'HGuideExp' in __host__ __device__ function}}
    DGuideImp dgi = 10; // host-error {{reference to __device__ function '<deduction guide for DGuideImp><int>' in __host__ __device__ function}}
                        // host-error@-1 {{reference to __device__ function 'DGuideImp' in __host__ __device__ function}}
    DGuideExp dge = 10; // host-error {{reference to __device__ function '<deduction guide for DGuideExp><int>' in __host__ __device__ function}}
                        // host-error@-1 {{reference to __device__ function 'DGuideExp' in __host__ __device__ function}}
    HDGuide hdg = 10;
}

HGuideImp hgi = 10;
HGuideExp hge = 10;
HDGuide hdg = 10;
