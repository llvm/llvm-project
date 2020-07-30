// RUN: %clang_cc1 -verify=expected,omp4 -fopenmp -fopenmp-version=45 -std=c++11 -o - %s
// RUN: %clang_cc1 -verify=expected,omp5 -fopenmp -fopenmp-version=50 -o - -std=c++11 %s -Wuninitialized

// RUN: not %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=aaa-bbb-ccc-ddd -o - %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -verify=expected,omp4 -fopenmp-simd -fopenmp-version=45 -std=c++11 -o - %s
// RUN: %clang_cc1 -verify=expected,omp5 -fopenmp-simd -fopenmp-version=50 -std=c++11 -o - %s
// CHECK: error: OpenMP target is invalid: 'aaa-bbb-ccc-ddd'
// RUN: not %clang_cc1 -fopenmp -std=c++11 -triple nvptx64-nvidia-cuda -o - %s 2>&1 | FileCheck --check-prefix CHECK-UNSUPPORTED-HOST-TARGET %s
// RUN: not %clang_cc1 -fopenmp -std=c++11 -triple nvptx-nvidia-cuda -o - %s 2>&1 | FileCheck --check-prefix CHECK-UNSUPPORTED-HOST-TARGET %s
// CHECK-UNSUPPORTED-HOST-TARGET: error: The target '{{nvptx64-nvidia-cuda|nvptx-nvidia-cuda}}' is not a supported OpenMP host target.
// RUN: not %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=hexagon-linux-gnu -o - %s 2>&1 | FileCheck --check-prefix CHECK-UNSUPPORTED-DEVICE-TARGET %s
// CHECK-UNSUPPORTED-DEVICE-TARGET: OpenMP target is invalid: 'hexagon-linux-gnu'

// RUN: not %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path 1111.bc -o - 2>&1 | FileCheck --check-prefix NO-HOST-BC %s
// NO-HOST-BC: The provided host compiler IR file '1111.bc' is required to generate code for OpenMP target regions but cannot be found.

// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc -DREGION_HOST
// RUN: not %clang_cc1 -verify=expected,omp4 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -DREGION_DEVICE 2>&1
// RUN: not %clang_cc1 -verify=expected,omp5 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -DREGION_DEVICE 2>&1

#if defined(REGION_HOST) || defined(REGION_DEVICE)
void foo() {
#ifdef REGION_HOST
#pragma omp target // expected-error {{Offloading entry for target region in _Z3foov is incorrect: either the address or the ID is invalid.}}
  ;
#endif
#ifdef REGION_DEVICE
#pragma omp target
  ;
#endif
}
#pragma omp declare target to(foo)
void bar() {
#ifdef REGION_HOST
#pragma omp target
  ;
#endif
#ifdef REGION_DEVICE
#pragma omp target
  ;
#endif
}
#else
void foo() {
}

class S {
  public:
  void zee() {
    #pragma omp target map(this[:2]) // expected-note {{expected length on mapping of 'this' array section expression to be '1'}} // expected-error {{invalid 'this' expression on 'map' clause}}
      int a;
    #pragma omp target map(this[1:1]) // expected-note {{expected lower bound on mapping of 'this' array section expression to be '0' or not specified}} // expected-error {{invalid 'this' expression on 'map' clause}}
      int b;
    #pragma omp target map(this[1]) // expected-note {{expected 'this' subscript expression on map clause to be 'this[0]'}} // expected-error {{invalid 'this' expression on 'map' clause}}
      int c;
#pragma omp target map(foo)         // omp4-error {{expected expression containing only member accesses and/or array sections based on named variables}} omp5-error {{expected addressable lvalue in 'map' clause}}
      int d;
#pragma omp target map(zee)         // omp4-error {{expected expression containing only member accesses and/or array sections based on named variables}} omp5-error {{expected addressable lvalue in 'map' clause}}
      int e;
#pragma omp target map(this->zee)   // omp4-error {{expected expression containing only member accesses and/or array sections based on named variables}} omp5-error {{expected addressable lvalue in 'map' clause}}
      int f;
  }
};

#pragma omp target // expected-error {{unexpected OpenMP directive '#pragma omp target'}}

int main(int argc, char **argv) {
  #pragma omp target { // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target ( // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target [ // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target ] // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target ) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target } // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target
  foo();
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp target' are ignored}}
  #pragma omp target unknown()
  foo();
  L1:
    foo();
  #pragma omp target
  ;
  #pragma omp target
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch(argc) {
     case (0):
      #pragma omp target
      {
        foo();
        break; // expected-error {{'break' statement not in loop or switch statement}}
        continue; // expected-error {{'continue' statement not in loop statement}}
      }
      default:
       break;
    }
  }

  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp target
  L2:
  foo();
  #pragma omp target
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
  #pragma omp target
  for (int n = 0; n < 100; ++n) {}

#pragma omp target map(foo) // omp4-error {{expected expression containing only member accesses and/or array sections based on named variables}} omp5-error {{expected addressable lvalue in 'map' clause}}
  {}

  S s;

#pragma omp target map(s.zee) // omp4-error {{expected expression containing only member accesses and/or array sections based on named variables}} omp5-error {{expected addressable lvalue in 'map' clause}}
  {}

  return 0;
}

template <class> struct a { static bool b; };
template <class c, bool = a<c>::b> void e(c) { // expected-note {{candidate template ignored: substitution failure [with c = int]: non-type template argument is not a constant expression}}
#pragma omp target
  {
    int d ; e(d); // expected-error {{no matching function for call to 'e'}}
  }
}
#endif
