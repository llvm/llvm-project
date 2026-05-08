// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

// Test openCL and SYCL spelling conversions for address space
// attributes.

void test_incompatible() {
  __attribute__((opencl_global)) int *opencl_global;
  int [[clang::sycl_local]] *sycl_local;
  int [[clang::sycl_private]] *sycl_private;

  // Address space attributes are resolved using mode of compilation and not the spelling itself. This results in the SYCL spelling
  // being used in both instances of each diagnostic despite openCL spelling being used. 
  opencl_global = sycl_local; // expected-error {{assigning 'sycl_local int *' to 'sycl_global int *' changes address space of pointer}}
  opencl_global = sycl_private; // expected-error {{assigning 'sycl_private int *' to 'sycl_global int *' changes address space of pointer}}
  sycl_local = opencl_global; // expected-error {{assigning 'sycl_global int *' to 'sycl_local int *' changes address space of pointer}}
}

void test_to_generic_mixed() {
  __attribute__((opencl_generic)) int *opencl_gen;
  int [[clang::sycl_generic]] *sycl_gen;

  __attribute__((opencl_global)) int *opencl_global;
  int [[clang::sycl_local]] *sycl_local;
  int [[clang::sycl_private]] *sycl_private;

  opencl_gen = sycl_local;
  opencl_gen = sycl_private;
  sycl_gen = opencl_global;

}

void overload_test(__attribute__((opencl_global)) int *p) { (void)p; } // expected-note {{previous definition is here}}
void overload_test(__attribute__((sycl_global)) int *p) { (void)p; } // expected-error {{redefinition of 'overload_test'}}

