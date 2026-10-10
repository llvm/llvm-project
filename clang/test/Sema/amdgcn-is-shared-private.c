// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fsyntax-only -verify -x c %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL1.2 -fsyntax-only -verify -x cl %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -fsyntax-only -verify -x cl %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -cl-std=CL3.0 -fsyntax-only -verify -x cl %s

// expected-no-diagnostics

#define AS(N) __attribute__((address_space(N)))
#define OCL(Name) __attribute__((opencl_##Name))
#define CHECK(p)                                                               \
  (void)__builtin_amdgcn_is_shared(p);                                         \
  (void)__builtin_amdgcn_is_private(p)

#if defined(__HIP__)
__attribute__((device))
#endif
void test(void *flat, void AS(0) *as0, void AS(1) *as1, void AS(3) *as3,
          void AS(5) *as5, void OCL(global) *g, void OCL(local) *l,
          void OCL(private) *p, void OCL(constant) *c, void OCL(generic) *gen) {
  CHECK(flat);
  CHECK(as0);
  CHECK(as1);
  CHECK(as3);
  CHECK(as5);
  CHECK(g);
  CHECK(l);
  CHECK(p);
  CHECK(c);
  CHECK(gen);
}
