// RUN: %clang_cc1        -O3 -triple aarch64 -target-feature +sve -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK-C
// RUN: %clang_cc1 -x c++ -O3 -triple aarch64 -target-feature +sve -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK-CXX

typedef __SVFloat32_t fvec32 __attribute__((arm_sve_vector_bits(128)));

// PST containing an empty union: when compiled as C pass it in registers,
// when compiled as C++ - in memory.
typedef struct {
  fvec32 x[4];
  union {} u;
} S0;

#ifdef __cplusplus
extern "C"
#endif
void use0(S0);

void f0(S0 *p) {
  use0(*p);
}
// CHECK-C:   declare void @use0(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
// CHECK-CXX: declare void @use0(ptr dead_on_return noundef)

#ifdef __cplusplus

// PST containing an empty union with `[[no_unique_address]]` - pass in registers.
typedef struct {
   fvec32 x[4];
   [[no_unique_address]]
   union {} u;
} S1;

extern "C" void use1(S1);
void f1(S1 *p) {
  use1(*p);
}
// CHECK-CXX: declare void @use1(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)

#endif // __cplusplus
