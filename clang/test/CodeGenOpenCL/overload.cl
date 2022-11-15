// RUN: %clang_cc1 -no-opaque-pointers -cl-std=CL2.0 -emit-llvm -o - -triple spir-unknown-unknown %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -cl-std=CL3.0 -cl-ext=+__opencl_c_generic_address_space -emit-llvm -o - -triple spir-unknown-unknown %s | FileCheck %s

typedef short short4 __attribute__((ext_vector_type(4)));

// CHECK-DAG: declare spir_func <4 x i16> @_Z5clampDv4_sS_S_(<4 x i16> noundef, <4 x i16> noundef, <4 x i16> noundef)
short4 __attribute__ ((overloadable)) clamp(short4 x, short4 minval, short4 maxval);
// CHECK-DAG: declare spir_func <4 x i16> @_Z5clampDv4_sss(<4 x i16> noundef, i16 noundef signext, i16 noundef signext)
short4 __attribute__ ((overloadable)) clamp(short4 x, short minval, short maxval);
void __attribute__((overloadable)) foo(global int *a, global int *b);
void __attribute__((overloadable)) foo(generic int *a, generic int *b);
void __attribute__((overloadable)) bar(generic int *global *a, generic int *global *b);
void __attribute__((overloadable)) bar(generic int *generic *a, generic int *generic *b);

// Checking address space resolution
void kernel test1() {
  global int *a = 0;
  global int *b = 0;
  generic int *c = 0;
  local int *d = 0;
  generic int *generic *gengen = 0;
  generic int *local *genloc = 0;
  generic int *global *genglob = 0;
  // CHECK-DAG: call spir_func void @_Z3fooPU3AS1iS0_(i32 addrspace(1)* noundef {{.*}}, i32 addrspace(1)* noundef {{.*}})
  foo(a, b);
  // CHECK-DAG: call spir_func void @_Z3fooPU3AS4iS0_(i32 addrspace(4)* noundef {{.*}}, i32 addrspace(4)* noundef {{.*}})
  foo(b, c);
  // CHECK-DAG: call spir_func void @_Z3fooPU3AS4iS0_(i32 addrspace(4)* noundef {{.*}}, i32 addrspace(4)* noundef {{.*}})
  foo(a, d);

  // CHECK-DAG: call spir_func void @_Z3barPU3AS4PU3AS4iS2_(i32 addrspace(4)* addrspace(4)* noundef {{.*}}, i32 addrspace(4)* addrspace(4)* noundef {{.*}})
  bar(gengen, genloc);
  // CHECK-DAG: call spir_func void @_Z3barPU3AS4PU3AS4iS2_(i32 addrspace(4)* addrspace(4)* noundef {{.*}}, i32 addrspace(4)* addrspace(4)* noundef {{.*}})
  bar(gengen, genglob);
  // CHECK-DAG: call spir_func void @_Z3barPU3AS1PU3AS4iS2_(i32 addrspace(4)* addrspace(1)* noundef {{.*}}, i32 addrspace(4)* addrspace(1)* noundef {{.*}})
  bar(genglob, genglob);
}

// Checking vector vs scalar resolution
void kernel test2() {
  short4 e0=0;

  // CHECK-DAG: call spir_func <4 x i16> @_Z5clampDv4_sss(<4 x i16> noundef {{.*}}, i16 noundef signext 0, i16 noundef signext 255)
  clamp(e0, 0, 255);
  // CHECK-DAG: call spir_func <4 x i16> @_Z5clampDv4_sS_S_(<4 x i16> noundef {{.*}}, <4 x i16> noundef {{.*}}, <4 x i16> noundef {{.*}})
  clamp(e0, e0, e0);
}
