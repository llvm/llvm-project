// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

unsigned char cxxstaticcast_0(unsigned int x) {
  return static_cast<unsigned char>(x);
}

// CHECK: cir.func @_Z15cxxstaticcast_0j
// CHECK:    %0 = cir.alloca !u32i, cir.ptr <!u32i>, ["x", init] {alignment = 4 : i64}
// CHECK:    %1 = cir.alloca !u8i, cir.ptr <!u8i>, ["__retval"] {alignment = 1 : i64}
// CHECK:    cir.store %arg0, %0 : !u32i, cir.ptr <!u32i>
// CHECK:    %2 = cir.load %0 : cir.ptr <!u32i>, !u32i
// CHECK:    %3 = cir.cast(integral, %2 : !u32i), !u8i
// CHECK:    cir.store %3, %1 : !u8i, cir.ptr <!u8i>
// CHECK:    %4 = cir.load %1 : cir.ptr <!u8i>, !u8i
// CHECK:    cir.return %4 : !u8i
// CHECK:  }


int cStyleCasts_0(unsigned x1, int x2) {
// CHECK: cir.func @_{{.*}}cStyleCasts_0{{.*}}

  char a = (char)x1; // truncate
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !u32i), !s8i

  short b = (short)x2; // truncate with sign
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !s32i), !s16i

  long long c = (long long)x1; // zero extend
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !u32i), !s64i

  long long d = (long long)x2; // sign extend
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !s32i), !s64i

  int arr[3];
  int* e = (int*)arr; // explicit pointer decay
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %{{[0-9]+}} : !cir.ptr<!cir.array<!s32i x 3>>), !cir.ptr<!s32i>

  return 0;
}

bool cptr(void *d) {
  bool x = d;
  return x;
}

// CHECK: cir.func @_Z4cptrPv(%arg0: !cir.ptr<i8>
// CHECK:   %0 = cir.alloca !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>, ["d", init] {alignment = 8 : i64}

// CHECK:   %3 = cir.load %0 : cir.ptr <!cir.ptr<i8>>, !cir.ptr<i8>
// CHECK:   %4 = cir.cast(ptr_to_bool, %3 : !cir.ptr<i8>), !cir.bool

void call_cptr(void *d) {
  if (!cptr(d)) {
  }
}

// CHECK: cir.func @_Z9call_cptrPv(%arg0: !cir.ptr<i8>
// CHECK:   %0 = cir.alloca !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>, ["d", init] {alignment = 8 : i64}

// CHECK:   cir.scope {
// CHECK:     %1 = cir.load %0 : cir.ptr <!cir.ptr<i8>>, !cir.ptr<i8>
// CHECK:     %2 = cir.call @_Z4cptrPv(%1) : (!cir.ptr<i8>) -> !cir.bool
// CHECK:     %3 = cir.unary(not, %2) : !cir.bool, !cir.bool
// CHECK:     cir.if %3 {