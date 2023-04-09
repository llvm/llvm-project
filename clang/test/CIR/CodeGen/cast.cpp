// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

unsigned char cxxstaticcast_0(unsigned int x) {
  return static_cast<unsigned char>(x);
}

// CHECK: cir.func @_Z15cxxstaticcast_0j
// CHECK:    %0 = cir.alloca i32, cir.ptr <i32>, ["x", init] {alignment = 4 : i64}
// CHECK:    %1 = cir.alloca i8, cir.ptr <i8>, ["__retval"] {alignment = 1 : i64}
// CHECK:    cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK:    %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:    %3 = cir.cast(integral, %2 : i32), i8
// CHECK:    cir.store %3, %1 : i8, cir.ptr <i8>
// CHECK:    %4 = cir.load %1 : cir.ptr <i8>, i8
// CHECK:    cir.return %4 : i8
// CHECK:  }


int cStyleCasts_0(unsigned x1, int x2) {
// CHECK: cir.func @_{{.*}}cStyleCasts_0{{.*}}

  char a = (char)x1; // truncate
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : i32), i8

  short b = (short)x2; // truncate with sign
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : i32), i16

  long long c = (long long)x1; // zero extend
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : i32), i64

  long long d = (long long)x2; // sign extend
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : i32), i64

  int arr[3];
  int* e = (int*)arr; // explicit pointer decay
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %{{[0-9]+}} : !cir.ptr<!cir.array<i32 x 3>>), !cir.ptr<i32>

  return 0;
}
