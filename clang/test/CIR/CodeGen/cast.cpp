// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

unsigned char cxxstaticcast_0(unsigned int x) {
  return static_cast<unsigned char>(x);
}

// CHECK: cir.func @cxxstaticcast_0
// CHECK:    %0 = cir.alloca !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>, ["x", init] {alignment = 4 : i64}
// CHECK:    cir.store %arg0, %0 : !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>
// CHECK:    %1 = cir.load %0 : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CHECK:    %2 = cir.cast(integral, %1 : !cir.int<u, 32>), !cir.int<u, 8>
// CHECK:    cir.return %2 : !cir.int<u, 8>
// CHECK:  }


int cStyleCasts_0(unsigned x1, int x2, float x3, short x4, double x5) {
// CHECK: cir.func @cStyleCasts_0

  char a = (char)x1; // truncate
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<u, 32>), !cir.int<s, 8>

  short b = (short)x2; // truncate with sign
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<s, 32>), !cir.int<s, 16>

  long long c = (long long)x1; // zero extend
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<u, 32>), !cir.int<s, 64>

  long long d = (long long)x2; // sign extend
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<s, 32>), !cir.int<s, 64>

  unsigned ui = (unsigned)x2; // sign drop
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<s, 32>), !cir.int<u, 32>

  int si = (int)x1; // sign add
  // CHECK: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<u, 32>), !cir.int<s, 32>

  bool ib;
  int bi = (int)ib; // bool to int
  // CHECK: %{{[0-9]+}} = cir.cast(bool_to_int, %{{[0-9]+}} : !cir.bool), !cir.int<s, 32>

  return 0;
}

bool cptr(void *d) {
  bool x = d;
  return x;
}

// CHECK: cir.func @cptr(%arg0: !cir.ptr<!cir.void>
// CHECK:   %0 = cir.alloca !cir.ptr<!cir.void>, !cir.ptr<!cir.ptr<!cir.void>>, ["d", init] {alignment = 8 : i64}

// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.ptr<!cir.void>>, !cir.ptr<!cir.void>
// CHECK:   %3 = cir.cast(ptr_to_bool, %2 : !cir.ptr<!cir.void>), !cir.bool

void should_not_cast() {
  unsigned x1;

  unsigned uu = (unsigned)x1;
  bool ib = (bool)x1;
  return (void) x1;
}

// CHECK:     cir.func @should_not_cast
// CHECK-NOT:   cir.cast
// CHECK:     }
