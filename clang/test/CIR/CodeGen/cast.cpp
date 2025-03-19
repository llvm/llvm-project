// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -DCIR_ONLY %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

unsigned char cxxstaticcast_0(unsigned int x) {
  return static_cast<unsigned char>(x);
}

// CIR: cir.func @cxxstaticcast_0
// CIR:    %[[XPTR:[0-9]+]] = cir.alloca !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>, ["x", init] {alignment = 4 : i64}
// CIR:    cir.store %arg0, %[[XPTR]] : !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>
// CIR:    %[[XVAL:[0-9]+]] = cir.load %[[XPTR]] : !cir.ptr<!cir.int<u, 32>>, !cir.int<u, 32>
// CIR:    %[[CASTED:[0-9]+]] = cir.cast(integral, %[[XVAL]] : !cir.int<u, 32>), !cir.int<u, 8>
// CIR:    cir.return %[[CASTED]] : !cir.int<u, 8>
// CIR:  }

// LLVM: define i8 @cxxstaticcast_0(i32 %{{[0-9]+}})
// LLVM: %[[LOAD:[0-9]+]] = load i32, ptr %{{[0-9]+}}, align 4
// LLVM: %[[TRUNC:[0-9]+]] = trunc i32 %[[LOAD]] to i8
// LLVM: ret i8 %[[TRUNC]]


int cStyleCasts_0(unsigned x1, int x2, float x3, short x4, double x5) {
// CIR: cir.func @cStyleCasts_0
// LLVM: define i32 @cStyleCasts_0

  char a = (char)x1; // truncate
  // CIR: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<u, 32>), !cir.int<s, 8>
  // LLVM: %{{[0-9]+}} = trunc i32 %{{[0-9]+}} to i8

  short b = (short)x2; // truncate with sign
  // CIR: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<s, 32>), !cir.int<s, 16>
  // LLVM: %{{[0-9]+}} = trunc i32 %{{[0-9]+}} to i16

  long long c = (long long)x1; // zero extend
  // CIR: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<u, 32>), !cir.int<s, 64>
  // LLVM: %{{[0-9]+}} = zext i32 %{{[0-9]+}} to i64

  long long d = (long long)x2; // sign extend
  // CIR: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<s, 32>), !cir.int<s, 64>
  // LLVM: %{{[0-9]+}} = sext i32 %{{[0-9]+}} to i64

  unsigned ui = (unsigned)x2; // sign drop
  // CIR: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<s, 32>), !cir.int<u, 32>

  int si = (int)x1; // sign add
  // CIR: %{{[0-9]+}} = cir.cast(integral, %{{[0-9]+}} : !cir.int<u, 32>), !cir.int<s, 32>

  bool ib;
  int bi = (int)ib; // bool to int
  // CIR: %{{[0-9]+}} = cir.cast(bool_to_int, %{{[0-9]+}} : !cir.bool), !cir.int<s, 32>
  // LLVM: %{{[0-9]+}} = zext i1 %{{[0-9]+}} to i32

  #ifdef CIR_ONLY
  bool b2 = x2; // int to bool
  // CIR: %{{[0-9]+}} = cir.cast(int_to_bool, %{{[0-9]+}} : !cir.int<s, 32>), !cir.bool
  #endif

  #ifdef CIR_ONLY
  void *p;
   bool b3 = p; // ptr to bool
  // CIR: %{{[0-9]+}} = cir.cast(ptr_to_bool, %{{[0-9]+}} : !cir.ptr<!cir.void>), !cir.bool
  #endif

  float f;
  bool b4 = f; // float to bool
  // CIR: %{{[0-9]+}} = cir.cast(float_to_bool, %{{[0-9]+}} : !cir.float), !cir.bool
  // LLVM: %{{[0-9]+}} = fcmp une float %{{[0-9]+}}, 0.000000e+00
  // LLVM: %{{[0-9]+}} = zext i1 %{{[0-9]+}} to i8

  return 0;
}

#ifdef CIR_ONLY
bool cptr(void *d) {
  bool x = d;
  return x;
}

// CIR: cir.func @cptr(%arg0: !cir.ptr<!cir.void>
// CIR:   %[[DPTR:[0-9]+]] = cir.alloca !cir.ptr<!cir.void>, !cir.ptr<!cir.ptr<!cir.void>>, ["d", init] {alignment = 8 : i64}

// CIR:   %[[DVAL:[0-9]+]] = cir.load %[[DPTR]] : !cir.ptr<!cir.ptr<!cir.void>>, !cir.ptr<!cir.void>
// CIR:   %{{[0-9]+}} = cir.cast(ptr_to_bool, %[[DVAL]] : !cir.ptr<!cir.void>), !cir.bool
#endif

void should_not_cast() {
  unsigned x1;
  unsigned uu = (unsigned)x1; // identity

  bool x2;
  bool ib = (bool)x2; // identity
  
  (void) ib; // void cast
}

// CIR:     cir.func @should_not_cast
// CIR-NOT:   cir.cast
// CIR:     cir.return
