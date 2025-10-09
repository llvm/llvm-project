// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

unsigned char cxxstaticcast_0(unsigned int x) {
  return static_cast<unsigned char>(x);
}

// CIR: cir.func{{.*}} @_Z15cxxstaticcast_0j
// CIR:    %[[XPTR:[0-9]+]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["x", init] {alignment = 4 : i64}
// CIR:    %[[RV:[0-9]+]] = cir.alloca !u8i, !cir.ptr<!u8i>, ["__retval"] {alignment = 1 : i64}
// CIR:    cir.store %arg0, %[[XPTR]] : !u32i, !cir.ptr<!u32i>
// CIR:    %[[XVAL:[0-9]+]] = cir.load{{.*}} %[[XPTR]] : !cir.ptr<!u32i>, !u32i
// CIR:    %[[CASTED:[0-9]+]] = cir.cast integral %[[XVAL]] : !u32i -> !u8i
// CIR:    cir.store %[[CASTED]], %[[RV]] : !u8i, !cir.ptr<!u8i>
// CIR:    %[[R:[0-9]+]] = cir.load{{.*}} %1 : !cir.ptr<!u8i>, !u8i
// CIR:    cir.return %[[R]] : !u8i
// CIR:  }

// LLVM: define{{.*}} i8 @_Z15cxxstaticcast_0j(i32 %{{[0-9]+}})
// LLVM: %[[LOAD:[0-9]+]] = load i32, ptr %{{[0-9]+}}, align 4
// LLVM: %[[TRUNC:[0-9]+]] = trunc i32 %[[LOAD]] to i8
// LLVM: store i8 %[[TRUNC]], ptr %[[RV:[0-9]+]], align 1
// LLVM: %[[R:[0-9]+]] = load i8, ptr %[[RV]], align 1
// LLVM: ret i8 %[[R]]

int cStyleCasts_0(unsigned x1, int x2, float x3, short x4, double x5) {
// CIR: cir.func{{.*}} @_Z13cStyleCasts_0jifsd
// LLVM: define{{.*}} i32 @_Z13cStyleCasts_0jifsd

  char a = (char)x1; // truncate
  // CIR: %{{[0-9]+}} = cir.cast integral %{{[0-9]+}} : !u32i -> !s8i
  // LLVM: %{{[0-9]+}} = trunc i32 %{{[0-9]+}} to i8

  short b = (short)x2; // truncate with sign
  // CIR: %{{[0-9]+}} = cir.cast integral %{{[0-9]+}} : !s32i -> !s16i
  // LLVM: %{{[0-9]+}} = trunc i32 %{{[0-9]+}} to i16

  long long c = (long long)x1; // zero extend
  // CIR: %{{[0-9]+}} = cir.cast integral %{{[0-9]+}} : !u32i -> !s64i
  // LLVM: %{{[0-9]+}} = zext i32 %{{[0-9]+}} to i64

  long long d = (long long)x2; // sign extend
  // CIR: %{{[0-9]+}} = cir.cast integral %{{[0-9]+}} : !s32i -> !s64i
  // LLVM: %{{[0-9]+}} = sext i32 %{{[0-9]+}} to i64

  unsigned ui = (unsigned)x2; // sign drop
  // CIR: %{{[0-9]+}} = cir.cast integral %{{[0-9]+}} : !s32i -> !u32i

  int si = (int)x1; // sign add
  // CIR: %{{[0-9]+}} = cir.cast integral %{{[0-9]+}} : !u32i -> !s32i

  bool ib;
  int bi = (int)ib; // bool to int
  // CIR: %{{[0-9]+}} = cir.cast bool_to_int %{{[0-9]+}} : !cir.bool -> !s32i
  // LLVM: %{{[0-9]+}} = zext i1 %{{[0-9]+}} to i32

  bool b2 = x2; // int to bool
  // CIR: %{{[0-9]+}} = cir.cast int_to_bool %{{[0-9]+}} : !s32i -> !cir.bool
  // LLVM: %[[INTTOBOOL:[0-9]+]]  = icmp ne i32 %{{[0-9]+}}, 0
  // LLVM: zext i1 %[[INTTOBOOL]] to i8

  void *p;
  bool b3 = p; // ptr to bool
  // CIR: %{{[0-9]+}} = cir.cast ptr_to_bool %{{[0-9]+}} : !cir.ptr<!void> -> !cir.bool
  // LLVM: %[[PTRTOBOOL:[0-9]+]]  = icmp ne ptr %{{[0-9]+}}, null
  // LLVM: zext i1 %[[PTRTOBOOL]] to i8

  float f;
  bool b4 = f; // float to bool
  // CIR: %{{[0-9]+}} = cir.cast float_to_bool %{{[0-9]+}} : !cir.float -> !cir.bool
  // LLVM: %{{[0-9]+}} = fcmp une float %{{[0-9]+}}, 0.000000e+00
  // LLVM: %{{[0-9]+}} = zext i1 %{{[0-9]+}} to i8

  double d2 = f; // float to double
  // CIR: %{{[0-9]+}} = cir.cast floating %{{[0-9]+}} : !cir.float -> !cir.double
  // LLVM: %{{[0-9]+}} = fpext float %{{[0-9]+}} to double

  f = d2; // double to float
  // CIR: %{{[0-9]+}} = cir.cast floating %{{[0-9]+}} : !cir.double -> !cir.float
  // LLVM: %{{[0-9]+}} = fptrunc double %{{[0-9]+}} to float

  return 0;
}

bool cptr(void *d) {
  bool x = d;
  return x;
}

// CIR: cir.func{{.*}} @_Z4cptrPv(%arg0: !cir.ptr<!void>
// CIR:   %[[DPTR:[0-9]+]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["d", init] {alignment = 8 : i64}

// CIR:   %[[DVAL:[0-9]+]] = cir.load{{.*}} %[[DPTR]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:   %{{[0-9]+}} = cir.cast ptr_to_bool %[[DVAL]] : !cir.ptr<!void> -> !cir.bool

// LLVM-LABEL: define{{.*}} i1 @_Z4cptrPv(ptr %0)
// LLVM:         %[[ARG_STORAGE:.*]] = alloca ptr, i64 1
// LLVM:         %[[RETVAL:.*]] = alloca i8, i64 1
// LLVM:         %[[X_STORAGE:.*]] = alloca i8, i64 1
// LLVM:         store ptr %0, ptr %[[ARG_STORAGE]]
// LLVM:         %[[LOADED_PTR:.*]] = load ptr, ptr %[[ARG_STORAGE]]
// LLVM:         %[[NULL_CHECK:.*]] = icmp ne ptr %[[LOADED_PTR]], null
// LLVM:         ret i1

void should_not_cast() {
  unsigned x1;
  unsigned uu = (unsigned)x1; // identity

  bool x2;
  bool ib = (bool)x2; // identity

  (void) ib; // void cast
}

// CIR:     cir.func{{.*}} @_Z15should_not_castv
// CIR-NOT:   cir.cast
// CIR:     cir.return

typedef int vi4 __attribute__((vector_size(16)));
typedef double vd2 __attribute__((vector_size(16)));

void bitcast() {
  vd2 a = {};
  vi4 b = (vi4)a;
}

// CIR: %[[D_VEC:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double>
// CIR: %[[I_VEC:.*]] = cir.cast bitcast %[[D_VEC]] : !cir.vector<2 x !cir.double> -> !cir.vector<4 x !s32i>

// LLVM: %[[D_VEC:.*]] = load <2 x double>, ptr {{.*}}, align 16
// LLVM: %[[I_VEC:.*]] = bitcast <2 x double> %[[D_VEC]] to <4 x i32>

void f(long int start) {
  void *p = (void*)start;
}
// CIR: %[[L:.*]] = cir.load {{.*}} : !cir.ptr<!s64i>, !s64i
// CIR: %[[MID:.*]] = cir.cast integral %[[L]] : !s64i -> !u64i
// CIR:          cir.cast int_to_ptr %[[MID]] : !u64i -> !cir.ptr<!void>

// LLVM-LABEL: define{{.*}} void @_Z1fl(i64 %0)
// LLVM: %[[ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[PADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: store i64 %0, ptr %[[ADDR]], align 8
// LLVM: %[[L:.*]] = load i64, ptr %[[ADDR]], align 8
// LLVM: %[[PTR:.*]] = inttoptr i64 %[[L]] to ptr
// LLVM: store ptr %[[PTR]], ptr %[[PADDR]], align 8
// LLVM: ret void

struct A { int x; };

void int_cast(long ptr) {
  ((A *)ptr)->x = 0;
}
// CIR: cir.cast int_to_ptr {{.*}} : !u64i -> !cir.ptr<!rec_A>
// LLVM: inttoptr {{.*}} to ptr

void null_cast(long) {
  *(int *)0 = 0;
  ((A *)0)->x = 0;
}
// CIR:    %[[NULLPTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR:    cir.store{{.*}} %{{.*}}, %[[NULLPTR]] : !s32i, !cir.ptr<!s32i>
// CIR:    %[[NULLPTR_A:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_A>
// CIR:    %[[A_X:.*]] = cir.get_member %[[NULLPTR_A]][0] {name = "x"} : !cir.ptr<!rec_A> -> !cir.ptr<!s32i>
