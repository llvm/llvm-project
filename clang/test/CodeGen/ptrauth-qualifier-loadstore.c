// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

#define IQ __ptrauth(1,0,50)
#define AQ __ptrauth(1,1,50)
#define DIFF_IQ __ptrauth(1,0,100)
#define DIFF_AQ __ptrauth(1,1,100)
#define ZERO_IQ __ptrauth(1,0,0)
#define ZERO_AQ __ptrauth(1,1,0)

extern int external_int;
extern int * global_upi;
extern int * IQ global_iqpi;
extern int * AQ global_aqpi;
extern void use_upi(int *ptr);

typedef void func_t(void);
extern void external_func(void);
extern func_t *global_upf;
extern func_t * IQ global_iqpf;
extern func_t * AQ global_aqpf;
extern void use_upf(func_t *ptr);

// Data with address-independent qualifiers.

// CHECK-LABEL: define void @test_store_data_i_constant()
void test_store_data_i_constant() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i32* @external_int to i64), i32 1, i64 50)
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to i32*
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * IQ iqpi = &external_int;
// CHECK-NEXT:    [[T0:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i32* @external_int to i64), i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T0]] to i32*
// CHECK-NEXT:    store i32* [[SIGNED]], i32** [[V]],
// CHECK-NEXT:    ret void
  iqpi = &external_int;
}

// CHECK-LABEL: define void @test_store_data_iu()
void test_store_data_iu() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_upi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * IQ iqpi = global_upi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_upi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  iqpi = global_upi;
}

// CHECK-LABEL: define void @test_store_data_ia()
void test_store_data_ia() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * IQ iqpi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  iqpi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[RESULT:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[RESULT]], i32** [[V]],
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[RESULT]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[RESULT]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[RESULT:%.*]] = phi i32* [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upi(i32* [[RESULT]])
  use_upi(iqpi = global_aqpi);
}

// CHECK-LABEL: define void @test_store_data_ii_same()
void test_store_data_ii_same() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    store i32* [[LOAD]], i32** [[V]],
  int * IQ iqpi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    store i32* [[LOAD]], i32** [[V]],
  iqpi = global_iqpi;
}

// CHECK-LABEL: define void @test_store_data_ii_different()
void test_store_data_ii_different() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 100)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * DIFF_IQ iqpi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 100)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  iqpi = global_iqpi;
}

// CHECK-LABEL: define void @test_store_data_ii_zero()
void test_store_data_ii_zero() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * ZERO_IQ iqpi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** [[V]]
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 0, i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** @global_iqpi,
  global_iqpi = iqpi;
}

// CHECK-LABEL: define void @test_load_data_i()
void test_load_data_i() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int *upi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  upi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upi(i32* [[T0]])
  use_upi(global_iqpi);
}

// Data with address-discriminated qualifiers.

// CHECK-LABEL: define void @test_store_data_a_constant()
void test_store_data_a_constant() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i32* @external_int to i64), i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to i32*
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * AQ aqpi = &external_int;
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 ptrtoint (i32* @external_int to i64), i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to i32*
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  aqpi = &external_int;
}

// CHECK-LABEL: define void @test_store_data_au()
void test_store_data_au() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_upi,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 [[T0]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * AQ aqpi = global_upi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_upi,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 [[T0]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  aqpi = global_upi;
}

// CHECK-LABEL: define void @test_store_data_ai()
void test_store_data_ai() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * AQ aqpi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  aqpi = global_iqpi;
}

// CHECK-LABEL: define void @test_store_data_aa_same()
void test_store_data_aa_same() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * AQ aqpi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  aqpi = global_aqpi;
}

// CHECK-LABEL: define void @test_store_data_aa_different()
void test_store_data_aa_different() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 100)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * DIFF_AQ aqpi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 100)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  aqpi = global_aqpi;
}

// CHECK-LABEL: define void @test_store_data_aa_zero()
void test_store_data_aa_zero() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[NEWDISC:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int * ZERO_AQ aqpi = global_aqpi;
// CHECK:         [[LOAD:%.*]] = load i32*, i32** [[V]],
// CHECK-NEXT:    [[OLDDISC:%.*]] = ptrtoint i32** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** @global_aqpi,
  global_aqpi = aqpi;
}

// CHECK-LABEL: define void @test_load_data_a()
void test_load_data_a() {
// CHECK:         [[V:%.*]] = alloca i32*,
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]])
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  int *upi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]])
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    store i32* [[T0]], i32** [[V]],
  upi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load i32*, i32** @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i32** @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne i32* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint i32* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]])
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to i32*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi i32* [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upi(i32* [[T0]])
  use_upi(global_aqpi);
}

// Function with address-independent qualifiers.

// CHECK-LABEL: define void @test_store_function_i_constant()
void test_store_function_i_constant() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 ptrtoint ({{.*}} @external_func.ptrauth to i64), i32 0, i64 0, i32 1, i64 50)
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to void ()*
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * IQ iqpf = &external_func;
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 ptrtoint ({{.*}} @external_func.ptrauth to i64), i32 0, i64 0, i32 1, i64 50)
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to void ()*
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  iqpf = &external_func;
}

// CHECK-LABEL: define void @test_store_function_iu()
void test_store_function_iu() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_upf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 0, i64 0, i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * IQ iqpf = global_upf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_upf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 0, i64 0, i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  iqpf = global_upf;
}

// CHECK-LABEL: define void @test_store_function_ia()
void test_store_function_ia() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * IQ iqpf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  iqpf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[RESULT:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[RESULT]], void ()** [[V]],
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[RESULT]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[RESULT]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 0, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upf(void ()* [[T0]])
  use_upf(iqpf = global_aqpf);
}

// CHECK-LABEL: define void @test_store_function_ii_same()
void test_store_function_ii_same() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    store void ()* [[LOAD]], void ()** [[V]],
  func_t * IQ iqpf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    store void ()* [[LOAD]], void ()** [[V]],
  iqpf = global_iqpf;
}

// CHECK-LABEL: define void @test_store_function_ii_different()
void test_store_function_ii_different() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 100)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * DIFF_IQ iqpf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 100)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  iqpf = global_iqpf;
}

// CHECK-LABEL: define void @test_load_function_i()
void test_load_function_i() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 0, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t *upf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 0, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  upf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 0, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upf(void ()* [[T0]])
  use_upf(global_iqpf);
}

// Function with address-discriminated qualifiers.

// CHECK-LABEL: define void @test_store_function_a_constant()
void test_store_function_a_constant() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 ptrtoint ({{.*}} @external_func.ptrauth to i64), i32 0, i64 0, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to void ()*
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * AQ aqpf = &external_func;
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 ptrtoint ({{.*}} @external_func.ptrauth to i64), i32 0, i64 0, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to void ()*
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  aqpf = &external_func;
}

// CHECK-LABEL: define void @test_store_function_au()
void test_store_function_au() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_upf,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 0, i64 0, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * AQ aqpf = global_upf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_upf,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 0, i64 0, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  aqpf = global_upf;
}

// CHECK-LABEL: define void @test_store_function_ai()
void test_store_function_ai() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * AQ aqpf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 50, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  aqpf = global_iqpf;
}

// CHECK-LABEL: define void @test_store_function_aa_same()
void test_store_function_aa_same() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * AQ aqpf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  aqpf = global_aqpf;
}

// CHECK-LABEL: define void @test_store_function_aa_different()
void test_store_function_aa_different() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 100)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t * DIFF_AQ aqpf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint void ()** [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 100)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  aqpf = global_aqpf;
}

// CHECK-LABEL: define void @test_load_function_a()
void test_load_function_a() {
// CHECK:         [[V:%.*]] = alloca void ()*,
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 0, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  func_t *upf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 0, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store void ()* [[T0]], void ()** [[V]],
  upf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load void ()*, void ()** @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (void ()** @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne void ()* [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint void ()* [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 0, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to void ()*
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi void ()* [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upf(void ()* [[T0]])
  use_upf(global_aqpf);
}
