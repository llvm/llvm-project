// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

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

// CHECK-LABEL: define {{.*}}void @test_store_data_i_constant()
void test_store_data_i_constant() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @external_int to i64), i32 1, i64 50)
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to ptr
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * IQ iqpi = &external_int;
// CHECK-NEXT:    [[T0:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @external_int to i64), i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T0]] to ptr
// CHECK-NEXT:    store ptr [[SIGNED]], ptr [[V]],
// CHECK-NEXT:    ret void
  iqpi = &external_int;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_iu()
void test_store_data_iu() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_upi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * IQ iqpi = global_upi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_upi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  iqpi = global_upi;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_ia()
void test_store_data_ia() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * IQ iqpi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  iqpi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[RESULT:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[RESULT]], ptr [[V]],
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[RESULT]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[RESULT]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[RESULT:%.*]] = phi ptr [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upi(ptr noundef [[RESULT]])
  use_upi(iqpi = global_aqpi);
}

// CHECK-LABEL: define {{.*}}void @test_store_data_ii_same()
void test_store_data_ii_same() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    store ptr [[LOAD]], ptr [[V]],
  int * IQ iqpi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    store ptr [[LOAD]], ptr [[V]],
  iqpi = global_iqpi;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_ii_different()
void test_store_data_ii_different() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 100)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * DIFF_IQ iqpi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 100)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  iqpi = global_iqpi;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_ii_zero()
void test_store_data_ii_zero() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 0)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * ZERO_IQ iqpi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr [[V]]
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 0, i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr @global_iqpi,
  global_iqpi = iqpi;
}

// CHECK-LABEL: define {{.*}}void @test_load_data_i()
void test_load_data_i() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int *upi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  upi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T0]], i32 1, i64 50)
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upi(ptr noundef [[T0]])
  use_upi(global_iqpi);
}

// Data with address-discriminated qualifiers.

// CHECK-LABEL: define {{.*}}void @test_store_data_a_constant()
void test_store_data_a_constant() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @external_int to i64), i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to ptr
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * AQ aqpi = &external_int;
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @external_int to i64), i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to ptr
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpi = &external_int;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_au()
void test_store_data_au() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_upi,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[T0]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * AQ aqpi = global_upi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_upi,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[T0]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpi = global_upi;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_ai()
void test_store_data_ai() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * AQ aqpi = global_iqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpi,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpi = global_iqpi;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_aa_same()
void test_store_data_aa_same() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * AQ aqpi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpi = global_aqpi;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_aa_different()
void test_store_data_aa_different() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 100)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * DIFF_AQ aqpi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 100)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpi = global_aqpi;
}

// CHECK-LABEL: define {{.*}}void @test_store_data_aa_zero()
void test_store_data_aa_zero() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[NEWDISC:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int * ZERO_AQ aqpi = global_aqpi;
// CHECK:         [[LOAD:%.*]] = load ptr, ptr [[V]],
// CHECK-NEXT:    [[OLDDISC:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr @global_aqpi,
  global_aqpi = aqpi;
}

// CHECK-LABEL: define {{.*}}void @test_load_data_a()
void test_load_data_a() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T0]], i32 1, i64 [[OLDDISC]])
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  int *upi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T0]], i32 1, i64 [[OLDDISC]])
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  upi = global_aqpi;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpi,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpi to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T0]], i32 1, i64 [[OLDDISC]])
// CHECK-NEXT:    [[AUTHED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[AUTHED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upi(ptr noundef [[T0]])
  use_upi(global_aqpi);
}

// Function with address-independent qualifiers.

// CHECK-LABEL: define {{.*}}void @test_store_function_i_constant()
void test_store_function_i_constant() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @external_func, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 1, i64 50)
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to ptr
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * IQ iqpf = &external_func;
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @external_func, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 1, i64 50)
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to ptr
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  iqpf = &external_func;
}

// CHECK-LABEL: define {{.*}}void @test_store_function_iu()
void test_store_function_iu() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_upf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 0, i64 18983, i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * IQ iqpf = global_upf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_upf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 0, i64 18983, i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  iqpf = global_upf;
}

// CHECK-LABEL: define {{.*}}void @test_store_function_ia()
void test_store_function_ia() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * IQ iqpf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  iqpf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 50)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[RESULT:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[RESULT]], ptr [[V]],
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[RESULT]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[RESULT]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 0, i64 18983)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upf(ptr noundef [[T0]])
  use_upf(iqpf = global_aqpf);
}

// CHECK-LABEL: define {{.*}}void @test_store_function_ii_same()
void test_store_function_ii_same() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    store ptr [[LOAD]], ptr [[V]],
  func_t * IQ iqpf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    store ptr [[LOAD]], ptr [[V]],
  iqpf = global_iqpf;
}

// CHECK-LABEL: define {{.*}}void @test_store_function_ii_different()
void test_store_function_ii_different() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 100)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * DIFF_IQ iqpf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 100)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  iqpf = global_iqpf;
}

// CHECK-LABEL: define {{.*}}void @test_load_function_i()
void test_load_function_i() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 0, i64 18983)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t *upf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 0, i64 18983)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  upf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 0, i64 18983)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upf(ptr noundef [[T0]])
  use_upf(global_iqpf);
}

// Function with address-discriminated qualifiers.

// CHECK-LABEL: define {{.*}}void @test_store_function_a_constant()
void test_store_function_a_constant() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @external_func, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to ptr
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * AQ aqpf = &external_func;
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[SIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @external_func, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[T0:%.*]] = inttoptr i64 [[SIGN]] to ptr
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpf = &external_func;
}

// CHECK-LABEL: define {{.*}}void @test_store_function_au()
void test_store_function_au() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_upf,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 0, i64 18983, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * AQ aqpf = global_upf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_upf,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 0, i64 18983, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpf = global_upf;
}

// CHECK-LABEL: define {{.*}}void @test_store_function_ai()
void test_store_function_ai() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * AQ aqpf = global_iqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_iqpf,
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 50, i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpf = global_iqpf;
}

// CHECK-LABEL: define {{.*}}void @test_store_function_aa_same()
void test_store_function_aa_same() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * AQ aqpf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpf = global_aqpf;
}

// CHECK-LABEL: define {{.*}}void @test_store_function_aa_different()
void test_store_function_aa_different() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 100)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t * DIFF_AQ aqpf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = ptrtoint ptr [[V]] to i64
// CHECK-NEXT:    [[NEWDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T0]], i64 100)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 1, i64 [[NEWDISC]])
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  aqpf = global_aqpf;
}

// CHECK-LABEL: define {{.*}}void @test_load_function_a()
void test_load_function_a() {
// CHECK:         [[V:%.*]] = alloca ptr,
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 0, i64 18983)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  func_t *upf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 0, i64 18983)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    store ptr [[T0]], ptr [[V]],
  upf = global_aqpf;
// CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr @global_aqpf,
// CHECK-NEXT:    [[OLDDISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @global_aqpf to i64), i64 50)
// CHECK-NEXT:    [[T0:%.*]] = icmp ne ptr [[LOAD]], null
// CHECK-NEXT:    br i1 [[T0]],
// CHECK:         [[T0:%.*]] = ptrtoint ptr [[LOAD]] to i64
// CHECK-NEXT:    [[T1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[T0]], i32 1, i64 [[OLDDISC]], i32 0, i64 18983)
// CHECK-NEXT:    [[SIGNED:%.*]] = inttoptr i64 [[T1]] to ptr
// CHECK-NEXT:    br label
// CHECK:         [[T0:%.*]] = phi ptr [ null, {{.*}} ], [ [[SIGNED]], {{.*}} ]
// CHECK-NEXT:    call void @use_upf(ptr noundef [[T0]])
  use_upf(global_aqpf);
}
