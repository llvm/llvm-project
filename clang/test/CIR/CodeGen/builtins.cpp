// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -emit-cir %s -o %t.cir  
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir \
// RUN:  -emit-llvm -fno-clangir-call-conv-lowering -o - %s \
// RUN:  | opt -S -passes=instcombine,mem2reg,simplifycfg -o %t.ll 
// RUN: FileCheck  --check-prefix=LLVM --input-file=%t.ll %s

// This test file is a collection of test cases for all target-independent
// builtins that are related to memory operations.

int s;

int *test_addressof() {
  return __builtin_addressof(s);
  
  // CIR-LABEL: test_addressof
  // CIR: [[ADDR:%.*]] = cir.get_global @s : !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  // LLVM-LABEL: test_addressof
  // LLVM: store ptr @s, ptr [[ADDR:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[ADDR]], align 8
  // LLVM: ret ptr [[RES]]
}

namespace std { template<typename T> T *addressof(T &); }
int *test_std_addressof() {
  return std::addressof(s);
  
  // CIR-LABEL: test_std_addressof
  // CIR: [[ADDR:%.*]] = cir.get_global @s : !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  // LLVM-LABEL: test_std_addressof
  // LLVM: store ptr @s, ptr [[ADDR:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[ADDR]], align 8
  // LLVM: ret ptr [[RES]]
}

namespace std { template<typename T> T *__addressof(T &); }
int *test_std_addressof2() {
  return std::__addressof(s);
  
  // CIR-LABEL: test_std_addressof2
  // CIR: [[ADDR:%.*]] = cir.get_global @s : !cir.ptr<!s32i>
  // CIR: cir.store [[ADDR]], [[SAVE:%.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: [[RES:%.*]] = cir.load [[SAVE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: cir.return [[RES]] : !cir.ptr<!s32i>

  /// LLVM-LABEL: test_std_addressof2
  // LLVM: store ptr @s, ptr [[ADDR:%.*]], align 8
  // LLVM: [[RES:%.*]] = load ptr, ptr [[ADDR]], align 8
  // LLVM: ret ptr [[RES]]
}

extern "C" char* test_memchr(const char arg[32]) {
  return __builtin_char_memchr(arg, 123, 32);

  // CIR-LABEL: test_memchr
  // CIR: [[PATTERN:%.*]] = cir.const #cir.int<123> : !s32i 
  // CIR: [[LEN:%.*]] = cir.const #cir.int<32> : !s32i 
  // CIR: [[LEN_U64:%.*]] = cir.cast(integral, [[LEN]] : !s32i), !u64i 
  // CIR: {{%.*}} = cir.libc.memchr({{%.*}}, [[PATTERN]], [[LEN_U64]])

  // LLVM: {{.*}}@test_memchr(ptr{{.*}}[[ARG:%.*]]) 
  // LLVM: [[TMP0:%.*]] = alloca ptr, i64 1, align 8
  // LLVM: store ptr [[ARG]], ptr [[TMP0]], align 8
  // LLVM: [[SRC:%.*]] = load ptr, ptr [[TMP0]], align 8
  // LLVM: [[RES:%.*]] = call ptr @memchr(ptr [[SRC]], i32 123, i64 32)
  // LLVM: store ptr [[RES]], ptr [[RET_P:%.*]], align 8
  // LLVM: [[RET:%.*]] = load ptr, ptr [[RET_P]], align 8
  // LLVM: ret ptr [[RET]]
}

extern "C"  wchar_t* test_wmemchr(const wchar_t *wc) {
  return __builtin_wmemchr(wc, 257u, 32);

  // CIR-LABEL: test_wmemchr
  // CIR: [[PATTERN:%.*]] = cir.const #cir.int<257> : !u32i 
  // CIR: [[LEN:%.*]] = cir.const #cir.int<32> : !s32i 
  // CIR: [[LEN_U64:%.*]] = cir.cast(integral, [[LEN]] : !s32i), !u64i 
  // CIR: cir.call @wmemchr({{%.*}}, [[PATTERN]], [[LEN_U64]]) : (!cir.ptr<!u32i>, !u32i, !u64i) -> !cir.ptr<!u32i>

  // LLVM: {{.*}}@test_wmemchr(ptr{{.*}}[[ARG:%.*]])
  // LLVM: [[TMP0:%.*]] = alloca ptr, i64 1, align 8
  // LLVM: store ptr [[ARG]], ptr [[TMP0]], align 8
  // LLVM: [[SRC:%.*]] = load ptr, ptr [[TMP0]], align 8
  // LLVM: [[RES:%.*]] = call ptr @wmemchr(ptr [[SRC]], i32 257, i64 32)
  // LLVM: store ptr [[RES]], ptr [[RET_P:%.*]], align 8
  // LLVM: [[RET:%.*]] = load ptr, ptr [[RET_P]], align 8
  // LLVM: ret ptr [[RET]]
}

extern "C" void *test_return_address(void) {
  return __builtin_return_address(1);

  // CIR-LABEL: test_return_address
  // CIR: [[ARG:%.*]] = cir.const #cir.int<1> : !u32i
  // CIR: {{%.*}} = cir.return_address([[ARG]])

  // LLVM-LABEL: @test_return_address
  // LLVM: {{%.*}} = call ptr @llvm.returnaddress(i32 1)
}

extern "C" void *test_frame_address(void) {
  return __builtin_frame_address(1);

  // CIR-LABEL: test_frame_address
  // CIR: [[ARG:%.*]] = cir.const #cir.int<1> : !u32i
  // CIR: {{%.*}} = cir.frame_address([[ARG]])

  // LLVM-LABEL: @test_frame_address
  // LLVM: {{%.*}} = call ptr @llvm.frameaddress.p0(i32 1)
}

// Following block of tests are for __builtin_launder
// FIXME: Once we fully __builtin_launder by allowing -fstrict-vtable-pointers,
//        we should move following block of tests to a separate file.
namespace launder_test {
//===----------------------------------------------------------------------===//
//                            Positive Cases
//===----------------------------------------------------------------------===//

struct TestVirtualFn {
  virtual void foo() {}
};

// CIR-LABEL: test_builtin_launder_virtual_fn
// LLVM: define{{.*}} void @test_builtin_launder_virtual_fn(ptr [[P:%.*]])
extern "C" void test_builtin_launder_virtual_fn(TestVirtualFn *p) {
  // CIR: cir.return

  // LLVM: store ptr [[P]], ptr [[P_ADDR:%.*]], align 8
  // LLVM-NEXT: [[TMP0:%.*]] = load ptr, ptr [[P_ADDR]], align 8
  // LLVM-NEXT: store ptr [[TMP0]], ptr {{%.*}}
  // LLVM-NEXT: ret void
  TestVirtualFn *d = __builtin_launder(p);
}

struct TestPolyBase : TestVirtualFn {
};

// CIR-LABEL: test_builtin_launder_poly_base
// LLVM: define{{.*}} void @test_builtin_launder_poly_base(ptr [[P:%.*]])
extern "C" void test_builtin_launder_poly_base(TestPolyBase *p) {
  // CIR: cir.return

  // LLVM: store ptr [[P]], ptr [[P_ADDR:%.*]], align 8
  // LLVM-NEXT: [[TMP0:%.*]] = load ptr, ptr [[P_ADDR]], align 8
  // LLVM-NEXT: store ptr [[TMP0]], ptr {{%.*}}
  // LLVM-NEXT: ret void
  TestPolyBase *d = __builtin_launder(p);
}

struct TestBase {};
struct TestVirtualBase : virtual TestBase {};

// CIR-LABEL: test_builtin_launder_virtual_base
// LLVM: define{{.*}} void @test_builtin_launder_virtual_base(ptr [[P:%.*]])
extern "C" void test_builtin_launder_virtual_base(TestVirtualBase *p) {
  TestVirtualBase *d = __builtin_launder(p);

  // CIR: cir.return

  // LLVM: store ptr [[P]], ptr [[P_ADDR:%.*]], align 8
  // LLVM-NEXT: [[TMP0:%.*]] = load ptr, ptr [[P_ADDR]], align 8
  // LLVM-NEXT: store ptr [[TMP0]], ptr {{%.*}}
  // LLVM-NEXT: ret void
}

//===----------------------------------------------------------------------===//
//                            Negative Cases
//===----------------------------------------------------------------------===//

// CIR-LABEL: test_builtin_launder_ommitted_one
// LLVM: define{{.*}} void @test_builtin_launder_ommitted_one(ptr [[P:%.*]])
extern "C" void test_builtin_launder_ommitted_one(int *p) {
  int *d = __builtin_launder(p);

  // CIR: cir.return

  // LLVM-NEXT: [[P_ADDR:%.*]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT: [[D:%.*]] = alloca ptr, i64 1, align 8
  // LLVM: store ptr [[P]], ptr [[P_ADDR:%.*]], align 8
  // LLVM-NEXT: [[TMP0:%.*]] = load ptr, ptr [[P_ADDR]], align 8
  // LLVM-NEXT: store ptr [[TMP0]], ptr [[D]]
  // LLVM-NEXT: ret void
}

struct TestNoInvariant {
  int x;
};

// CIR-LABEL: test_builtin_launder_ommitted_two
// LLVM: define{{.*}} void @test_builtin_launder_ommitted_two(ptr [[P:%.*]])
extern "C" void test_builtin_launder_ommitted_two(TestNoInvariant *p) {
  TestNoInvariant *d = __builtin_launder(p);
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM-NEXT: [[P_ADDR:%.*]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT: [[D:%.*]] = alloca ptr, i64 1, align 8
  // LLVM: store ptr [[P]], ptr [[P_ADDR:%.*]], align 8
  // LLVM-NEXT: [[TMP0:%.*]] = load ptr, ptr [[P_ADDR]], align 8
  // LLVM-NEXT: store ptr [[TMP0]], ptr [[D]]
  // LLVM-NEXT: ret void
}

struct TestVirtualMember {
  TestVirtualFn member;
};

// CIR-LABEL: test_builtin_launder_virtual_member
// LLVM: define{{.*}} void @test_builtin_launder_virtual_member
extern "C" void test_builtin_launder_virtual_member(TestVirtualMember *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  TestVirtualMember *d = __builtin_launder(p);
}

struct TestVirtualMemberDepth2 {
  TestVirtualMember member;
};

// CIR-LABEL: test_builtin_launder_virtual_member_depth_2
// LLVM: define{{.*}} void @test_builtin_launder_virtual_member_depth_2
extern "C" void test_builtin_launder_virtual_member_depth_2(TestVirtualMemberDepth2 *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  TestVirtualMemberDepth2 *d = __builtin_launder(p);
}

struct TestVirtualReferenceMember {
  TestVirtualFn &member;
};

// CIR-LABEL: test_builtin_launder_virtual_reference_member
// LLVM: define{{.*}} void @test_builtin_launder_virtual_reference_member
extern "C" void test_builtin_launder_virtual_reference_member(TestVirtualReferenceMember *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  TestVirtualReferenceMember *d = __builtin_launder(p);
}

struct TestRecursiveMember {
  TestRecursiveMember() : member(*this) {}
  TestRecursiveMember &member;
};

// CIR-LABEL: test_builtin_launder_recursive_member
// LLVM: define{{.*}} void @test_builtin_launder_recursive_member
extern "C" void test_builtin_launder_recursive_member(TestRecursiveMember *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  TestRecursiveMember *d = __builtin_launder(p);
}

struct TestVirtualRecursiveMember {
  TestVirtualRecursiveMember() : member(*this) {}
  TestVirtualRecursiveMember &member;
  virtual void foo();
};

// CIR-LABEL: test_builtin_launder_virtual_recursive_member
// LLVM: define{{.*}} void @test_builtin_launder_virtual_recursive_member
extern "C" void test_builtin_launder_virtual_recursive_member(TestVirtualRecursiveMember *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  TestVirtualRecursiveMember *d = __builtin_launder(p);
}

// CIR-LABEL: test_builtin_launder_array
// LLVM: define{{.*}} void @test_builtin_launder_array
extern "C" void test_builtin_launder_array(TestVirtualFn (&Arr)[5]) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  TestVirtualFn *d = __builtin_launder(Arr);
}

// CIR-LABEL: test_builtin_launder_array_nested
// LLVM: define{{.*}} void @test_builtin_launder_array_nested
extern "C" void test_builtin_launder_array_nested(TestVirtualFn (&Arr)[5][2]) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  using RetTy = TestVirtualFn(*)[2];
  RetTy d = __builtin_launder(Arr);
}

// CIR-LABEL: test_builtin_launder_array_no_invariant
// LLVM: define{{.*}} void @test_builtin_launder_array_no_invariant
extern "C" void test_builtin_launder_array_no_invariant(TestNoInvariant (&Arr)[5]) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  TestNoInvariant *d = __builtin_launder(Arr);
}

// CIR-LABEL: test_builtin_launder_array_nested_no_invariant
// LLVM: define{{.*}} void @test_builtin_launder_array_nested_no_invariant
extern "C" void test_builtin_launder_array_nested_no_invariant(TestNoInvariant (&Arr)[5][2]) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  using RetTy = TestNoInvariant(*)[2];
  RetTy d = __builtin_launder(Arr);
}

template <class Member>
struct WithMember {
  Member mem;
};

template struct WithMember<TestVirtualFn[5]>;

// CIR-LABEL: test_builtin_launder_member_array
// LLVM: define{{.*}} void @test_builtin_launder_member_array
extern "C" void test_builtin_launder_member_array(WithMember<TestVirtualFn[5]> *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  auto *d = __builtin_launder(p);
}

template struct WithMember<TestVirtualFn[5][2]>;

// CIR-LABEL: test_builtin_launder_member_array_nested
// LLVM: define{{.*}} void @test_builtin_launder_member_array_nested
extern "C" void test_builtin_launder_member_array_nested(WithMember<TestVirtualFn[5][2]> *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  auto *d = __builtin_launder(p);
}

template struct WithMember<TestNoInvariant[5]>;

// CIR-LABEL: test_builtin_launder_member_array_no_invariant
// LLVM: define{{.*}} void @test_builtin_launder_member_array_no_invariant
extern "C" void test_builtin_launder_member_array_no_invariant(WithMember<TestNoInvariant[5]> *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  auto *d = __builtin_launder(p);
}

template struct WithMember<TestNoInvariant[5][2]>;

// CIR-LABEL: test_builtin_launder_member_array_nested_no_invariant
// LLVM: define{{.*}} void @test_builtin_launder_member_array_nested_no_invariant
extern "C" void test_builtin_launder_member_array_nested_no_invariant(WithMember<TestNoInvariant[5][2]> *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  auto *d = __builtin_launder(p);
}

template <class T>
struct WithBase : T {};

template struct WithBase<TestNoInvariant>;

// CIR-LABEL: test_builtin_launder_base_no_invariant
// LLVM: define{{.*}} void @test_builtin_launder_base_no_invariant
extern "C" void test_builtin_launder_base_no_invariant(WithBase<TestNoInvariant> *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  auto *d = __builtin_launder(p);
}

template struct WithBase<TestVirtualFn>;

// CIR-LABEL: test_builtin_launder_base
// LLVM: define{{.*}} void @test_builtin_launder_base
extern "C" void test_builtin_launder_base(WithBase<TestVirtualFn> *p) {
  // CIR: cir.return

  // LLVM-NOT: llvm.launder.invariant.group
  // LLVM: ret void
  auto *d = __builtin_launder(p);
}
}
