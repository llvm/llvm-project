// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fblocks %s -emit-llvm -o - | FileCheck %s -check-prefix=UNINIT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO

// None of the synthesized globals should contain `undef`.
// PATTERN-NOT: undef
// ZERO-NOT: undef

template<typename T> void used(T &) noexcept;

extern "C" {

// UNINIT-LABEL:  test_selfinit(
// ZERO-LABEL:    test_selfinit(
// ZERO: store i32 0, ptr %self, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_selfinit(
// PATTERN: store i32 -1431655766, ptr %self, align 4, !annotation [[AUTO_INIT:!.+]]
void test_selfinit() {
  int self = self + 1;
  used(self);
}

// UNINIT-LABEL:  test_block(
// ZERO-LABEL:    test_block(
// ZERO: store i32 0, ptr %block, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_block(
// PATTERN: store i32 -1431655766, ptr %block, align 4, !annotation [[AUTO_INIT:!.+]]
void test_block() {
  __block int block;
  used(block);
}

// Using the variable being initialized is typically UB in C, but for blocks we
// can be nice: they imply extra book-keeping and we can do the auto-init before
// any of said book-keeping.
//
// UNINIT-LABEL:  test_block_self_init(
// ZERO-LABEL:    test_block_self_init(
// ZERO:          %block = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// ZERO:          %captured1 = getelementptr inbounds nuw %struct.__block_byref_captured, ptr %captured, i32 0, i32 4
// ZERO-NEXT:     store ptr null, ptr %captured1, align 8, !annotation [[AUTO_INIT:!.+]]
// ZERO:          %call = call ptr @create(
// PATTERN-LABEL: test_block_self_init(
// PATTERN:       %block = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// PATTERN:       %captured1 = getelementptr inbounds nuw %struct.__block_byref_captured, ptr %captured, i32 0, i32 4
// PATTERN-NEXT:  store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %captured1, align 8, !annotation [[AUTO_INIT:!.+]]
// PATTERN:       %call = call ptr @create(
using Block = void (^)();
typedef struct XYZ {
  Block block;
} * xyz_t;
void test_block_self_init() {
  extern xyz_t create(Block block);
  __block xyz_t captured = create(^() {
    used(captured);
  });
}

// Capturing with escape after initialization is also an edge case.
//
// UNINIT-LABEL:  test_block_captures_self_after_init(
// ZERO-LABEL:    test_block_captures_self_after_init(
// ZERO:          %block = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// ZERO:          %captured1 = getelementptr inbounds nuw %struct.__block_byref_captured.1, ptr %captured, i32 0, i32 4
// ZERO-NEXT:     store ptr null, ptr %captured1, align 8, !annotation [[AUTO_INIT:!.+]]
// ZERO:          %call = call ptr @create(
// PATTERN-LABEL: test_block_captures_self_after_init(
// PATTERN:       %block = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// PATTERN:       %captured1 = getelementptr inbounds nuw %struct.__block_byref_captured.1, ptr %captured, i32 0, i32 4
// PATTERN-NEXT:  store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %captured1, align 8, !annotation [[AUTO_INIT:!.+]]
// PATTERN:       %call = call ptr @create(
void test_block_captures_self_after_init() {
  extern xyz_t create(Block block);
  __block xyz_t captured;
  captured = create(^() {
    used(captured);
  });
}

// This type of code is currently not handled by zero / pattern initialization.
// The test will break when that is fixed.
// UNINIT-LABEL:  test_goto_unreachable_value(
// ZERO-LABEL:    test_goto_unreachable_value(
// ZERO-NOT: store {{.*}}%oops
// PATTERN-LABEL: test_goto_unreachable_value(
// PATTERN-NOT: store {{.*}}%oops
void test_goto_unreachable_value() {
  goto jump;
  int oops;
 jump:
  used(oops);
}

// This type of code is currently not handled by zero / pattern initialization.
// The test will break when that is fixed.
// UNINIT-LABEL:  test_goto(
// ZERO-LABEL:    test_goto(
// ZERO: if.then:
// ZERO: br label %jump
// ZERO: store i32 0, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO: br label %jump
// ZERO: jump:
// PATTERN-LABEL: test_goto(
// PATTERN: if.then:
// PATTERN: br label %jump
// PATTERN: store i32 -1431655766, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN: br label %jump
// PATTERN: jump:
void test_goto(int i) {
  if (i)
    goto jump;
  int oops;
 jump:
  used(oops);
}

// This type of code is currently not handled by zero / pattern initialization.
// The test will break when that is fixed.
// UNINIT-LABEL:  test_switch(
// ZERO-LABEL:    test_switch(
// ZERO:      sw.bb:
// ZERO-NEXT: store i32 0, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO:      sw.bb1:
// ZERO-NEXT: call void @{{.*}}used
// PATTERN-LABEL: test_switch(
// PATTERN:      sw.bb:
// PATTERN-NEXT: store i32 -1431655766, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN:      sw.bb1:
// PATTERN-NEXT: call void @{{.*}}used
void test_switch(int i) {
  switch (i) {
  case 0:
    int oops;
    break;
  case 1:
    used(oops);
  }
}

// UNINIT-LABEL:  test_vla(
// ZERO-LABEL:    test_vla(
// ZERO:  %[[SIZE:[0-9]+]] = mul nuw i64 %{{.*}}, 4
// ZERO:  call void @llvm.memset{{.*}}(ptr align 16 %{{.*}}, i8 0, i64 %[[SIZE]], i1 false), !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_vla(
// PATTERN:  %vla.iszerosized = icmp eq i64 %{{.*}}, 0
// PATTERN:  br i1 %vla.iszerosized, label %vla-init.cont, label %vla-setup.loop
// PATTERN: vla-setup.loop:
// PATTERN:  %[[SIZE:[0-9]+]] = mul nuw i64 %{{.*}}, 4
// PATTERN:  %vla.end = getelementptr inbounds i8, ptr %vla, i64 %[[SIZE]]
// PATTERN:  br label %vla-init.loop
// PATTERN: vla-init.loop:
// PATTERN:  %vla.cur = phi ptr [ %vla, %vla-setup.loop ], [ %vla.next, %vla-init.loop ]
// PATTERN:  call void @llvm.memcpy{{.*}} %vla.cur, {{.*}}@__const.test_vla.vla{{.*}}), !annotation [[AUTO_INIT:!.+]]
// PATTERN:  %vla.next = getelementptr inbounds i8, ptr %vla.cur, i64 4
// PATTERN:  %vla-init.isdone = icmp eq ptr %vla.next, %vla.end
// PATTERN:  br i1 %vla-init.isdone, label %vla-init.cont, label %vla-init.loop
// PATTERN: vla-init.cont:
// PATTERN:  call void @{{.*}}used
void test_vla(int size) {
  // Variable-length arrays can't have a zero size according to C11 6.7.6.2/5.
  // Neither can they be negative-sized.
  //
  // We don't use the former fact because some code creates zero-sized VLAs and
  // doesn't use them. clang makes these share locations with other stack
  // values, which leads to initialization of the wrong values.
  //
  // We rely on the later fact because it generates better code.
  //
  // Both cases are caught by UBSan.
  int vla[size];
  int *ptr = vla;
  used(ptr);
}

// UNINIT-LABEL:  test_alloca(
// ZERO-LABEL:    test_alloca(
// ZERO:          %[[SIZE:[a-z0-9]+]] = sext i32 %{{.*}} to i64
// ZERO-NEXT:     %[[ALLOCA:[a-z0-9]+]] = alloca i8, i64 %[[SIZE]], align [[ALIGN:[0-9]+]]
// ZERO-NEXT:     call void @llvm.memset{{.*}}(ptr align [[ALIGN]] %[[ALLOCA]], i8 0, i64 %[[SIZE]], i1 false), !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_alloca(
// PATTERN:       %[[SIZE:[a-z0-9]+]] = sext i32 %{{.*}} to i64
// PATTERN-NEXT:  %[[ALLOCA:[a-z0-9]+]] = alloca i8, i64 %[[SIZE]], align [[ALIGN:[0-9]+]]
// PATTERN-NEXT:  call void @llvm.memset{{.*}}(ptr align [[ALIGN]] %[[ALLOCA]], i8 -86, i64 %[[SIZE]], i1 false), !annotation [[AUTO_INIT:!.+]]
void test_alloca(int size) {
  void *ptr = __builtin_alloca(size);
  used(ptr);
}

// UNINIT-LABEL:  test_alloca_with_align(
// ZERO-LABEL:    test_alloca_with_align(
// ZERO:          %[[SIZE:[a-z0-9]+]] = sext i32 %{{.*}} to i64
// ZERO-NEXT:     %[[ALLOCA:[a-z0-9]+]] = alloca i8, i64 %[[SIZE]], align 128
// ZERO-NEXT:     call void @llvm.memset{{.*}}(ptr align 128 %[[ALLOCA]], i8 0, i64 %[[SIZE]], i1 false), !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_alloca_with_align(
// PATTERN:       %[[SIZE:[a-z0-9]+]] = sext i32 %{{.*}} to i64
// PATTERN-NEXT:  %[[ALLOCA:[a-z0-9]+]] = alloca i8, i64 %[[SIZE]], align 128
// PATTERN-NEXT:  call void @llvm.memset{{.*}}(ptr align 128 %[[ALLOCA]], i8 -86, i64 %[[SIZE]], i1 false), !annotation [[AUTO_INIT:!.+]]
void test_alloca_with_align(int size) {
  void *ptr = __builtin_alloca_with_align(size, 1024);
  used(ptr);
}

// UNINIT-LABEL:  test_alloca_uninitialized(
// ZERO-LABEL:    test_alloca_uninitialized(
// ZERO:          %[[SIZE:[a-z0-9]+]] = sext i32 %{{.*}} to i64
// ZERO-NEXT:     %[[ALLOCA:[a-z0-9]+]] = alloca i8, i64 %[[SIZE]], align [[ALIGN:[0-9]+]]
// ZERO-NOT:      call void @llvm.memset
// PATTERN-LABEL: test_alloca_uninitialized(
// PATTERN:       %[[SIZE:[a-z0-9]+]] = sext i32 %{{.*}} to i64
// PATTERN-NEXT:  %[[ALLOCA:[a-z0-9]+]] = alloca i8, i64 %[[SIZE]], align [[ALIGN:[0-9]+]]
// PATTERN-NOT:   call void @llvm.memset
void test_alloca_uninitialized(int size) {
  void *ptr = __builtin_alloca_uninitialized(size);
  used(ptr);
}

// UNINIT-LABEL:  test_alloca_with_align_uninitialized(
// ZERO-LABEL:    test_alloca_with_align_uninitialized(
// ZERO:          %[[SIZE:[a-z0-9]+]] = sext i32 %{{.*}} to i64
// ZERO-NEXT:     %[[ALLOCA:[a-z0-9]+]] = alloca i8, i64 %[[SIZE]], align 128
// ZERO-NOT:      call void @llvm.memset
// PATTERN-LABEL: test_alloca_with_align_uninitialized(
// PATTERN:       %[[SIZE:[a-z0-9]+]] = sext i32 %{{.*}} to i64
// PATTERN-NEXT:  %[[ALLOCA:[a-z0-9]+]] = alloca i8, i64 %[[SIZE]], align 128
// PATTERN-NOT:   call void @llvm.memset
void test_alloca_with_align_uninitialized(int size) {
  void *ptr = __builtin_alloca_with_align_uninitialized(size, 1024);
  used(ptr);
}

// UNINIT-LABEL:  test_struct_vla(
// ZERO-LABEL:    test_struct_vla(
// ZERO:  %[[SIZE:[0-9]+]] = mul nuw i64 %{{.*}}, 16
// ZERO:  call void @llvm.memset{{.*}}(ptr align 16 %{{.*}}, i8 0, i64 %[[SIZE]], i1 false), !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_struct_vla(
// PATTERN:  %vla.iszerosized = icmp eq i64 %{{.*}}, 0
// PATTERN:  br i1 %vla.iszerosized, label %vla-init.cont, label %vla-setup.loop
// PATTERN: vla-setup.loop:
// PATTERN:  %[[SIZE:[0-9]+]] = mul nuw i64 %{{.*}}, 16
// PATTERN:  %vla.end = getelementptr inbounds i8, ptr %vla, i64 %[[SIZE]]
// PATTERN:  br label %vla-init.loop
// PATTERN: vla-init.loop:
// PATTERN:  %vla.cur = phi ptr [ %vla, %vla-setup.loop ], [ %vla.next, %vla-init.loop ]
// PATTERN:  call void @llvm.memcpy{{.*}} %vla.cur, {{.*}}@__const.test_struct_vla.vla{{.*}}), !annotation [[AUTO_INIT:!.+]]
// PATTERN:  %vla.next = getelementptr inbounds i8, ptr %vla.cur, i64 16
// PATTERN:  %vla-init.isdone = icmp eq ptr %vla.next, %vla.end
// PATTERN:  br i1 %vla-init.isdone, label %vla-init.cont, label %vla-init.loop
// PATTERN: vla-init.cont:
// PATTERN:  call void @{{.*}}used
void test_struct_vla(int size) {
  // Same as above, but with a struct that doesn't just memcpy.
  struct {
    float f;
    char c;
    void *ptr;
  } vla[size];
  void *ptr = static_cast<void*>(vla);
  used(ptr);
}

// UNINIT-LABEL:  test_zsa(
// ZERO-LABEL:    test_zsa(
// ZERO: %zsa = alloca [0 x i32], align 4
// ZERO-NOT: %zsa
// ZERO:  call void @{{.*}}used
// PATTERN-LABEL: test_zsa(
// PATTERN: %zsa = alloca [0 x i32], align 4
// PATTERN-NOT: %zsa
// PATTERN:  call void @{{.*}}used
void test_zsa(int size) {
  // Technically not valid, but as long as clang accepts them we should do
  // something sensible (i.e. not store to the zero-size array).
  int zsa[0];
  used(zsa);
}

// UNINIT-LABEL:  test_huge_uninit(
// ZERO-LABEL:    test_huge_uninit(
// ZERO: call void @llvm.memset{{.*}}, i8 0, i64 65536, {{.*}}), !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_huge_uninit(
// PATTERN: call void @llvm.memset{{.*}}, i8 -86, i64 65536, {{.*}}), !annotation [[AUTO_INIT:!.+]]
void test_huge_uninit() {
  // We can't emit this as an inline constant to a store instruction because
  // SDNode hits an internal size limit.
  char big[65536];
  used(big);
}

// UNINIT-LABEL:  test_huge_small_init(
// ZERO-LABEL:    test_huge_small_init(
// ZERO: call void @llvm.memset{{.*}}, i8 0, i64 65536,
// ZERO-NOT: !annotation
// ZERO: store i8 97,
// ZERO: store i8 98,
// ZERO: store i8 99,
// ZERO: store i8 100,
// PATTERN-LABEL: test_huge_small_init(
// PATTERN: call void @llvm.memset{{.*}}, i8 0, i64 65536,
// PATTERN-NOT: !annotation
// PATTERN: store i8 97,
// PATTERN: store i8 98,
// PATTERN: store i8 99,
// PATTERN: store i8 100,
void test_huge_small_init() {
  char big[65536] = { 'a', 'b', 'c', 'd' };
  used(big);
}

// UNINIT-LABEL:  test_huge_larger_init(
// ZERO-LABEL:    test_huge_larger_init(
// ZERO:  call void @llvm.memcpy{{.*}} @__const.test_huge_larger_init.big, i64 65536,
// ZERO-NOT: !annotation
// PATTERN-LABEL: test_huge_larger_init(
// PATTERN:  call void @llvm.memcpy{{.*}} @__const.test_huge_larger_init.big, i64 65536,
// PATTERN-NOT: !annotation
void test_huge_larger_init() {
  char big[65536] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
  used(big);
}

} // extern "C"

// CHECK: [[AUTO_INIT]] = !{ !"auto-init" }
