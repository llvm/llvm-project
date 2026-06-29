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

// Bypassed variables are initialized at the goto source (before the branch).
// UNINIT-LABEL:  test_goto_unreachable_value(
// ZERO-LABEL:    test_goto_unreachable_value(
// ZERO: %oops = alloca i32, align 4
// ZERO: store i32 0, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO: br label %jump
// PATTERN-LABEL: test_goto_unreachable_value(
// PATTERN: %oops = alloca i32, align 4
// PATTERN: store i32 -1431655766, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN: br label %jump
void test_goto_unreachable_value() {
  goto jump;
  int oops;
 jump:
  used(oops);
}

// Bypassed variables are initialized at the jump target.
// UNINIT-LABEL:  test_goto(
// ZERO-LABEL:    test_goto(
// ZERO: %oops = alloca i32, align 4
// ZERO: store i32 0, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_goto(
// PATTERN: %oops = alloca i32, align 4
// PATTERN: store i32 -1431655766, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
void test_goto(int i) {
  if (i)
    goto jump;
  int oops;
 jump:
  used(oops);
}

// Bypassed variables are initialized at the case target.
// UNINIT-LABEL:  test_switch(
// ZERO-LABEL:    test_switch(
// ZERO: %oops = alloca i32, align 4
// ZERO: store i32 0, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_switch(
// PATTERN: %oops = alloca i32, align 4
// PATTERN: store i32 -1431655766, ptr %oops, align 4, !annotation [[AUTO_INIT:!.+]]
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

// UNINIT-LABEL:  test_goto_multiple_bypassed(
// ZERO-LABEL:    test_goto_multiple_bypassed(
// ZERO: %a = alloca i32, align 4
// ZERO: %b = alloca i32, align 4
// ZERO-DAG: store i32 0, ptr %a, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO-DAG: store i32 0, ptr %b, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_goto_multiple_bypassed(
// PATTERN: %a = alloca i32, align 4
// PATTERN: %b = alloca i32, align 4
// PATTERN-DAG: store i32 -1431655766, ptr %a, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-DAG: store i32 -1431655766, ptr %b, align 4, !annotation [[AUTO_INIT:!.+]]
void test_goto_multiple_bypassed() {
  goto jump;
  int a;
  int b;
 jump:
  used(a);
  used(b);
}

// UNINIT-LABEL:  test_goto_bypassed_uninitialized_attr(
// ZERO-LABEL:    test_goto_bypassed_uninitialized_attr(
// ZERO-NOT: store {{.*}}%skip_me
// ZERO: call void @{{.*}}used
// PATTERN-LABEL: test_goto_bypassed_uninitialized_attr(
// PATTERN-NOT: store {{.*}}%skip_me
// PATTERN: call void @{{.*}}used
void test_goto_bypassed_uninitialized_attr() {
  goto jump;
  [[clang::uninitialized]] int skip_me;
 jump:
  used(skip_me);
}

// UNINIT-LABEL:  test_switch_between_cases(
// ZERO-LABEL:    test_switch_between_cases(
// ZERO: %x = alloca i32, align 4
// ZERO: store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_switch_between_cases(
// PATTERN: %x = alloca i32, align 4
// PATTERN: store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
void test_switch_between_cases(int c) {
  switch (c) {
  case 0:
    int x;
    x = 42;
    used(x);
    break;
  case 1:
    used(x);
    break;
  }
}

// UNINIT-LABEL:  test_switch_precase(
// ZERO-LABEL:    test_switch_precase(
// ZERO: %x = alloca i32, align 4
// ZERO: store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_switch_precase(
// PATTERN: %x = alloca i32, align 4
// PATTERN: store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
void test_switch_precase(int c) {
  switch (c) {
    int x;
  case 0:
    x = 1;
    used(x);
    break;
  }
}

// UNINIT-LABEL:  test_computed_goto(
// ZERO-LABEL:    test_computed_goto(
// ZERO: %y = alloca i32, align 4
// ZERO: store i32 0, ptr %y, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_computed_goto(
// PATTERN: %y = alloca i32, align 4
// PATTERN: store i32 -1431655766, ptr %y, align 4, !annotation [[AUTO_INIT:!.+]]
void test_computed_goto(int x) {
  void *targets[] = {&&label1, &&label2};
  goto *targets[x];
  int y;
label1:
  used(y);
  return;
label2:
  return;
}

// UNINIT-LABEL:  test_loop_bypass(
// ZERO-LABEL:    test_loop_bypass(
// ZERO: %x = alloca i32, align 4
// ZERO: while.body:
// ZERO: store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO: br label %X
// PATTERN-LABEL: test_loop_bypass(
// PATTERN: %x = alloca i32, align 4
// PATTERN: while.body:
// PATTERN: store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN: br label %X
void test_loop_bypass() {
  while (true) {
    goto X;
    int x;
    X:
    used(x);
    if (x) break;
  }
}

// UNINIT-LABEL:  test_complex_multi_goto(
// ZERO-LABEL:    test_complex_multi_goto(
// ZERO:      Z:
// ZERO-NEXT: store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO-NEXT: br label %Y
// ZERO:      X:
// ZERO-NEXT: store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO-NEXT: br label %Y
// ZERO:      sw.bb:
// ZERO-NOT:  store {{.*}}%x
// ZERO:      br label %X
// ZERO:      sw.bb1:
// ZERO-NOT:  store {{.*}}%x
// ZERO:      br label %Z
// ZERO:      sw.epilog:
// ZERO-NOT:  store {{.*}}%x
// ZERO:      br label %Y
// PATTERN-LABEL: test_complex_multi_goto(
// PATTERN:      Z:
// PATTERN-NEXT: store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-NEXT: br label %Y
// PATTERN:      X:
// PATTERN-NEXT: store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-NEXT: br label %Y
// PATTERN:      sw.bb:
// PATTERN-NOT:  store {{.*}}%x
// PATTERN:      br label %X
// PATTERN:      sw.bb1:
// PATTERN-NOT:  store {{.*}}%x
// PATTERN:      br label %Z
// PATTERN:      sw.epilog:
// PATTERN-NOT:  store {{.*}}%x
// PATTERN:      br label %Y
void test_complex_multi_goto(int g(int*)) {
  while (true) {
    Z:
    goto Y;
    X:
    goto Y;
    int x;
    Y:
    switch (g(&x)) {
    case 0:
      goto X;
    case 1:
      goto Z;
    }
    goto Y;
  }
}

// UNINIT-LABEL:  test_no_reinit_in_scope(
// ZERO-LABEL:    test_no_reinit_in_scope(
// ZERO:      while.body:
// ZERO-NEXT: store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO-NEXT: br label %Y
// ZERO:      if.end:
// ZERO-NOT:  store {{.*}}%x
// ZERO:      br label %Y
// PATTERN-LABEL: test_no_reinit_in_scope(
// PATTERN:      while.body:
// PATTERN-NEXT: store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-NEXT: br label %Y
// PATTERN:      if.end:
// PATTERN-NOT:  store {{.*}}%x
// PATTERN:      br label %Y
void test_no_reinit_in_scope(int g(int*)) {
  while (true) {
    goto Y;
    int x;
    Y:
    if (g(&x))
      break;
    goto Y;
  }
}

// Backward goto: x is already in scope, no bypass init should occur at the
// goto.
// UNINIT-LABEL:  test_backward_goto_no_init(
// ZERO-LABEL:    test_backward_goto_no_init(
// ZERO:      store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO:      L:
// ZERO-NOT:  store {{.*}}%x
// ZERO:      if.then:
// ZERO-NOT:  store {{.*}}%x
// ZERO:      br label %L
// PATTERN-LABEL: test_backward_goto_no_init(
// PATTERN:      store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN:      L:
// PATTERN-NOT:  store {{.*}}%x
// PATTERN:      if.then:
// PATTERN-NOT:  store {{.*}}%x
// PATTERN:      br label %L
void test_backward_goto_no_init() {
  int x;
 L:
  used(x);
  if (x)
    goto L;
}

// Switch with default case bypassing a variable declared in case 0.
// UNINIT-LABEL:  test_switch_default_bypass(
// ZERO-LABEL:    test_switch_default_bypass(
// ZERO:      sw.default:
// ZERO-NEXT: store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_switch_default_bypass(
// PATTERN:      sw.default:
// PATTERN-NEXT: store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
void test_switch_default_bypass(int c) {
  switch (c) {
  case 0:
    int x;
    x = 10;
    used(x);
    break;
  default:
    used(x);
    break;
  }
}

// Multipe variables bypassed by the same goto so both must be initialized.
// UNINIT-LABEL:  test_goto_multiple_vars(
// ZERO-LABEL:    test_goto_multiple_vars(
// ZERO: %a = alloca i32, align 4
// ZERO: %b = alloca i32, align 4
// ZERO: store i32 0, ptr %a, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO: store i32 0, ptr %b, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO: br label %jump
// PATTERN-LABEL: test_goto_multiple_vars(
// PATTERN: %a = alloca i32, align 4
// PATTERN: %b = alloca i32, align 4
// PATTERN: store i32 -1431655766, ptr %a, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN: store i32 -1431655766, ptr %b, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN: br label %jump
void test_goto_multiple_vars() {
  goto jump;
  int a;
  int b;
 jump:
  used(a);
  used(b);
}

// UNINIT-LABEL:  test_backward_goto_bypass(
// ZERO-LABEL:    test_backward_goto_bypass(
// ZERO:      jump:
// ZERO:      call void @{{.*}}used
// ZERO:      call void @{{.*}}used
// ZERO-DAG:  store i32 0, ptr %b, align 4
// ZERO-DAG:  store i32 0, ptr %a, align 4
// ZERO:      br label %jump
// PATTERN-LABEL: test_backward_goto_bypass(
// PATTERN:      jump:
// PATTERN:      call void @{{.*}}used
// PATTERN:      call void @{{.*}}used
// PATTERN-DAG:  store i32 -1431655766, ptr %b
// PATTERN-DAG:  store i32 -1431655766, ptr %a
// PATTERN:      br label %jump
void test_backward_goto_bypass() {
  {
    int a;
    int b;
jump:
    used(a);
    used(b);
  }
  goto jump;
}

// C++ [basic.stc.auto]: scope re-entry restarts the lifetime, so the init is
// emitted at the goto source and reruns each iteration (store in BEGIN, not
// entry). Contrast the C version, which inits once in entry and returns 10.
// UNINIT-LABEL:  test_backward_goto_around_decl(
// ZERO-LABEL:    test_backward_goto_around_decl(
// ZERO:      entry:
// ZERO-NOT:  !annotation
// ZERO:      BEGIN:
// ZERO:      store ptr null, ptr %p, align 8, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_backward_goto_around_decl(
// PATTERN:      entry:
// PATTERN-NOT:  !annotation
// PATTERN:      BEGIN:
// PATTERN:      store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %p, align 8, !annotation [[AUTO_INIT:!.+]]
int test_backward_goto_around_decl(int b) {
BEGIN:;
  goto CONT;
  int *p;
CONT:
  if (b)
    *p = 10;
  p = &b;
  if (!b) {
    b = 1;
    goto BEGIN;
  }
  return b;
}

// Nested loops: goto source is in the inner body, so reinit lands there
// (while.body3), every inner iteration.
// UNINIT-LABEL:  nested_loops(
// ZERO-LABEL:    nested_loops(
// ZERO:      entry:
// ZERO-NOT:  store {{.*}}%x{{.*}}!annotation
// ZERO:      while.body3:
// ZERO:      store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: nested_loops(
// PATTERN:      entry:
// PATTERN-NOT:  store {{.*}}%x{{.*}}!annotation
// PATTERN:      while.body3:
// PATTERN:      store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
void nested_loops(int n) {
  while (n) {
    while (n) {
      goto X;
      int x;
    X:
      used(x);
      n--;
    }
  }
}

// Nested loops + switch: in C++ the reinit is emitted at each case target, so
// it runs on every case entry (one store per case).
// UNINIT-LABEL:  nested_loops_switch(
// ZERO-LABEL:    nested_loops_switch(
// ZERO:      while.body3:
// ZERO:      switch i32
// ZERO-DAG:  store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO-DAG:  store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT]]
// PATTERN-LABEL: nested_loops_switch(
// PATTERN:      while.body3:
// PATTERN:      switch i32
// PATTERN-DAG:  store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-DAG:  store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT]]
void nested_loops_switch(int n, int c) {
  while (n) {
    while (n) {
      switch (c) {
        int x;
      case 0:
        x = 1;
        used(x);
        break;
      default:
        used(x);
        break;
      }
      n--;
    }
  }
}

// Computed goto with multiple scopes: jump sources are unknown, so all bypassed
// variables fall back to a single function-scope init in entry. Even though a
// regular switch is also present, its case targets must NOT reinitialize -- that
// could clobber a variable still live across the computed jump. One init in
// entry, none after the indirectbr.
// UNINIT-LABEL:  test_computed_goto_multi_scope(
// ZERO-LABEL:    test_computed_goto_multi_scope(
// ZERO:      entry:
// ZERO:      store i32 0, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// ZERO:      indirectbr
// ZERO-NOT:  store i32 0, ptr %x, align 4, !annotation
// PATTERN-LABEL: test_computed_goto_multi_scope(
// PATTERN:      entry:
// PATTERN:      store i32 -1431655766, ptr %x, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN:      indirectbr
// PATTERN-NOT:  store i32 -1431655766, ptr %x, align 4, !annotation
void test_computed_goto_multi_scope(int n, int c) {
  void *targets[] = {&&L1, &&L2};
  goto *targets[n];
  int x;
  switch (c) {
  case 0:
  L1:
    used(x);
    break;
  default:
  L2:
    used(x);
    break;
  }
}

} // extern "C"

// CHECK: [[AUTO_INIT]] = !{ !"auto-init" }
