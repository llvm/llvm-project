// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm -o - %s | FileCheck %s --check-prefixes=O1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -o - %s | FileCheck %s --check-prefix=O0
//
// Ensure that we place appropriate lifetime markers around indirectly returned
// temporaries, and that the lifetime.ends appear in a timely manner.
//
// -O1 is used so lifetime markers actually get emitted and optnone is added
// to avoid elimination of lifetime markers by optimizations.

struct S {
  int ns[40];
};

struct S foo(void);

// CHECK-LABEL: define dso_local void @bar
__attribute__((optnone))
struct S bar(void) {
  // O0-NOT: @llvm.lifetime.start
  // O0-NOT: @llvm.lifetime.end

  struct S r;
  // O1: %[[TMP1_ALLOCA:[^ ]+]] = alloca %struct.S
  // O1: %[[TMP2_ALLOCA:[^ ]+]] = alloca %struct.S
  // O1: %[[TMP3_ALLOCA:[^ ]+]] = alloca %struct.S

  // O1: call void @llvm.lifetime.start.p0(ptr %[[TMP1_ALLOCA]])
  // O1: call void @foo
  r = foo();
  // O1: memcpy
  // O1: call void @llvm.lifetime.end.p0(ptr %[[TMP1_ALLOCA]])

  // O1: call void @llvm.lifetime.start.p0(ptr %[[TMP2_ALLOCA]])
  // O1: call void @foo
  r = foo();
  // O1: memcpy
  // O1: call void @llvm.lifetime.end.p0(ptr %[[TMP2_ALLOCA]])

  // O1: call void @llvm.lifetime.start.p0(ptr %[[TMP3_ALLOCA]])
  // O1: call void @foo
  r = foo();
  // O1: memcpy
  // O1: call void @llvm.lifetime.end.p0(ptr %[[TMP3_ALLOCA]])

  return r;
}

struct S foo_int(int);

// Be sure that we're placing the lifetime.end so that all paths go through it.
// CHECK-LABEL: define dso_local void @baz
__attribute__((optnone))
struct S baz(int i, volatile int *j) {
  // O0-NOT: @llvm.lifetime.start
  // O0-NOT: @llvm.lifetime.end

  struct S r;
  // O1: %[[TMP1_ALLOCA:[^ ]+]] = alloca %struct.S
  // O1: %[[TMP2_ALLOCA:[^ ]+]] = alloca %struct.S

  do {
    // O1: call void @llvm.lifetime.start.p0(ptr %[[TMP1_ALLOCA]])
    //
    // O1: call void @llvm.lifetime.end.p0(ptr %[[TMP1_ALLOCA]])
    //
    // O1: call void @foo_int(ptr dead_on_unwind writable sret(%struct.S) align 4 %[[TMP1_ALLOCA]],
    // O1: call void @llvm.memcpy
    // O1: call void @llvm.lifetime.end.p0(ptr %[[TMP1_ALLOCA]])
    // O1: call void @llvm.lifetime.start.p0(ptr %[[TMP2_ALLOCA]])
    // O1: call void @foo_int(ptr dead_on_unwind writable sret(%struct.S) align 4 %[[TMP2_ALLOCA]],
    // O1: call void @llvm.memcpy
    // O1: call void @llvm.lifetime.end.p0(ptr %[[TMP2_ALLOCA]])
    r = foo_int(({
      if (*j)
        break;
      i++;
    }));

    r = foo_int(i++);
   } while (1);

  return r;
}
