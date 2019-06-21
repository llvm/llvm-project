// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fno-experimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=O1,O1-LEGACY
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fexperimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s --check-prefixes=O1,O1-NEWPM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=O0
//
// Ensure that we place appropriate lifetime markers around indirectly returned
// temporaries, and that the lifetime.ends appear in a timely manner.
//
// -O1 is used so lifetime markers actually get emitted.

struct S {
  int ns[40];
};

struct S foo(void);

// CHECK-LABEL: define dso_local void @bar
struct S bar() {
  // O0-NOT: @llvm.lifetime.start
  // O0-NOT: @llvm.lifetime.end

  struct S r;
  // O1: call void @llvm.lifetime.start.p0i8({{[^,]*}}, i8* nonnull %[[TMP1:[^)]+]])
  // O1: call void @foo
  r = foo();
  // O1: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* nonnull %[[TMP1]])

  // O1: call void @llvm.lifetime.start.p0i8({{[^,]*}}, i8* nonnull %[[TMP2:[^)]+]])
  // O1: call void @foo
  r = foo();
  // O1: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* nonnull %[[TMP2]])

  // O1: call void @llvm.lifetime.start.p0i8({{[^,]*}}, i8* nonnull %[[TMP3:[^)]+]])
  // O1: call void @foo
  r = foo();
  // O1: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* nonnull %[[TMP3]])

  return r;
}

struct S foo_int(int);

// Be sure that we're placing the lifetime.end so that all paths go through it.
// Since this function turns out to be large-ish, optnone to hopefully keep it
// stable.
// CHECK-LABEL: define dso_local void @baz
__attribute__((optnone))
struct S baz(int i, volatile int *j) {
  // O0-NOT: @llvm.lifetime.start
  // O0-NOT: @llvm.lifetime.end

  struct S r;
  // O1: %[[TMP1_ALLOCA:[^ ]+]] = alloca %struct.S
  // O1: %[[TMP2_ALLOCA:[^ ]+]] = alloca %struct.S

  do {
    // O1: %[[P:[^ ]+]] = bitcast %struct.S* %[[TMP1_ALLOCA]] to i8*
    // O1-LEGACY: call void @llvm.lifetime.start.p0i8({{[^,]*}}, i8* %[[P]])
    // O1-NEWPM: call void @llvm.lifetime.start.p0i8({{[^,]*}}, i8* nonnull %[[P]])
    //
    // O1-LEGACY: %[[P:[^ ]+]] = bitcast %struct.S* %[[TMP1_ALLOCA]] to i8*
    // O1-LEGACY: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* %[[P]])
    // O1-NEWPM: %[[TMP3:.*]] = bitcast %struct.S* %[[TMP2_ALLOCA]] to i8*
    // O1-NEWPM: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* nonnull %[[P]])
    //
    // O1-LEGACY: call void @foo_int(%struct.S* sret %[[TMP1_ALLOCA]],
    // O1-NEWPM: call void @foo_int(%struct.S* nonnull sret %[[TMP1_ALLOCA]],
    // O1: call void @llvm.memcpy
    // O1-LEGACY: %[[P:[^ ]+]] = bitcast %struct.S* %[[TMP1_ALLOCA]] to i8*
    // O1-LEGACY: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* %[[P]])
    // O1-NEWPM: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* nonnull %[[P]])
    // O1-LEGACY: %[[P:[^ ]+]] = bitcast %struct.S* %[[TMP2_ALLOCA]] to i8*
    // O1-LEGACY: call void @llvm.lifetime.start.p0i8({{[^,]*}}, i8* %[[P]])
    // O1-NEWPM: call void @llvm.lifetime.start.p0i8({{[^,]*}}, i8* nonnull %[[TMP3]])
    // O1-LEGACY: call void @foo_int(%struct.S* sret %[[TMP2_ALLOCA]],
    // O1-NEWPM: call void @foo_int(%struct.S* nonnull sret %[[TMP2_ALLOCA]],
    // O1: call void @llvm.memcpy
    // O1-LEGACY: %[[P:[^ ]+]] = bitcast %struct.S* %[[TMP2_ALLOCA]] to i8*
    // O1-LEGACY: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* %[[P]])
    // O1-NEWPM: call void @llvm.lifetime.end.p0i8({{[^,]*}}, i8* nonnull %[[TMP3]])
    r = foo_int(({
      if (*j)
        break;
      i++;
    }));

    r = foo_int(i++);
   } while (1);

  return r;
}
