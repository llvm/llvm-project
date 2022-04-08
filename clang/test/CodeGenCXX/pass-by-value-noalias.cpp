// RUN: %clang_cc1 -no-opaque-pointers -fpass-by-value-is-noalias -triple arm64-apple-iphoneos -emit-llvm -disable-llvm-optzns %s -o - 2>&1 | FileCheck --check-prefix=WITH_NOALIAS %s
// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-apple-iphoneos -emit-llvm -disable-llvm-optzns %s -o - 2>&1 | FileCheck --check-prefix=NO_NOALIAS %s

// A trivial struct large enough so it is not passed in registers on ARM64.
struct Foo {
  int a;
  int b;
  int c;
  int d;
  int e;
  int f;
};

// Make sure noalias is added to indirect arguments with trivially copyable types
// if -fpass-by-value-is-noalias is provided.

// WITH_NOALIAS: define{{.*}} void @_Z4take3Foo(%struct.Foo* noalias noundef %arg)
// NO_NOALIAS: define{{.*}} void @_Z4take3Foo(%struct.Foo* noundef %arg)
void take(Foo arg) {}

int G;

// NonTrivial is not trivially-copyable, because it has a non-trivial copy
// constructor.
struct NonTrivial {
  int a;
  int b;
  int c;
  int d;
  int e;
  int f;

  NonTrivial(const NonTrivial &Other) {
    a = G + 10 + Other.a;
  }
};

// Make sure noalias is not added to indirect arguments that are not trivially
// copyable even if -fpass-by-value-is-noalias is provided.

// WITH_NOALIAS: define{{.*}} void @_Z4take10NonTrivial(%struct.NonTrivial* noundef %arg)
// NO_NOALIAS:   define{{.*}} void @_Z4take10NonTrivial(%struct.NonTrivial* noundef %arg)
void take(NonTrivial arg) {}

// Escape examples. Pointers to the objects passed to take() may escape, depending on whether a temporary copy is created or not (e.g. due to NRVO).
struct A {
  A(A **where) : data{"hello world 1"} {
    *where = this; //Escaped pointer 1 (proposed UB?)
  }

  A() : data{"hello world 2"} {}

  char data[32];
};
A *p;

// WITH_NOALIAS: define{{.*}} void @_Z4take1A(%struct.A* noalias noundef %arg)
// NO_NOALIAS: define{{.*}} void @_Z4take1A(%struct.A* noundef %arg)
void take(A arg) {}

// WITH_NOALIAS: define{{.*}} void @_Z7CreateAPP1A(%struct.A* noalias sret(%struct.A) align 1 %agg.result, %struct.A** noundef %where)
// NO_NOALIAS: define{{.*}} void @_Z7CreateAPP1A(%struct.A* noalias sret(%struct.A) align 1 %agg.result, %struct.A** noundef %where)
A CreateA(A **where) {
  A justlikethis;
  *where = &justlikethis; //Escaped pointer 2 (should also be UB, then)
  return justlikethis;
}

// elsewhere, perhaps compiled by a smarter compiler that doesn't make a copy here
void test() {
  take({&p});        // 1
  take(CreateA(&p)); // 2
}
