// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm -o - %s | FileCheck %s

// A struct defined lexically inside another struct is a type of its own. The
// -fsanitize=array-bounds check for a 'counted_by' member must find the count
// in the struct that actually contains the member, not in the outermost
// lexically enclosing one.

// The inner struct is embedded in the outer one, but is accessed on its own.
struct outer {
  long x;
  struct inner {
    int count;
    int *buf __attribute__((counted_by(count)));
  } in;
};

// CHECK-LABEL: define {{.*}} @test_embedded(
// CHECK:       [[COUNT:%.*]] = load i32, ptr %q
// CHECK:       icmp ult i32 %i, [[COUNT]]
// CHECK:       call void @llvm.ubsantrap(
int test_embedded(struct inner *q, int i) {
  return q->buf[i];
}

// The inner struct only appears in the outer one through a pointer, so the
// outer struct doesn't contain the 'counted_by' member at all.
struct holder {
  struct pinner {
    int count;
    int *buf __attribute__((counted_by(count)));
  } *ptr;
};

// CHECK-LABEL: define {{.*}} @test_by_pointer(
// CHECK:       [[COUNT:%.*]] = load i32, ptr %q
// CHECK:       icmp ult i32 %i, [[COUNT]]
// CHECK:       call void @llvm.ubsantrap(
int test_by_pointer(struct pinner *q, int i) {
  return q->buf[i];
}

// The same holds for a flexible array member.
struct fam_holder {
  struct fam_inner *next;
  struct fam_inner {
    int count;
    int fam[] __attribute__((counted_by(count)));
  } *fi;
};

// CHECK-LABEL: define {{.*}} @test_fam_by_pointer(
// CHECK:       [[COUNT:%.*]] = load i32, ptr %q
// CHECK:       icmp ult i32 %i, [[COUNT]]
// CHECK:       call void @llvm.ubsantrap(
int test_fam_by_pointer(struct fam_inner *q, int i) {
  return q->fam[i];
}

// Anonymous members still belong to the enclosing struct.
struct anon {
  struct {
    long pad;
    int count;
  };
  int *buf __attribute__((counted_by(count)));
};

// CHECK-LABEL: define {{.*}} @test_anon(
// CHECK:       [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %a, i64 8
// CHECK:       [[COUNT:%.*]] = load i32, ptr [[GEP]]
// CHECK:       icmp ult i32 %i, [[COUNT]]
// CHECK:       call void @llvm.ubsantrap(
int test_anon(struct anon *a, int i) {
  return a->buf[i];
}
