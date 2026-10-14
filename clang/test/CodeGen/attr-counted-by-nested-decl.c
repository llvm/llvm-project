// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm -o - %s | FileCheck %s --check-prefix=O0

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

// Stepping back from 'buf' to the start of the struct that holds the count
// must land on 'struct inner' (8 bytes back), not on a 'struct outer' that
// needn't exist (16 bytes back, before the start of the object).
// O0-LABEL: define {{.*}} @test_embedded(
// O0:       [[BUF:%.*]] = getelementptr inbounds nuw %struct.inner, ptr {{%.*}}, i32 0, i32 1
// O0:       [[REC:%.*]] = getelementptr inbounds i8, ptr [[BUF]], i64 -8
// O0:       [[COUNT:%.*]] = getelementptr inbounds nuw %struct.inner, ptr [[REC]], i32 0, i32 0
// O0:       load i32, ptr [[COUNT]]
// CHECK-LABEL: define {{.*}} @test_embedded(
// CHECK:       [[COUNT:%.*]] = load i32, ptr %q
// CHECK:       icmp ult i32 %i, [[COUNT]]
// CHECK:       call void @llvm.ubsantrap(
int test_embedded(struct inner *q, int i) {
  return q->buf[i];
}

// The same inner struct embedded twice. The count must be reached through the
// object that's actually accessed, so that its TBAA access tag is right.
// Tagging it as 'pair.a.count' when 'pair.b.count' is accessed would let the
// optimizer reuse the first count after the store through 'u', which may
// alias 't'.
struct pair {
  struct pinner2 {
    int count;
    int *buf __attribute__((counted_by(count)));
  } a, b;
};

// CHECK-LABEL: define {{.*}} @test_tbaa(
// CHECK:       [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %t, i64 16
// CHECK:       [[COUNT1:%.*]] = load i32, ptr [[GEP]]
// CHECK:       store i32 %n
// CHECK:       [[COUNT2:%.*]] = load i32, ptr [[GEP]]
// CHECK:       icmp ult i32 %i, [[COUNT2]]
int test_tbaa(struct pair *t, struct pair *u, int n, int i) {
  int x = t->b.buf[0];
  u->b.count = n;
  return x + t->b.buf[i];
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
