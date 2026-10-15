// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm -o - %s | FileCheck %s

// When a struct type with a 'counted_by' member is embedded more than once in
// another struct, the count must come from the copy that's accessed, not from
// the first one.

struct inner {
  int count;
  int *buf __attribute__((counted_by(count)));
};

struct outer {
  struct inner a, b;
};

// 'b.count' is at offset 16, 'a.count' at offset 0.

// CHECK-LABEL: define {{.*}} @test_bdos_deref(
// CHECK:       [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %o, i64 16
// CHECK:       load i32, ptr [[GEP]]
unsigned long test_bdos_deref(struct outer *o) {
  return __builtin_dynamic_object_size((*o).b.buf, 1);
}

// CHECK-LABEL: define {{.*}} @test_bdos_arrow(
// CHECK:       [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %o, i64 16
// CHECK:       load i32, ptr [[GEP]]
unsigned long test_bdos_arrow(struct outer *o) {
  return __builtin_dynamic_object_size(o->b.buf, 1);
}

// CHECK-LABEL: define {{.*}} @test_bdos_array(
// CHECK:       [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %o, i64 48
// CHECK:       load i32, ptr [[GEP]]
unsigned long test_bdos_array(struct outer *o) {
  return __builtin_dynamic_object_size(o[1].b.buf, 1);
}

// CHECK-LABEL: define {{.*}} @test_counted_by_ref(
// CHECK:       [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %o, i64 16
// CHECK:       store i32 %n, ptr [[GEP]]
void test_counted_by_ref(struct outer *o, int n) {
  *__builtin_counted_by_ref((*o).b.buf) = n;
}

// CHECK-LABEL: define {{.*}} @test_index(
// CHECK:       [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %o, i64 16
// CHECK:       [[COUNT:%.*]] = load i32, ptr [[GEP]]
// CHECK:       icmp ult i32 %i, [[COUNT]]
// CHECK:       call void @llvm.ubsantrap(
int test_index(struct outer *o, int i) {
  return (*o).b.buf[i];
}

// Doubly nested: the count is 'w.y.b.count', at offset 8 + 16.
struct wrapper {
  long x;
  struct outer y;
};

// CHECK-LABEL: define {{.*}} @test_bdos_nested(
// CHECK:       [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %w, i64 24
// CHECK:       load i32, ptr [[GEP]]
unsigned long test_bdos_nested(struct wrapper *w) {
  return __builtin_dynamic_object_size(w->y.b.buf, 1);
}
