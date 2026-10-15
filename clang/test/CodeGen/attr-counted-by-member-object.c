// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -Wno-gnu-variable-sized-type-not-at-end -emit-llvm -o - %s | FileCheck %s

// __builtin_dynamic_object_size(&p->field, 0) for a struct with a 'counted_by'
// flexible array member is
//
//   sizeof(struct) - offsetof(struct, field) + count * sizeof(element)
//
// where 'struct' must be the struct of the object that's accessed. It's found
// from the access expression, not from where the field's struct is declared.

#define __bdos(P) __builtin_dynamic_object_size(P, 0)

// The member belongs to a separately declared sub-struct. 'h.b' is at offset 8
// of the 12-byte 'struct sep', so the result is 4 + count * 4.
struct sep_hdr {
  int a;
  int b;
};

struct sep {
  int count;
  struct sep_hdr h;
  int fam[] __attribute__((counted_by(count)));
};

// CHECK-LABEL: define {{.*}} @test_separate_sub_struct(
// CHECK:       [[COUNT:%.*]] = load i32, ptr %p
// CHECK:       [[EXT:%.*]] = sext i32 [[COUNT]] to i64
// CHECK:       [[FAM:%.*]] = shl nsw i64 [[EXT]], 2
// CHECK:       add nsw i64 [[FAM]], 4
unsigned long test_separate_sub_struct(struct sep *p) {
  return __bdos(&p->h.b);
}

// CHECK-LABEL: define {{.*}} @test_separate_sub_struct_deref(
// CHECK:       add nsw i64 {{%.*}}, 4
unsigned long test_separate_sub_struct_deref(struct sep *p) {
  return __bdos(&(*p).h.b);
}

// CHECK-LABEL: define {{.*}} @test_separate_sub_struct_subscript(
// CHECK:       add nsw i64 {{%.*}}, 4
unsigned long test_separate_sub_struct_subscript(struct sep *p) {
  return __bdos(&p[1].h.b);
}

// The sub-struct is declared inside the struct with the flexible array member.
struct nested {
  int count;
  struct nested_hdr {
    int a;
    int b;
  } h;
  int fam[] __attribute__((counted_by(count)));
};

// CHECK-LABEL: define {{.*}} @test_nested_sub_struct(
// CHECK:       add nsw i64 {{%.*}}, 4
unsigned long test_nested_sub_struct(struct nested *p) {
  return __bdos(&p->h.b);
}

// A struct with a flexible array member, declared inside an over-aligned
// struct but accessed on its own. 'a' is at offset 4 of the 8-byte
// 'struct inner', so the result is 4 + count * 4, not 'struct outer''s
// 32 - (16 + 4) + count * 4.
struct outer {
  long x;
  _Alignas(16) struct inner {
    int count;
    int a;
    int fam[] __attribute__((counted_by(count)));
  } in;
};

// CHECK-LABEL: define {{.*}} @test_inner_struct(
// CHECK:       [[COUNT:%.*]] = load i32, ptr %q
// CHECK:       [[EXT:%.*]] = sext i32 [[COUNT]] to i64
// CHECK:       [[FAM:%.*]] = shl nsw i64 [[EXT]], 2
// CHECK:       add nsw i64 [[FAM]], 4
unsigned long test_inner_struct(struct inner *q) {
  return __bdos(&q->a);
}

// A struct with a flexible array member, declared inside a struct that only
// holds a pointer to it.
struct holder {
  long x;
  struct pinner {
    int count;
    int a;
    int fam[] __attribute__((counted_by(count)));
  } *p;
};

// CHECK-LABEL: define {{.*}} @test_pointer_only_inner_struct(
// CHECK:       [[COUNT:%.*]] = load i32, ptr %q
// CHECK:       [[EXT:%.*]] = sext i32 [[COUNT]] to i64
// CHECK:       [[FAM:%.*]] = shl nsw i64 [[EXT]], 2
// CHECK:       add nsw i64 [[FAM]], 4
unsigned long test_pointer_only_inner_struct(struct pinner *q) {
  return __bdos(&q->a);
}
