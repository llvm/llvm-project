
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include <stdint.h>

int global;

void cb(int *__counted_by(count) p, int count);
void sb(void *__sized_by(size) p, int size);
void cb_multi(int *__counted_by(c1 - c2) p, int c1, int c2);
void cb_out(int *__counted_by(*count) p, int *count);
void cb_or_null(int *__counted_by_or_null(count) p, int count);
void sb_or_null(void *__sized_by_or_null(size) p, int size);

// Check if the pointer argument has an explicit attribute.

// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
void explicit_parm(int *__single p1, int l1, int *__counted_by(l2) p2, int l2) {
  cb(p1, l1);
  cb(p2, l2);

  int k1 = l1;
  int *__counted_by(k1) q1 = p1;

  int k2 = l2;
  int *__counted_by(k1) q2 = p2;

  q1 = p1;
  k1 = l1;

  q2 = p2;
  k2 = l2;
}

void explicit_struct(void) {
  struct s {
    // CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
    int *__single p1;
    int l1;
    // CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
    int *__counted_by(l2) p2;
    int l2;
  } s;
  cb(s.p1, s.l1);
  cb(s.p2, s.l2);

  int k1 = s.l1;
  int *__counted_by(k1) q1 = s.p1;

  int k2 = s.l2;
  int *__counted_by(k1) q2 = s.p2;

  q1 = s.p1;
  k1 = s.l1;

  q2 = s.p2;
  k2 = s.l2;
}

// Check if the pointer types are compatible.

void cb_i8(int8_t *__counted_by(len) p, int len);
void sb_i8(int8_t *__sized_by(size) p, int size);
void cb_or_null_i8(int8_t *__counted_by_or_null(len) p, int len);
void sb_or_null_i8(int8_t *__sized_by_or_null(size) p, int size);

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:34-[[@LINE+1]]:34}:"__counted_by(len) "
void cb_types_ok_passing(int8_t *p, int len) {
  cb_i8(p, len);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:31-[[@LINE+1]]:31}:"__counted_by(len) "
void cb_types_ok_init(int8_t *p, int len) {
  int l = len;
  int8_t *__counted_by(l) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:33-[[@LINE+1]]:33}:"__counted_by(len) "
void cb_types_ok_assign(int8_t *p, int len) {
  int l;
  int8_t *__counted_by(l) q;
  q = p;
  l = len;
}

// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
void cb_types_mismatch(int16_t *p, int len) {
  cb_i8(p, len);

  int l = len;
  int8_t *__counted_by(l) q = p;

  q = p;
  l = len;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:34-[[@LINE+1]]:34}:"__sized_by(size) "
void sb_types_ok_passing(int8_t *p, int size) {
  sb_i8(p, size);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:31-[[@LINE+1]]:31}:"__sized_by(size) "
void sb_types_ok_init(int8_t *p, int size) {
  int s = size;
  int8_t *__sized_by(s) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:33-[[@LINE+1]]:33}:"__sized_by(size) "
void sb_types_ok_assign(int8_t *p, int size) {
  int s;
  int8_t *__sized_by(s) q;
  q = p;
  s = size;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:36-[[@LINE+1]]:36}:"__sized_by(size) "
void sb_types_ok2_passing(int16_t *p, int size) {
  sb_i8(p, size); // size is in bytes, pointee size does not matter.
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:33-[[@LINE+1]]:33}:"__sized_by(size) "
void sb_types_ok2_init(int16_t *p, int size) {
  int s = size;
  int8_t *__sized_by(s) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:35-[[@LINE+1]]:35}:"__sized_by(size) "
void sb_types_ok2_assign(int16_t *p, int size) {
  int s;
  int8_t *__sized_by(s) q;
  q = p;
  s = size;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:42-[[@LINE+1]]:42}:"__counted_by_or_null(len) "
void cb_or_null_types_ok_passing(int8_t *p, int len) {
  cb_or_null_i8(p, len);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:39-[[@LINE+1]]:39}:"__counted_by_or_null(len) "
void cb_or_null_types_ok_init(int8_t *p, int len) {
  int l = len;
  int8_t *__counted_by_or_null(l) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:41-[[@LINE+1]]:41}:"__counted_by_or_null(len) "
void cb_or_null_types_ok_assign(int8_t *p, int len) {
  int l;
  int8_t *__counted_by_or_null(l) q;
  q = p;
  l = len;
}

// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
void cb_or_null_types_mismatch(int16_t *p, int len) {
  cb_or_null_i8(p, len);

  int l = len;
  int8_t *__counted_by_or_null(l) q = p;

  q = p;
  l = len;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:42-[[@LINE+1]]:42}:"__sized_by_or_null(size) "
void sb_or_null_types_ok_passing(int8_t *p, int size) {
  sb_or_null_i8(p, size);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:39-[[@LINE+1]]:39}:"__sized_by_or_null(size) "
void sb_or_null_types_ok_init(int8_t *p, int size) {
  int s = size;
  int8_t *__sized_by_or_null(s) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:41-[[@LINE+1]]:41}:"__sized_by_or_null(size) "
void sb_or_null_types_ok_assign(int8_t *p, int size) {
  int s;
  int8_t *__sized_by_or_null(s) q;

  q = p;
  s = size;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:44-[[@LINE+1]]:44}:"__sized_by_or_null(size) "
void sb_or_null_types_ok2_passing(int16_t *p, int size) {
  sb_or_null_i8(p, size); // size is in bytes, pointee size does not matter.
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:41-[[@LINE+1]]:41}:"__sized_by_or_null(size) "
void sb_or_null_types_ok2_init(int16_t *p, int size) {
  int s = size;
  int8_t *__sized_by_or_null(s) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:43-[[@LINE+1]]:43}:"__sized_by_or_null(size) "
void sb_or_null_types_ok2_assign(int16_t *p, int size) {
  int s;
  int8_t *__sized_by_or_null(s) q;

  q = p;
  s = size;
}

// Check syntax of count expression.

// TODO: Blocked on purpose until we have CodeGen tests.
// rdar://119737451
// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:29-[[@LINE+1]]:29}:"__counted_by(len + 2 * (len + 1) - 42) "
void syntax_ok_passing(int *p, int len) {
  cb(p, len + 2 * (len + 1) - 42);
}

// TODO: Blocked on purpose until we have CodeGen tests.
// rdar://119737451
// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:26-[[@LINE+1]]:26}:"__counted_by(len + 2 * (len + 1) - 42) "
void syntax_ok_init(int *p, int len) {
  int l = len + 2 * (len + 1) - 42;
  int *__counted_by(l) q = p;
}

// TODO: Blocked on purpose until we have CodeGen tests.
// rdar://119737451
// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:28-[[@LINE+1]]:28}:"__counted_by(len + 2 * (len + 1) - 42) "
void syntax_ok_assign(int *p, int len) {
  int l;
  int *__counted_by(l) q;

  q = p;
  l = len + 2 * (len + 1) - 42;
}

// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
void syntax_bad(int *p, int len) {
  cb(p, len == 42 ? 42 : len + 1); // ?: operator is unsupported.

  int l = len == 42 ? 42 : len + 1;
  int *__counted_by(l) q = p;

  q = p;
  l = len == 42 ? 42 : len + 1;
}

// Check if the decls in the count argument are parameters of the same function as the pointer argument.

// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
void decls_bad(int *p, int len) {
  int local;
  cb(p, local);
  cb(p, global);
  cb(p, len + local + global);
  cb_multi(p, len, local);
  cb_multi(p, global, len);

  int l1 = local;
  int *__counted_by(l1) q2 = p;
  q2 = p;
  l1 = local;

  int l2 = global;
  int *__counted_by(l2) q2 = p;
  q2 = p;
  l2 = global;

  int l3 = len + local + global;
  int *__counted_by(l3) q3 = p;
  q3 = p;
  l3 = len + local + global;

  int l4_a = len;
  int l4_b = local;
  int *__counted_by(l4 - l4_b) q4 = p;
  q4 = p;
  len4_a = len;
  len4_b = local;

  int l5_a = global;
  int l5_b = len;
  int *__counted_by(l5_a - l5_b) q5 = p;
  q5 = p;
  l5_a = global;
  l5_b = len;
}

// Check if the decls in the count argument are parameters of the same function as the pointer argument.

void member_base_ok_passing(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(len) "
    int *p;
    int len;
  } f;
  cb(f.p, f.len);
}

void member_base_ok_init(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(len) "
    int *p;
    int len;
  } f;
  int l = f.len;
  int *__counted_by(l) q = f.p;
}

void member_base_ok_assign(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(len) "
    int *p;
    int len;
  } f;
  int l;
  int *__counted_by(l) q;
  q = f.p;
  l = f.len;
}

void member_base_ok_ptr_passing(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(len) "
    int *p;
    int len;
  } *f;
  cb(f->p, f->len);
}

void member_base_ok_ptr_init(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(len) "
    int *p;
    int len;
  } *f;
  int l = f->len;
  int *__counted_by(l) q = f->p;
}

void member_base_ok_ptr_assign(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(len) "
    int *p;
    int len;
  } *f;
  int l;
  int *__counted_by(l) q;
  q = f->p;
  l = f->len;
}

void member_base_ok_multi_passing(void) {
  struct foo {
    // TODO: Blocked on purpose until we have CodeGen tests.
    // rdar://119737451
    // CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by((a + b) - (b - 42)) "
    int *p;
    int a;
    int b;
  } f;
  cb_multi(f.p, f.a + f.b, f.b - 42);
}

void member_base_ok_multi_init(void) {
  struct foo {
    // TODO: Blocked on purpose until we have CodeGen tests.
    // rdar://119737451
    // CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by((a + b) - (b - 42)) "
    int *p;
    int a;
    int b;
  } f;
  int l1 = f.a + f.b;
  int l2 = f.b - 42;
  int *__counted_by(l1 - l2) q = f.p;
}

void member_base_ok_multi_assign(void) {
  struct foo {
    // TODO: Blocked on purpose until we have CodeGen tests.
    // rdar://119737451
    // CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by((a + b) - (b - 42)) "
    int *p;
    int a;
    int b;
  } f;
  int l1;
  int l2;
  int *__counted_by(l1 - l2) q;
  q = f.p;
  l1 = f.a + f.b;
  l2 = f.b - 42;
}

void member_base_ok_nested_passing(void) {
  struct foo {
    struct bar {
      // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:12-[[@LINE+1]]:12}:"__counted_by(len) "
      int *p;
      int len;
    } b;
  } f;
  cb(f.b.p, f.b.len);
}

void member_base_ok_nested_init(void) {
  struct foo {
    struct bar {
      // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:12-[[@LINE+1]]:12}:"__counted_by(len) "
      int *p;
      int len;
    } b;
  } f;
  int l = f.b.len;
  int *__counted_by(l) q = f.b.p;
}

void member_base_ok_nested_assign(void) {
  struct foo {
    struct bar {
      // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:12-[[@LINE+1]]:12}:"__counted_by(len) "
      int *p;
      int len;
    } b;
  } f;
  int l;
  int *__counted_by(l) q;
  q = f.b.p;
  l = f.b.len;
}

void member_base_ok_ptr_nested_passing(void) {
  struct foo {
    struct bar {
      // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:12-[[@LINE+1]]:12}:"__counted_by(len) "
      int *p;
      int len;
    } *b;
  } *f;
  cb(f->b->p, f->b->len);
}

void member_base_ok_ptr_nested_init(void) {
  struct foo {
    struct bar {
      // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:12-[[@LINE+1]]:12}:"__counted_by(len) "
      int *p;
      int len;
    } *b;
  } *f;
  int l = f->b->len;
  int *__counted_by(l) q = f->b->p;
}

void member_base_ok_ptr_nested_assign(void) {
  struct foo {
    struct bar {
      // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:12-[[@LINE+1]]:12}:"__counted_by(len) "
      int *p;
      int len;
    } *b;
  } *f;
  int l;
  int *__counted_by(l) q;
  q = f->b->p;
  l = f->b->len;
}

struct member_base_foo_passing {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_ok_from_parm_passing(struct member_base_foo_passing *f) {
  cb(f->p, f->len);
}

struct member_base_foo_init {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_ok_from_parm_init(struct member_base_foo_init *f) {
  int l = f->len;
  int *__counted_by(l) q = f->p;
}

struct member_base_foo_assign {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_ok_from_parm_assign(struct member_base_foo_assign *f) {
  int l;
  int *__counted_by(l) q;
  q = f->p;
  l = f->len;
}

struct member_base_foo2_passing {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_ok_from_parm2_passing(struct member_base_foo2_passing f) {
  cb(f.p, f.len);
}

struct member_base_foo2_init {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_ok_from_parm2_init(struct member_base_foo2_init f) {
  int l = f.len;
  int *__counted_by(l) q = f.p;
}

struct member_base_foo2_assign {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_ok_from_parm2_assign(struct member_base_foo2_assign f) {
  int l;
  int *__counted_by(l) q;
  q = f.p;
  l = f.len;
}

void member_base_bad(int parm) {
  int local;
  struct foo {
    // CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
    int *p;
    int len;
  } f, f2;
  cb(f.p, local);
  cb(f.p, parm);
  cb(f.p, global);
  cb(f.p, f2.len);
  cb_multi(f.p, f.len, f2.len);
  cb_multi(f.p, f2.len, f.len);
  cb_multi(f.p, f.len, local);
  cb_multi(f.p, f.len, parm);
  cb_multi(f.p, global, f.len);

  int l1 = local;
  int *__counted_by(l1) q1 = f.p;
  l1 = parm;
  q1 = f.p;

  int l2 = global;
  int *__counted_by(l2) q2 = f.p;
  l2 = f2.len;
  q2 = f.p;

  int l3_a = f.len;
  int l3_b = f2.len;
  int *__counted_by(l3_a - l3_b) q3 = f.p;
  q3 = f.p;
  l3_a = f.len;
  l3_b = parm;
}

void member_base_bad2(int parm) {
  int local;
  struct foo {
    struct bar {
      // CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
      int *p;
      int len;
    } b;
    int size;
  } f, f2;
  cb(f.b.p, f.size);
  cb(f.b.p, f2.b.len);
  cb(f.b.p, f2.size);
  cb(f.b.p, f.b.len + f2.b.len);
  cb(f.b.p, f.b.len + f.size);
  cb(f.b.p, f.b.len + f2.size);
  cb(f.b.p, f.b.len + local);
  cb(f.b.p, f.b.len + parm);
  cb(f.b.p, f.b.len + global);
  cb_multi(f.b.p, f.b.len, f2.b.len);
  cb_multi(f.b.p, f.b.len, f.size);

  int l1 = f.size;
  int *__counted_by(l1) q1 = f.b.p;
  l1 = f2.b.len;
  q1 = f.b.p;

  int l2 = f.b.len + f2.b.len;
  int *__counted_by(l2) q2 = f.b.p;
  l2 = f.b.len + parm;
  q2 = f.b.p;

  int l3_a = f.b.len;
  int l3_b = f2.b.len;
  int *__counted_by(l3_a - l3_b) q3 = f.b.p;
  q3 = f.b.p;
  l3_a = f.b.len;
  l3_b = f.size;
}

struct member_base_global_foo_passing {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_global_ok_passing(void) {
  struct member_base_global_foo_passing f;
  cb(f.p, f.len);
}

struct member_base_global_foo_init {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_global_ok_init(void) {
  struct member_base_global_foo_init f;
  int l = f.len;
  int *__counted_by(l) q = f.p;
}

struct member_base_global_foo_assign {
  // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:8-[[@LINE+1]]:8}:"__counted_by(len) "
  int *p;
  int len;
};
void member_base_global_ok_assign(void) {
  struct member_base_global_foo_assign f;
  int l;
  int *__counted_by(l) q;
  l = f.len;
  q = f.p;
}

// Check if parentheses are emitted for complex expressions, so that we can
// avoid dealing with operator precedence.

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:18-[[@LINE+1]]:18}:"__counted_by(len) "
void parens(int *p, int len) {
  cb(p, len);
}

// TODO: Blocked on purpose until we have CodeGen tests.
// rdar://119737451
// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:19-[[@LINE+1]]:19}:"__counted_by(len + len2) "
void parens2(int *p, int len, int len2) {
  cb(p, len + len2);
}

// TODO: Blocked on purpose until we have CodeGen tests.
// rdar://119737451
// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:19-[[@LINE+1]]:19}:"__counted_by(len - len2) "
void parens3(int *p, int len, int len2) {
  cb_multi(p, len, len2);
}

// TODO: Blocked on purpose until we have CodeGen tests.
// rdar://119737451
// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:19-[[@LINE+1]]:19}:"__counted_by(len - (len + len2)) "
void parens4(int *p, int len, int len2) {
  cb_multi(p, len, len + len2);
}

// TODO: Blocked on purpose until we have CodeGen tests.
// rdar://119737451
// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:19-[[@LINE+1]]:19}:"__counted_by((len - 42) - (len + len2)) "
void parens5(int *p, int len, int len2) {
  cb_multi(p, len - 42, len + len2);
}

// Check out counts.

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:29-[[@LINE+1]]:29}:"__counted_by(*count) "
void out_to_in_passing(int *p, int *count) {
  cb(p, *count);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:26-[[@LINE+1]]:26}:"__counted_by(*count) "
void out_to_in_init(int *p, int *count) {
  int c = *count;
  int *__counted_by(c) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:28-[[@LINE+1]]:28}:"__counted_by(*count) "
void out_to_in_assign(int *p, int *count) {
  int c;
  int *__counted_by(c) q;
  q = p;
  c = *count;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:22-[[@LINE+1]]:22}:"__counted_by(*count) "
void out_to_out(int *p, int *count) {
  cb_out(p, count);
}

// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
void out_to_out_with_arith(int *p, int *count) {
  cb_out(p, count + 1);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:21-[[@LINE+1]]:21}:"__counted_by(count) "
void in_to_out(int *p, int count) {
  cb_out(p, &count);
}

// CHECK-NOT: fix-it:{{.+}}:{[[@LINE+1]]:
void in_to_out_with_arith(int *p, int count) {
  cb_out(p, &count + 1);
}

// Check *_or_null() variant.

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:33-[[@LINE+1]]:33}:"__counted_by_or_null(count) "
void to_cb_or_null_passing(int *p, int count) {
  cb_or_null(p, count);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:30-[[@LINE+1]]:30}:"__counted_by_or_null(count) "
void to_cb_or_null_init(int *p, int count) {
  int c = count;
  int *__counted_by_or_null(c) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:32-[[@LINE+1]]:32}:"__counted_by_or_null(count) "
void to_cb_or_null_assign(int *p, int count) {
  int c;
  int *__counted_by_or_null(c) q;
  q = p;
  c = count;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:33-[[@LINE+1]]:33}:"__sized_by_or_null(size) "
void to_sb_or_null_passing(int *p, int size) {
  sb_or_null(p, size);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:30-[[@LINE+1]]:30}:"__sized_by_or_null(size) "
void to_sb_or_null_init(int *p, int size) {
  int s = size;
  int *__sized_by_or_null(s) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:32-[[@LINE+1]]:32}:"__sized_by_or_null(size) "
void to_sb_or_null_assign(int *p, int size) {
  int s;
  int *__sized_by_or_null(s) q;
  q = p;
  s = size;
}

// Check constant counts.

void cb_const(int *__counted_by(16) p);
void sb_const(void *__sized_by(16) p);

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:36-[[@LINE+1]]:36}:"__counted_by(16) "
void to_cb_const_parm_passing(int *p) {
  cb_const(p);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:33-[[@LINE+1]]:33}:"__counted_by(16) "
void to_cb_const_parm_init(int *p) {
  int *__counted_by(16) q = p;
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:35-[[@LINE+1]]:35}:"__counted_by(16) "
void to_cb_const_parm_assign(int *p) {
  int *__counted_by(16) q;
  q = p;
}

void to_cb_const_struct(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(16) "
    int *p;
  } f;
  cb_const(f.p);
}

void to_cb_const_struct_ptr(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(16) "
    int *p;
  } *f;
  cb_const(f->p);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:29-[[@LINE+1]]:29}:"__sized_by(16) "
void to_sb_const_parm(void *p) {
  sb_const(p);
}

void to_sb_const_struct(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:11-[[@LINE+1]]:11}:"__sized_by(16) "
    void *p;
  } f;
  sb_const(f.p);
}

void to_sb_const_struct_ptr(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:11-[[@LINE+1]]:11}:"__sized_by(16) "
    void *p;
  } *f;
  sb_const(f->p);
}

// Check macros.

// TODO: We should suggest __counted_by(COUNT)/__sized_by(COUNT) instead.
// rdar://119737647

#define COUNT 16

void cb_const_macro(int *__counted_by(COUNT) p);
void sb_const_macro(void *__sized_by(COUNT) p);

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:34-[[@LINE+1]]:34}:"__counted_by(16) "
void to_cb_const_parm_macro(int *p) {
  cb_const_macro(p);
}

void to_cb_const_struct_macro(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(16) "
    int *p;
  } f;
  cb_const_macro(f.p);
}

void to_cb_const_struct_macro_ptr(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__counted_by(16) "
    int *p;
  } *f;
  cb_const_macro(f->p);
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:35-[[@LINE+1]]:35}:"__sized_by(16) "
void to_sb_const_parm_macro(void *p) {
  sb_const_macro(p);
}

void to_sb_const_struct_macro(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:11-[[@LINE+1]]:11}:"__sized_by(16) "
    void *p;
  } f;
  sb_const_macro(f.p);
}

void to_sb_const_struct_macro_ptr(void) {
  struct foo {
    // CHECK: fix-it:{{.+}}:{[[@LINE+1]]:11-[[@LINE+1]]:11}:"__sized_by(16) "
    void *p;
  } *f;
  sb_const_macro(f->p);
}

// Check init patterns.

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:23-[[@LINE+1]]:23}:"__counted_by(len) "
void init_struct(int *p, int len) {
  struct foo {
    int *__counted_by(l) q;
    int l;
  };
  struct foo f = { p, len };
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:24-[[@LINE+1]]:24}:"__counted_by(len) "
void init_struct2(int *p, int len) {
  struct foo {
    int *__counted_by(l) q;
    int l;
  } f = { p, len };
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:34-[[@LINE+1]]:34}:"__counted_by(len) "
void init_struct_designated(int *p, int len) {
  struct foo {
    int *__counted_by(l) q;
    int l;
  };
  struct foo f = { .q = p, .l = len };
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:35-[[@LINE+1]]:35}:"__counted_by(len) "
void init_struct_designated2(int *p, int len) {
  struct foo {
    int *__counted_by(l) q;
    int l;
  } f = { .q = p, .l = len };
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:30-[[@LINE+1]]:30}:"__counted_by(len) "
void init_struct_nested(int *p, int len) {
  struct bar {
    struct foo {
      int *__counted_by(l) q;
      int l;
    } f;
    int x;
  };
  struct bar b = { { p, len }, 0 };
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:41-[[@LINE+1]]:41}:"__counted_by(len) "
void init_struct_nested_designated(int *p, int len) {
  struct bar {
    struct foo {
      int *__counted_by(l) q;
      int l;
    } f;
    int x;
  };
  struct bar b = { .x = 0, .f = { .l = len, .q = p} };
}

// CHECK: fix-it:{{.+}}:{[[@LINE+1]]:22-[[@LINE+1]]:22}:"__counted_by(len) "
void init_array(int *p, int len) {
  struct foo {
    int *__counted_by(l) q;
    int l;
  };
  struct foo array[2] = { { p, len } };
}
