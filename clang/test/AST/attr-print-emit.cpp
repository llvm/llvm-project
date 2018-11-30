// RUN: %clang_cc1 %s -ast-print | FileCheck %s
// RUN: %clang -emit-ast -o %t.ast %s
// RUN: %clang_cc1 %t.ast -ast-print | FileCheck %s

// CHECK: void xla(int a) __attribute__((xray_log_args(1)));
void xla(int a) __attribute__((xray_log_args(1)));

// CHECK: void *as2(int, int) __attribute__((alloc_size(1, 2)));
void *as2(int, int) __attribute__((alloc_size(1, 2)));
// CHECK: void *as1(void *, int) __attribute__((alloc_size(2)));
void *as1(void *, int) __attribute__((alloc_size(2)));

// CHECK: void fmt(int, const char *, ...) __attribute__((format(printf, 2, 3)));
void fmt(int, const char *, ...) __attribute__((format(printf, 2, 3)));

// CHECK: char *fmta(int, const char *) __attribute__((format_arg(2)));
char *fmta(int, const char *) __attribute__((format_arg(2)));

// CHECK: void nn(int *, int *) __attribute__((nonnull(1, 2)));
void nn(int *, int *) __attribute__((nonnull(1, 2)));

// CHECK: int *aa(int i) __attribute__((alloc_align(1)));
int *aa(int i) __attribute__((alloc_align(1)));

// CHECK: void ownt(int *, int *) __attribute__((ownership_takes(foo, 1, 2)));
void ownt(int *, int *) __attribute__((ownership_takes(foo, 1, 2)));
// CHECK: void ownh(int *, int *) __attribute__((ownership_holds(foo, 1, 2)));
void ownh(int *, int *) __attribute__((ownership_holds(foo, 1, 2)));
// CHECK: void ownr(int) __attribute__((ownership_returns(foo, 1)));
void ownr(int) __attribute__((ownership_returns(foo, 1)));

// CHECK: void awtt(int, int, ...) __attribute__((argument_with_type_tag(foo, 3, 2)));
void awtt(int, int, ...) __attribute__((argument_with_type_tag(foo, 3, 2)));
// CHECK: void pwtt(void *, int) __attribute__((pointer_with_type_tag(foo, 1, 2)));
void pwtt(void *, int) __attribute__((pointer_with_type_tag(foo, 1, 2)));

class C {
  // CHECK: void xla(int a) __attribute__((xray_log_args(2)));
  void xla(int a) __attribute__((xray_log_args(2)));

  // CHECK: void *as2(int, int) __attribute__((alloc_size(2, 3)));
  void *as2(int, int) __attribute__((alloc_size(2, 3)));
  // CHECK: void *as1(void *, int) __attribute__((alloc_size(3)));
  void *as1(void *, int) __attribute__((alloc_size(3)));

  // CHECK: void fmt(int, const char *, ...) __attribute__((format(printf, 3, 4)));
  void fmt(int, const char *, ...) __attribute__((format(printf, 3, 4)));

  // CHECK: char *fmta(int, const char *) __attribute__((format_arg(3)));
  char *fmta(int, const char *) __attribute__((format_arg(3)));

  // CHECK: void nn(int *, int *) __attribute__((nonnull(2, 3)));
  void nn(int *, int *) __attribute__((nonnull(2, 3)));

  // CHECK: int *aa(int i) __attribute__((alloc_align(2)));
  int *aa(int i) __attribute__((alloc_align(2)));

  // CHECK: void ownt(int *, int *) __attribute__((ownership_takes(foo, 2, 3)));
  void ownt(int *, int *) __attribute__((ownership_takes(foo, 2, 3)));
  // CHECK: void ownh(int *, int *) __attribute__((ownership_holds(foo, 2, 3)));
  void ownh(int *, int *) __attribute__((ownership_holds(foo, 2, 3)));
  // CHECK: void ownr(int) __attribute__((ownership_returns(foo, 2)));
  void ownr(int) __attribute__((ownership_returns(foo, 2)));

  // CHECK: void awtt(int, int, ...) __attribute__((argument_with_type_tag(foo, 4, 3)));
  void awtt(int, int, ...) __attribute__((argument_with_type_tag(foo, 4, 3)));
  // CHECK: void pwtt(void *, int) __attribute__((pointer_with_type_tag(foo, 2, 3)));
  void pwtt(void *, int) __attribute__((pointer_with_type_tag(foo, 2, 3)));
};
