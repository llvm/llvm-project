// RUN: %clang_cc1 -x c -triple bpf-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

struct s { int a; int b[4]; int c:1; };
union u { int a; int b[4]; int c:1; };

unsigned invalid1(const int *arg) {
  return __builtin_preserve_field_info(arg, 1); // expected-error {{__builtin_preserve_field_info argument 1 not a field access}}
}

unsigned invalid2(const int *arg) {
  return __builtin_preserve_field_info(*arg, 1); // expected-error {{__builtin_preserve_field_info argument 1 not a field access}}
}

void *invalid3(struct s *arg) {
  return __builtin_preserve_field_info(arg->a, 1); // expected-warning {{incompatible integer to pointer conversion returning 'unsigned int' from a function with result type 'void *'}}
}

unsigned valid4(struct s *arg) {
  return __builtin_preserve_field_info(arg->b[1], 1);
}

unsigned valid5(union u *arg) {
  return __builtin_preserve_field_info(arg->b[2], 1);
}

unsigned valid6(struct s *arg) {
  return __builtin_preserve_field_info(arg->a, 1);
}

unsigned valid7(struct s *arg) {
  return __builtin_preserve_field_info(arg->c, 1ULL);
}

unsigned valid8(union u *arg) {
  return __builtin_preserve_field_info(arg->a, 1);
}

unsigned valid9(union u *arg) {
  return __builtin_preserve_field_info(arg->c, 'a');
}

unsigned invalid10(struct s *arg) {
  return __builtin_preserve_field_info(arg->a, arg); // expected-error {{__builtin_preserve_field_info argument 2 not a constant}}
}

unsigned invalid11(struct s *arg, int info_kind) {
  return __builtin_preserve_field_info(arg->a, info_kind); // expected-error {{__builtin_preserve_field_info argument 2 not a constant}}
}
