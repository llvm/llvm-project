// RUN: %clang_cc1 -x c -triple bpf-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

struct s {
  int a;
  int b[4];
  int c:1;
};
union u {
  int a;
  int b[4];
  int c:1;
};
typedef struct {
  int a;
  int b;
} __t;
typedef int (*__f)(void);
enum AA {
  VAL1 = 10,
  VAL2 = 0xffffffff80000000UL,
};
typedef enum {
  VAL10 = 10,
  VAL11 = 11,
} __BB;

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

unsigned valid12() {
  const struct s t;
  return __builtin_preserve_type_info(t, 0) +
         __builtin_preserve_type_info(*(struct s *)0, 1);
}

unsigned valid13() {
  __t t;
  return __builtin_preserve_type_info(t, 1) +
         __builtin_preserve_type_info(*(__t *)0, 0);
}

unsigned valid14() {
  enum AA t;
  return __builtin_preserve_type_info(t, 0) +
         __builtin_preserve_type_info(*(enum AA *)0, 1);
}

unsigned valid15() {
  return __builtin_preserve_enum_value(*(enum AA *)VAL1, 1) +
         __builtin_preserve_enum_value(*(enum AA *)VAL2, 1);
}

unsigned invalid16() {
  return __builtin_preserve_enum_value(*(enum AA *)0, 1); // expected-error {{__builtin_preserve_enum_value argument 1 invalid}}
}

unsigned invalid17() {
  return __builtin_preserve_enum_value(*(enum AA *)VAL10, 1); // expected-error {{__builtin_preserve_enum_value argument 1 invalid}}
}

unsigned invalid18(struct s *arg) {
  return __builtin_preserve_type_info(arg->a + 2, 0); // expected-error {{__builtin_preserve_type_info argument 1 invalid}}
}
