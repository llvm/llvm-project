// RUN: %clang_cc1 -fsyntax-only -Wreturn-type -Wunreachable-code -Warray-bounds -Wconditional-uninitialized -verify=default,expected %s
// RUN: %clang_cc1 -fsyntax-only -Wreturn-type -Wunreachable-code -Warray-bounds -Wconditional-uninitialized -fno-strict-enum-switch-coverage -verify=default,expected %s
// RUN: %clang_cc1 -fsyntax-only -Wreturn-type -Wunreachable-code -Warray-bounds -Wconditional-uninitialized -fstrict-enum-switch-coverage -verify=strict,expected %s

enum E { A, B };

int test_missing_return(E e) {
  switch (e) {
    case A: return 1;
    case B: return 2;
  }
} // strict-warning {{non-void function does not return a value in all control paths}}

int test_unreachable(E e) {
  switch (e) {
    case A: return 1;
    case B: return 2;
  }
  return 3;
}

void test_array_bounds(E e) {
  int x[2]; // strict-note {{array 'x' declared here}}
  switch (e) {
    case A: return;
    case B: return;
  }
  x[2] = 0; // strict-warning {{array index 2 is past the end of the array (that has type 'int[2]')}}
}

int test_useless_default_no_unreachable_warning(E e) {
  switch (e) {
    case A: return 1;
    case B: return 2;
    default: return 3;
  }
}

int test_nested(E e1, E e2) {
  switch (e1) {
    case A:
      switch (e2) {
        case A: return 1;
        case B: return 2;
      }
      break; // break out of outer switch if e2 falls through!
    case B:
      return 3;
  }
} // strict-warning {{non-void function does not return a value in all control paths}}

enum class BoolEnum : bool { False = false, True = true };

int test_full_range_exhaustive(BoolEnum e) {
  switch (e) {
    case BoolEnum::False: return 0;
    case BoolEnum::True: return 1;
  }
}

int test_useless_default_full_range_exhaustive(BoolEnum e) {
  switch (e) {
    case BoolEnum::False: return 0;
    case BoolEnum::True: return 1;
    default: return 2;
  }
}

int test_bool(bool b) {
  switch (b) { // expected-warning {{switch condition has boolean value}}
    case false: return 0;
    case true: return 1;
  }
} // default-warning {{non-void function does not return a value in all control paths}}

enum class SignedCharEnum : signed char { Min = -128, Max = 127 };

int test_signed_char_exhaustive(SignedCharEnum e) {
  switch (e) {
    case SignedCharEnum::Min ... SignedCharEnum::Max: return 0;
  }
}

enum class BoolEnumMissing : bool { False = false };

int test_bool_enum_missing(BoolEnumMissing e) {
  switch (e) {
    case BoolEnumMissing::False: return 0;
  }
} // strict-warning {{non-void function does not return a value in all control paths}}

int test_uninit_exhaustive(E e) {
  int x; // strict-note {{initialize the variable 'x' to silence this warning}}
  switch (e) {
    case A: x = 1; break;
    case B: x = 2; break;
  }
  return x; // strict-warning {{variable 'x' may be uninitialized when used here}}
}

int test_uninit_missing(BoolEnumMissing e) {
  int x; // strict-note {{initialize the variable 'x' to silence this warning}}
  switch (e) {
    case BoolEnumMissing::False: x = 1; break;
  }
  return x; // strict-warning {{variable 'x' may be uninitialized when used here}}
}

enum EmptyEnum {};

int test_empty_enum(EmptyEnum e) {
  switch (e) {
  }
} // expected-warning {{non-void function does not return a value}}

int test_empty_switch_bool(bool b) {
  switch (b) { // expected-warning {{switch condition has boolean value}}
  }
} // expected-warning {{non-void function does not return a value}}

template <typename T>
int test_template(T e) {
  switch (e) {
    case (T)0: return 1;
    case (T)1: return 2;
  }
} // strict-warning {{non-void function does not return a value in all control paths}}

template int test_template<E>(E); // strict-note {{in instantiation of function template specialization 'test_template<E>' requested here}}
template int test_template<BoolEnum>(BoolEnum);

enum class U32 : unsigned int { Min = 0, Max = 0xFFFFFFFF };

int test_overflow(U32 e) {
  switch (e) {
    case U32::Min ... U32::Max: return 0;
  }
}
