// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s

void test_basic_type_checks() {
  static_assert(__is_same(char, decltype(__builtin_bitreverseg((char)0))), "");
  static_assert(__is_same(unsigned char, decltype(__builtin_bitreverseg((unsigned char)0))), "");
  static_assert(__is_same(short, decltype(__builtin_bitreverseg((short)0))), "");
  static_assert(__is_same(unsigned short, decltype(__builtin_bitreverseg((unsigned short)0))), "");
  static_assert(__is_same(int, decltype(__builtin_bitreverseg((int)0))), "");
  static_assert(__is_same(unsigned int, decltype(__builtin_bitreverseg((unsigned int)0))), "");
  static_assert(__is_same(long, decltype(__builtin_bitreverseg((long)0))), "");
  static_assert(__is_same(unsigned long, decltype(__builtin_bitreverseg((unsigned long)0))), "");
  static_assert(__is_same(_BitInt(8), decltype(__builtin_bitreverseg((_BitInt(8))0))), "");
  static_assert(__is_same(_BitInt(16), decltype(__builtin_bitreverseg((_BitInt(16))0))), "");
  static_assert(__is_same(_BitInt(32), decltype(__builtin_bitreverseg((_BitInt(32))0))), "");
  static_assert(__is_same(_BitInt(64), decltype(__builtin_bitreverseg((_BitInt(64))0))), "");
  static_assert(__is_same(_BitInt(128), decltype(__builtin_bitreverseg((_BitInt(128))0))), "");
}

template<typename T>
void test_template_type_check() {
  static_assert(__is_same(T, decltype(__builtin_bitreverseg(T{}))),
                "bitreverseg should return the same type as its argument");
  constexpr T zero{};
  constexpr T max = ~T{};
  constexpr T one = T{1};

  static_assert(__is_same(T, decltype(__builtin_bitreverseg(zero))), "");
  static_assert(__is_same(T, decltype(__builtin_bitreverseg(max))), "");
  static_assert(__is_same(T, decltype(__builtin_bitreverseg(one))), "");
}

template void test_template_type_check<char>();
template void test_template_type_check<unsigned char>();
template void test_template_type_check<short>();
template void test_template_type_check<unsigned short>();
template void test_template_type_check<int>();
template void test_template_type_check<unsigned int>();
template void test_template_type_check<long>();
template void test_template_type_check<_BitInt(8)>();
template void test_template_type_check<_BitInt(16)>();
template void test_template_type_check<_BitInt(32)>();
template void test_template_type_check<_BitInt(64)>();
template void test_template_type_check<_BitInt(128)>();

void test_lambda_type_checks() {
  auto lambda = [](auto x) {
    static_assert(__is_same(decltype(x), decltype(__builtin_bitreverseg(x))),
                  "bitreverseg in lambda should preserve type");
    return __builtin_bitreverseg(x);
  };
  auto result_long = lambda(42UL);
  static_assert(__is_same(unsigned long, decltype(result_long)), "");

  auto result_int = lambda(42);
  static_assert(__is_same(int, decltype(result_int)), "");

  auto result_short = lambda(static_cast<short>(42));
  static_assert(__is_same(short, decltype(result_short)), "");

  auto result_char = lambda(static_cast<char>(42));
  static_assert(__is_same(char, decltype(result_char)), "");
}

decltype(auto) test_decltype_auto(int x) {
  return __builtin_bitreverseg(x);
}

void test_decltype_auto_check() {
  int x = 42;
  auto result = test_decltype_auto(x);
  static_assert(__is_same(int, decltype(result)), "");
}

template<auto Value>
struct ValueTemplateTypeTest {
  using value_type = decltype(Value);
  using result_type = decltype(__builtin_bitreverseg(Value));

  static constexpr bool type_matches = __is_same(value_type, result_type);
  static_assert(type_matches, "Value template bitreverseg should preserve type");

  static constexpr auto reversed_value = __builtin_bitreverseg(Value);
};

template<auto... Values>
void test_template_pack_types() {
  static_assert((__is_same(decltype(Values), decltype(__builtin_bitreverseg(Values))) && ...),
                "All pack elements should preserve type");
}

template struct ValueTemplateTypeTest<0x1234>;
template struct ValueTemplateTypeTest<0x12345678UL>;
template struct ValueTemplateTypeTest<(short)0x1234>;
template struct ValueTemplateTypeTest<(char)0x12>;

template<typename T>
void test_invalid_type() {
  __builtin_bitreverseg(T{}); // #invalid_type_use
}

void test_basic_errors() {
  test_invalid_type<float>();
  // expected-note@-1 {{in instantiation of function template specialization 'test_invalid_type<float>' requested here}}
  // expected-error@#invalid_type_use {{1st argument must be a scalar integer type (was 'float')}}

  test_invalid_type<double>();
  // expected-note@-1 {{in instantiation of function template specialization 'test_invalid_type<double>' requested here}}
  // expected-error@#invalid_type_use {{1st argument must be a scalar integer type (was 'double')}}

  test_invalid_type<void*>();
  // expected-note@-1 {{in instantiation of function template specialization 'test_invalid_type<void *>' requested here}}
  // expected-error@#invalid_type_use {{1st argument must be a scalar integer type (was 'void *')}}
}

template<typename T>
auto test_dependent_context(T value) -> decltype(__builtin_bitreverseg(value)) { // #dependent_use
  return __builtin_bitreverseg(value);
}

void test_dependent_errors() {
  test_dependent_context(1.0f);
  // expected-error@-1 {{no matching function for call to 'test_dependent_context'}}
  // expected-note@#dependent_use {{candidate template ignored: substitution failure [with T = float]: 1st argument must be a scalar integer type (was 'float')}}
  test_dependent_context(1.0l);
  // expected-error@-1 {{no matching function for call to 'test_dependent_context'}}
  // expected-note@#dependent_use {{candidate template ignored: substitution failure [with T = long double]: 1st argument must be a scalar integer type (was 'long double')}}
  test_dependent_context("hello");
  // expected-error@-1 {{no matching function for call to 'test_dependent_context'}}
  // expected-note@#dependent_use {{candidate template ignored: substitution failure [with T = const char *]: 1st argument must be a scalar integer type (was 'const char *')}}
}

void test_lambda_errors() {
  auto lambda = [](auto x) {
    return __builtin_bitreverseg(x); // #lambda_use
  };

  lambda(1.0f);
  // expected-error@#lambda_use {{1st argument must be a scalar integer type (was 'float')}}
  // expected-note@-2 {{in instantiation of function template specialization 'test_lambda_errors()::(lambda)::operator()<float>' requested here}}
  lambda(1.0l);
  // expected-error@#lambda_use {{1st argument must be a scalar integer type (was 'long double')}}
  // expected-note@-2 {{in instantiation of function template specialization 'test_lambda_errors()::(lambda)::operator()<long double>' requested here}}
  lambda("hello");
  // expected-error@#lambda_use {{1st argument must be a scalar integer type (was 'const char *')}}
  // expected-note@-2 {{in instantiation of function template specialization 'test_lambda_errors()::(lambda)::operator()<const char *>' requested here}}
}

template <class... Args> void test_variadic_template_argument_count(Args... args) {
  int result = __builtin_bitreverseg(args...); // #arg_use
}

void test_variadic_template_args() {
  test_variadic_template_argument_count();
  // expected-error@#arg_use {{too few arguments to function call, expected 1, have 0}}
  // expected-note@-2 {{in instantiation of function template specialization 'test_variadic_template_argument_count<>' requested here}}
  test_variadic_template_argument_count(1);
  test_variadic_template_argument_count(1, 2);
  // expected-error@#arg_use {{too many arguments to function call, expected 1, have 2}}
  // expected-note@-2 {{in instantiation of function template specialization 'test_variadic_template_argument_count<int, int>' requested here}}
}

void test_lvalue_reference(int& a) {
  auto result = __builtin_bitreverseg(a);
  static_assert(__is_same(int, decltype(result)), "Should decay reference to value type");
}

void test_const_lvalue_reference(const int& a) {
  auto result = __builtin_bitreverseg(a);
  static_assert(__is_same(int, decltype(result)), "Should decay const reference to value type");
}

void test_rvalue_reference(int&& a) {
  auto result = __builtin_bitreverseg(a);
  static_assert(__is_same(int, decltype(result)), "Should decay rvalue reference to value type");
}

void test_const_rvalue_reference(const int&& a) {
  auto result = __builtin_bitreverseg(a);
  static_assert(__is_same(int, decltype(result)), "Should decay const rvalue reference to value type");
}

void test_array() {
  int arr[4] = {0x12, 0x34, 0x56, 0x78};
  __builtin_bitreverseg(arr);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'int[4]')}}
}

void test_pointer() {
  int x = 0x12345678;
  int *ptr = &x;
  __builtin_bitreverseg(ptr);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'int *')}}
}

enum BasicEnum {
  ENUM_VALUE1 = 0x1234,
  ENUM_VALUE2 = 0x2C480000
};

void test_enum() {
  const BasicEnum e = ENUM_VALUE1;
  static_assert(__builtin_bitreverseg(e) == ENUM_VALUE2, "");
}

class testClass {
public:
  int value;
  testClass(int v) : value(v) {}

  int getValue() { return value; }
};

void test_class() {
  testClass c((int)0x12345678);
  __builtin_bitreverseg(c);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'testClass')}}
}

void test_nullptr() {
  __builtin_bitreverseg(nullptr);
  // expected-error@-1 {{1st argument must be a scalar integer type (was 'std::nullptr_t')}}
}

void test_bitint() {
  static_assert(__builtin_bitreverseg((unsigned _BitInt(1))1) == (unsigned _BitInt(1))1, "");
  static_assert(__builtin_bitreverseg((_BitInt(8))0x12) == (_BitInt(8))0x48, "");
  static_assert(__builtin_bitreverseg((_BitInt(16))0x1234) == (_BitInt(16))0x2C48, "");
  static_assert(__builtin_bitreverseg((_BitInt(32))0x00001234) == (_BitInt(32))0x2C480000, "");
  static_assert(__builtin_bitreverseg((_BitInt(64))0x0000000000001234) == (_BitInt(64))0x2C48000000000000, "");
  static_assert(__builtin_bitreverseg((_BitInt(128))0x1) == ((_BitInt(128))1 << 127), "");
  static_assert(__builtin_bitreverseg((_BitInt(24))0x1234) == (_BitInt(24))0x2C4800, "");
}
