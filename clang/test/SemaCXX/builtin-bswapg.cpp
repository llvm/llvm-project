// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s

void test_basic_type_checks() {
  static_assert(__is_same(char, decltype(__builtin_bswapg((char)0))), "");
  static_assert(__is_same(unsigned char, decltype(__builtin_bswapg((unsigned char)0))), "");
  static_assert(__is_same(short, decltype(__builtin_bswapg((short)0))), "");
  static_assert(__is_same(unsigned short, decltype(__builtin_bswapg((unsigned short)0))), "");
  static_assert(__is_same(int, decltype(__builtin_bswapg((int)0))), "");
  static_assert(__is_same(unsigned int, decltype(__builtin_bswapg((unsigned int)0))), "");
  static_assert(__is_same(long, decltype(__builtin_bswapg((long)0))), "");
  static_assert(__is_same(unsigned long, decltype(__builtin_bswapg((unsigned long)0))), "");
	static_assert(__is_same(_BitInt(8), decltype(__builtin_bswapg((_BitInt(8))0))), "");
	static_assert(__is_same(_BitInt(16), decltype(__builtin_bswapg((_BitInt(16))0))), "");
	static_assert(__is_same(_BitInt(32), decltype(__builtin_bswapg((_BitInt(32))0))), "");
	static_assert(__is_same(_BitInt(64), decltype(__builtin_bswapg((_BitInt(64))0))), "");
	static_assert(__is_same(_BitInt(128), decltype(__builtin_bswapg((_BitInt(128))0))), "");
}

template<typename T>
void test_template_type_check() {
  static_assert(__is_same(T, decltype(__builtin_bswapg(T{}))), 
                "bswapg should return the same type as its argument");
  constexpr T zero{};
  constexpr T max = ~T{};
  constexpr T one = T{1};
    
  static_assert(__is_same(T, decltype(__builtin_bswapg(zero))), "");
  static_assert(__is_same(T, decltype(__builtin_bswapg(max))), "");
  static_assert(__is_same(T, decltype(__builtin_bswapg(one))), "");
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
    static_assert(__is_same(decltype(x), decltype(__builtin_bswapg(x))), 
                  "bswapg in lambda should preserve type");
    return __builtin_bswapg(x);
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
  return __builtin_bswapg(x);
}

void test_decltype_auto_check() {
  int x = 42;
  auto result = test_decltype_auto(x);
  static_assert(__is_same(int, decltype(result)), "");
}

template<auto Value>
struct ValueTemplateTypeTest {
  using value_type = decltype(Value);
  using result_type = decltype(__builtin_bswapg(Value));
    
  static constexpr bool type_matches = __is_same(value_type, result_type);
  static_assert(type_matches, "Value template bswapg should preserve type");
    
  static constexpr auto swapped_value = __builtin_bswapg(Value);
};

template<auto... Values>
void test_template_pack_types() {
  static_assert((__is_same(decltype(Values), decltype(__builtin_bswapg(Values))) && ...), "All pack elements should preserve type");
}

template struct ValueTemplateTypeTest<0x1234>;
template struct ValueTemplateTypeTest<0x12345678UL>;
template struct ValueTemplateTypeTest<(short)0x1234>;
template struct ValueTemplateTypeTest<(char)0x12>;

template<typename T>
void test_invalid_type() {
  __builtin_bswapg(T{}); // #invalid_type_use
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
auto test_dependent_context(T value) -> decltype(__builtin_bswapg(value)) { // #dependent_use
  return __builtin_bswapg(value); 
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
    return __builtin_bswapg(x); // #lambda_use
  };
  
  lambda(1.0f);
  // expected-error@#lambda_use {{1st argument must be a scalar integer type (was 'float')}}
  // expected-note@-2 {{in instantiation of function template specialization 'test_lambda_errors()::(anonymous class)::operator()<float>' requested here}}
  lambda(1.0l);
  // expected-error@#lambda_use {{1st argument must be a scalar integer type (was 'long double')}}
  // expected-note@-2 {{in instantiation of function template specialization 'test_lambda_errors()::(anonymous class)::operator()<long double>' requested here}}
  lambda("hello");
  // expected-error@#lambda_use {{1st argument must be a scalar integer type (was 'const char *')}}
  // expected-note@-2 {{in instantiation of function template specialization 'test_lambda_errors()::(anonymous class)::operator()<const char *>' requested here}}
}

template <class... Args> void test_variadic_template_argument_count(Args... args) {
   int result = __builtin_bswapg(args...); // #arg_use
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
  auto result = __builtin_bswapg(a);
  static_assert(__is_same(int, decltype(result)), "Should decay reference to value type");
}

void test_const_lvalue_reference(const int& a) {
  auto result = __builtin_bswapg(a);
  static_assert(__is_same(int, decltype(result)), "Should decay const reference to value type");
}

void test_rvalue_reference(int&& a) {
  auto result = __builtin_bswapg(a);
  static_assert(__is_same(int, decltype(result)), "Should decay rvalue reference to value type");
}

void test_const_rvalue_reference(const int&& a) {
	auto result = __builtin_bswapg(a);
  static_assert(__is_same(int, decltype(result)), "Should decay const rvalue reference to value type");
}

void test_array() {
  int arr[4] = {0x12, 0x34, 0x56, 0x78};
	__builtin_bswapg(arr);
	// expected-error@-1 {{1st argument must be a scalar integer type (was 'int[4]')}}
}

void test_pointer() {
  int x = 0x12345678;
	int *ptr = &x;
	__builtin_bswapg(ptr);
	// expected-error@-1 {{1st argument must be a scalar integer type (was 'int *')}}
}

enum BasicEnum {
  ENUM_VALUE1 = 0x1234,
  ENUM_VALUE2 = 0x34120000
};

void test_enum() {
	const BasicEnum e = ENUM_VALUE1;
	static_assert(__builtin_bswapg(e) == ENUM_VALUE2, "");
}

class testClass {
public:
  int value;
  testClass(int v) : value(v) {}

	int getValue() { return value; }
};

void test_class() {
	testClass c((int)0x12345678);
	__builtin_bswapg(c);
	// expected-error@-1 {{1st argument must be a scalar integer type (was 'testClass')}}
}

void test_nullptr() {
	__builtin_bswapg(nullptr);
	// expected-error@-1 {{1st argument must be a scalar integer type (was 'std::nullptr_t')}}
}

void test_bitint() {
  static_assert(__builtin_bswapg((_BitInt(8))0x12) == (_BitInt(8))0x12, "");
  static_assert(__builtin_bswapg((_BitInt(16))0x1234) == (_BitInt(16))0x3412, "");
  static_assert(__builtin_bswapg((_BitInt(32))0x00001234) == (_BitInt(32))0x34120000, "");
  static_assert(__builtin_bswapg((_BitInt(64))0x0000000000001234) == (_BitInt(64))0x3412000000000000, "");
  static_assert(__builtin_bswapg(~(_BitInt(128))0) == (~(_BitInt(128))0), "");
  static_assert(__builtin_bswapg((_BitInt(24))0x1234) == (_BitInt(24))0x3412, "");
  // expected-error@-1 {{_BitInt type '_BitInt(24)' (24 bits) must be a multiple of 16 bits for byte swapping}}
}
