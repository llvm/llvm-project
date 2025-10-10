// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// expected-no-diagnostics

void test_basic_type_checks() {
  static_assert(__is_same(char, decltype(__builtin_bswapg((char)0))), "");
  static_assert(__is_same(unsigned char, decltype(__builtin_bswapg((unsigned char)0))), "");
  static_assert(__is_same(short, decltype(__builtin_bswapg((short)0))), "");
  static_assert(__is_same(unsigned short, decltype(__builtin_bswapg((unsigned short)0))), "");
  static_assert(__is_same(int, decltype(__builtin_bswapg((int)0))), "");
  static_assert(__is_same(unsigned int, decltype(__builtin_bswapg((unsigned int)0))), "");
  static_assert(__is_same(long, decltype(__builtin_bswapg((long)0))), "");
  static_assert(__is_same(unsigned long, decltype(__builtin_bswapg((unsigned long)0))), "");
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
template void test_template_type_check<unsigned long>();

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
