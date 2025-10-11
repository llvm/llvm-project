// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fexperimental-new-constant-interpreter %s
// expected-no-diagnostics
template <class A, class B> 
static constexpr bool is_same_type = false;

template <class A> 
static constexpr bool is_same_type<A, A> = true;

void test_basic_type_checks() {
  static_assert(is_same_type<char, decltype(__builtin_bswapg((char)0))>, "");
  static_assert(is_same_type<unsigned char, decltype(__builtin_bswapg((unsigned char)0))>, "");
  static_assert(is_same_type<short, decltype(__builtin_bswapg((short)0))>, "");
  static_assert(is_same_type<unsigned short, decltype(__builtin_bswapg((unsigned short)0))>, "");
  static_assert(is_same_type<int, decltype(__builtin_bswapg((int)0))>, "");
  static_assert(is_same_type<unsigned int, decltype(__builtin_bswapg((unsigned int)0))>, "");
  static_assert(is_same_type<long, decltype(__builtin_bswapg((long)0))>, "");
  static_assert(is_same_type<unsigned long, decltype(__builtin_bswapg((unsigned long)0))>, "");
}

template<typename T>
void test_template_type_check() {
  static_assert(is_same_type<T, decltype(__builtin_bswapg(T{}))>, 
                "bswapg should return the same type as its argument");
  constexpr T zero{};
  constexpr T max = ~T{};
  constexpr T one = T{1};
    
  static_assert(is_same_type<T, decltype(__builtin_bswapg(zero))>, "");
  static_assert(is_same_type<T, decltype(__builtin_bswapg(max))>, "");
  static_assert(is_same_type<T, decltype(__builtin_bswapg(one))>, "");
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
    static_assert(is_same_type<decltype(x), decltype(__builtin_bswapg(x))>, 
                  "bswapg in lambda should preserve type");
    return __builtin_bswapg(x);
  };
  auto result_long = lambda(42UL);
  static_assert(is_same_type<unsigned long, decltype(result_long)>, "");
    
  auto result_int = lambda(42);
  static_assert(is_same_type<int, decltype(result_int)>, "");
    
  auto result_short = lambda(static_cast<short>(42));
  static_assert(is_same_type<short, decltype(result_short)>, "");

  auto result_char = lambda(static_cast<char>(42));
  static_assert(is_same_type<char, decltype(result_char)>, "");
}

auto test_auto_return_type_long(long x) {
  auto result = __builtin_bswapg(x);
  static_assert(is_same_type<long, decltype(result)>, "");
  return result;
}

auto test_auto_return_type_int(int x) {
  auto result = __builtin_bswapg(x);
  static_assert(is_same_type<int, decltype(result)>, "");
  return result;
}

auto test_auto_return_type_short(short x) {
  auto result = __builtin_bswapg(x);
  static_assert(is_same_type<short, decltype(result)>, "");
  return result;
}

auto test_auto_return_type_char(char x) {
  auto result = __builtin_bswapg(x);
  static_assert(is_same_type<char, decltype(result)>, "");
  return result;
}

void test_auto_return_type() {
  test_auto_return_type_long(42);
  test_auto_return_type_int(42);
  test_auto_return_type_short(42);
  test_auto_return_type_char(42);
}

decltype(auto) test_decltype_auto(int x) {
  return __builtin_bswapg(x);
}

void test_decltype_auto_check() {
  int x = 42;
  auto result = test_decltype_auto(x);
  static_assert(is_same_type<int, decltype(result)>, "");
}

template<auto Value>
struct ValueTemplateTypeTest {
  using value_type = decltype(Value);
  using result_type = decltype(__builtin_bswapg(Value));
    
  static constexpr bool type_matches = is_same_type<value_type, result_type>;
  static_assert(type_matches, "Value template bswapg should preserve type");
    
  static constexpr auto swapped_value = __builtin_bswapg(Value);
};

template<auto... Values>
void test_template_pack_types() {
  static_assert((is_same_type<decltype(Values), decltype(__builtin_bswapg(Values))> && ...), "All pack elements should preserve type");
}

template struct ValueTemplateTypeTest<0x1234>;
template struct ValueTemplateTypeTest<0x12345678UL>;
template struct ValueTemplateTypeTest<(short)0x1234>;
template struct ValueTemplateTypeTest<(char)0x12>;