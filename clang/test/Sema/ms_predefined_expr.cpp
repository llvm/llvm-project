// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions
// RUN: %clang_cc1 %s -fsyntax-only -Wmicrosoft -verify -fms-extensions -fexperimental-new-constant-interpreter

using size_t = __SIZE_TYPE__;

// Test array initialization
void array_init() {
 const char a[] = __FUNCTION__; // expected-warning{{initializing an array from a '__FUNCTION__' predefined identifier is a Microsoft extension}}
 const char b[] = __FUNCDNAME__; // expected-warning{{initializing an array from a '__FUNCDNAME__' predefined identifier is a Microsoft extension}}
 const char c[] = __FUNCSIG__; // expected-warning{{initializing an array from a '__FUNCSIG__' predefined identifier is a Microsoft extension}}
 const char d[] = __func__; // expected-warning{{initializing an array from a '__func__' predefined identifier is a Microsoft extension}}
 const char e[] = __PRETTY_FUNCTION__; // expected-warning{{initializing an array from a '__PRETTY_FUNCTION__' predefined identifier is a Microsoft extension}}
 const wchar_t f[] = L__FUNCTION__; // expected-warning{{initializing an array from a 'L__FUNCTION__' predefined identifier is a Microsoft extension}}
 const wchar_t g[] = L__FUNCSIG__; // expected-warning{{initializing an array from a 'L__FUNCSIG__' predefined identifier is a Microsoft extension}}
}

// Test function local identifiers outside of a function
const char* g_function = __FUNCTION__;            // expected-warning{{predefined identifier is only valid inside function}}
const char* g_function_lconcat = "" __FUNCTION__; // expected-warning{{predefined identifier is only valid inside function}} \
                                                  // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
const char* g_function_rconcat = __FUNCTION__ ""; // expected-warning{{predefined identifier is only valid inside function}} \
                                                  // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}

namespace NS
{
  const char* function = __FUNCTION__;            // expected-warning{{predefined identifier is only valid inside function}}
  const char* function_lconcat = "" __FUNCTION__; // expected-warning{{predefined identifier is only valid inside function}} \
                                                  // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  const char* function_rconcat = __FUNCTION__ ""; // expected-warning{{predefined identifier is only valid inside function}} \
                                                  // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}

  struct S
  {
    static constexpr const char* function = __FUNCTION__;            // expected-warning{{predefined identifier is only valid inside function}}
    static constexpr const char* function_lconcat = "" __FUNCTION__; // expected-warning{{predefined identifier is only valid inside function}} \
                                                                     // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
    static constexpr const char* function_rconcat = __FUNCTION__ ""; // expected-warning{{predefined identifier is only valid inside function}} \
                                                                     // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  };
}

template<class T, class U>
constexpr bool is_same = false;
template<class T>
constexpr bool is_same<T, T> = true;

template<typename T, size_t N>
constexpr bool equal(const T (&a)[N], const T (&b)[N]) {
  for (size_t i = 0; i < N; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

#define ASSERT_EQ(X, Y) static_assert(equal(X, Y), "")
#define ASSERT_EQ_TY(X, Y) static_assert(is_same<decltype((X)[0]), decltype((Y)[0])>, "")

#define _WIDE(s) L##s
#define WIDE(s)  _WIDE(s)

// Test value
extern "C" void test_value() {
  ASSERT_EQ(__FUNCTION__, "test_value");
  ASSERT_EQ(__FUNCSIG__, "void __cdecl test_value(void)");

  ASSERT_EQ(WIDE(__FUNCTION__), L"test_value");
  ASSERT_EQ(WIDE(__FUNCSIG__), L"void __cdecl test_value(void)");
}

namespace PR13206 {
  template<class T> class A {
    public:
      void method() {
        ASSERT_EQ(L__FUNCTION__, L"method");
      }
  };

  void test_template_value() {
    A<int> x;
    x.method();
  }
}

// Test concatenation
extern "C" void test_concat() {
  ASSERT_EQ("left_" __FUNCTION__, "left_test_concat");                   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ("left_" __FUNCDNAME__, "left_test_concat");                  // expected-warning{{expansion of predefined identifier '__FUNCDNAME__' to a string literal is a Microsoft extension}}
  ASSERT_EQ("left " __FUNCSIG__, "left void __cdecl test_concat(void)"); // expected-warning{{expansion of predefined identifier '__FUNCSIG__' to a string literal is a Microsoft extension}}

  ASSERT_EQ(__FUNCTION__ "_right", "test_concat_right");                   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(__FUNCDNAME__ "_right", "test_concat_right");                  // expected-warning{{expansion of predefined identifier '__FUNCDNAME__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(__FUNCSIG__ " right", "void __cdecl test_concat(void) right"); // expected-warning{{expansion of predefined identifier '__FUNCSIG__' to a string literal is a Microsoft extension}}

  ASSERT_EQ("left_" __FUNCTION__ "_right", "left_test_concat_right");                   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ("left_" __FUNCDNAME__ "_right", "left_test_concat_right");                  // expected-warning{{expansion of predefined identifier '__FUNCDNAME__' to a string literal is a Microsoft extension}}
  ASSERT_EQ("left " __FUNCSIG__ " right", "left void __cdecl test_concat(void) right"); // expected-warning{{expansion of predefined identifier '__FUNCSIG__' to a string literal is a Microsoft extension}}

  ASSERT_EQ(__FUNCTION__ "/" __FUNCSIG__, "test_concat/void __cdecl test_concat(void)"); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}} \
                                                                                         // expected-warning{{expansion of predefined identifier '__FUNCSIG__' to a string literal is a Microsoft extension}}
}

extern "C" void test_wide_concat() {
  // test L"" + ""
  ASSERT_EQ(L"" __FUNCTION__, L__FUNCTION__); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(L"" __FUNCSIG__, L__FUNCSIG__);   // expected-warning{{expansion of predefined identifier '__FUNCSIG__' to a string literal is a Microsoft extension}}

  // test Lx + ""
  ASSERT_EQ(L__FUNCTION__, L__FUNCTION__ ""); // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(L__FUNCSIG__, L__FUNCSIG__ "");   // expected-warning{{expansion of predefined identifier 'L__FUNCSIG__' to a string literal is a Microsoft extension}}

  ASSERT_EQ(L"left_" L__FUNCTION__, L"left_test_wide_concat");                   // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(L"left " L__FUNCSIG__, L"left void __cdecl test_wide_concat(void)"); // expected-warning{{expansion of predefined identifier 'L__FUNCSIG__' to a string literal is a Microsoft extension}}

  ASSERT_EQ(L__FUNCTION__ L"_right", L"test_wide_concat_right");                   // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(L__FUNCSIG__ L" right", L"void __cdecl test_wide_concat(void) right"); // expected-warning{{expansion of predefined identifier 'L__FUNCSIG__' to a string literal is a Microsoft extension}}

  ASSERT_EQ(L"left_" L__FUNCTION__ L"_right", L"left_test_wide_concat_right");                   // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(L"left " L__FUNCSIG__ L" right", L"left void __cdecl test_wide_concat(void) right"); // expected-warning{{expansion of predefined identifier 'L__FUNCSIG__' to a string literal is a Microsoft extension}}

  ASSERT_EQ(L__FUNCTION__ L"/" L__FUNCSIG__, L"test_wide_concat/void __cdecl test_wide_concat(void)"); // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}} \
                                                                                                       // expected-warning{{expansion of predefined identifier 'L__FUNCSIG__' to a string literal is a Microsoft extension}}
}

void test_encoding() {
  ASSERT_EQ_TY("" __FUNCTION__, "");     // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY(L"" __FUNCTION__, L"");   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY(L"" L__FUNCTION__, L"");  // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY("" L__FUNCTION__, L"");   // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY(u8"" __FUNCTION__, u8""); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY(u"" __FUNCTION__, u"");   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY(U"" __FUNCTION__, U"");   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
                                         //
  ASSERT_EQ_TY(__FUNCTION__ L"", L"");   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY(__FUNCTION__ u8"", u8""); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY(__FUNCTION__ u"", u"");   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ_TY(__FUNCTION__ U"", U"");   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
}

extern "C" void test_𐀀() {
  ASSERT_EQ(U"" __FUNCTION__, U"test_𐀀");   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(u"" __FUNCTION__, u"test_𐀀");   // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ(u8"" __FUNCTION__, u8"test_𐀀"); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
}

template<typename T>
void unused(T);

void test_invalid_encoding() {
  unused(u8"" L__FUNCTION__); // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}} \
                              // expected-error{{unsupported non-standard concatenation of string literals}}
  unused(u"" L__FUNCTION__);  // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}} \
                              // expected-error{{unsupported non-standard concatenation of string literals}}
  unused(U"" L__FUNCTION__);  // expected-warning{{expansion of predefined identifier 'L__FUNCTION__' to a string literal is a Microsoft extension}} \
                              // expected-error{{unsupported non-standard concatenation of string literals}}
}

constexpr size_t operator""_len(const char*, size_t len) {
  return len;
}

void test_udliteral() {
  static_assert(__FUNCTION__ ""_len == 14, ""); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
}

void test_static_assert() {
  static_assert(true, __FUNCTION__); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
}

void test_char_injection(decltype(sizeof('"')), decltype(sizeof("()"))) {
  unused("" __FUNCSIG__); // expected-warning{{expansion of predefined identifier '__FUNCSIG__' to a string literal is a Microsoft extension}}
}

void test_in_struct_init() {
  struct {
    char F[sizeof(__FUNCTION__)];
  } s1 = { __FUNCTION__ }; // expected-warning{{initializing an array from a '__FUNCTION__' predefined identifier is a Microsoft extension}}

  struct {
    char F[sizeof("F:" __FUNCTION__)]; // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  } s2 = { "F:" __FUNCTION__ }; // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}

  class C {
    public:
      struct {
        char F[sizeof(__FUNCTION__)];
      } s;
  } c1 = { { __FUNCTION__ } }; // expected-warning{{initializing an array from a '__FUNCTION__' predefined identifier is a Microsoft extension}}
}

void test_in_constexpr_struct_init() {
  struct {
    char F[sizeof(__FUNCTION__)];
  } constexpr s1 = { __FUNCTION__ }; // expected-warning{{initializing an array from a '__FUNCTION__' predefined identifier is a Microsoft extension}}
  ASSERT_EQ(__FUNCTION__, s1.F);

  struct {
    char F[sizeof("F:" __FUNCTION__)]; // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  } constexpr s2 = { "F:" __FUNCTION__ }; // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ("F:" __FUNCTION__, s2.F); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}

  class C {
    public:
      struct {
        char F[sizeof("F:" __FUNCTION__)]; // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
      } s;
  } constexpr c1 = { { "F:" __FUNCTION__ } }; // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
  ASSERT_EQ("F:" __FUNCTION__, c1.s.F); // expected-warning{{expansion of predefined identifier '__FUNCTION__' to a string literal is a Microsoft extension}}
}
