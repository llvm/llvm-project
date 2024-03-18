// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fexceptions -verify %s
// RUN: %clang_cc1 -std=c++2a -fcxx-exceptions -DUSE_CONSTEVAL -fexceptions -verify %s
// RUN: %clang_cc1 -std=c++2b -fcxx-exceptions -DUSE_CONSTEVAL -DPAREN_INIT -fexceptions -verify %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fms-extensions -DMS -fexceptions -verify %s
// RUN: %clang_cc1 -std=c++2a -fcxx-exceptions -fms-extensions -DMS -DUSE_CONSTEVAL -fexceptions -verify %s
//
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fexceptions -fexperimental-new-constant-interpreter -DNEW_INTERP -verify %s
// RUN: %clang_cc1 -std=c++2a -fcxx-exceptions -DUSE_CONSTEVAL -fexceptions -fexperimental-new-constant-interpreter -DNEW_INTERP -verify %s
// RUN: %clang_cc1 -std=c++2b -fcxx-exceptions -DUSE_CONSTEVAL -DPAREN_INIT -fexceptions -fexperimental-new-constant-interpreter -DNEW_INTERP -verify %s
// RUN: %clang_cc1 -std=c++1z -fcxx-exceptions -fms-extensions -DMS -fexceptions -fexperimental-new-constant-interpreter -DNEW_INTERP -verify %s
// RUN: %clang_cc1 -std=c++2a -fcxx-exceptions -fms-extensions -DMS -DUSE_CONSTEVAL -fexceptions -fexperimental-new-constant-interpreter -DNEW_INTERP -verify %s
// expected-no-diagnostics

#define assert(...) ((__VA_ARGS__) ? ((void)0) : throw 42)
#define CURRENT_FROM_MACRO() SL::current()
#define FORWARD(...) __VA_ARGS__

template <unsigned>
struct Printer;

#ifdef USE_CONSTEVAL
#define SOURCE_LOC_EVAL_KIND consteval
#else
#define SOURCE_LOC_EVAL_KIND constexpr
#endif

namespace std {
class source_location {
  struct __impl;

public:
  static SOURCE_LOC_EVAL_KIND source_location
    current(const __impl *__p = __builtin_source_location()) noexcept {
      source_location __loc;
      __loc.__m_impl = __p;
      return __loc;
  }
  constexpr source_location() = default;
  constexpr source_location(source_location const &) = default;
  constexpr unsigned int line() const noexcept { return __m_impl ? __m_impl->_M_line : 0; }
  constexpr unsigned int column() const noexcept { return __m_impl ? __m_impl->_M_column : 0; }
  constexpr const char *file() const noexcept { return __m_impl ? __m_impl->_M_file_name : ""; }
  constexpr const char *function() const noexcept { return __m_impl ? __m_impl->_M_function_name : ""; }

private:
  // Note: The type name "std::source_location::__impl", and its constituent
  // field-names are required by __builtin_source_location().
  struct __impl {
    const char *_M_file_name;
    const char *_M_function_name;
    unsigned _M_line;
    unsigned _M_column;
  };
  const __impl *__m_impl = nullptr;

public:
  using public_impl_alias = __impl;
};
} // namespace std

using SL = std::source_location;

#include "Inputs/source-location-file.h"
namespace SLF = source_location_file;

constexpr bool is_equal(const char *LHS, const char *RHS) {
  while (*LHS != 0 && *RHS != 0) {
    if (*LHS != *RHS)
      return false;
    ++LHS;
    ++RHS;
  }
  return *LHS == 0 && *RHS == 0;
}

template <class T>
constexpr T identity(T t) {
  return t;
}

template <class T, class U>
struct Pair {
  T first;
  U second;
};

template <class T, class U>
constexpr bool is_same = false;
template <class T>
constexpr bool is_same<T, T> = true;

// test types
static_assert(is_same<decltype(__builtin_LINE()), unsigned>);
static_assert(is_same<decltype(__builtin_COLUMN()), unsigned>);
static_assert(is_same<decltype(__builtin_FILE()), const char *>);
static_assert(is_same<decltype(__builtin_FILE_NAME()), const char *>);
static_assert(is_same<decltype(__builtin_FUNCTION()), const char *>);
#ifdef MS
static_assert(is_same<decltype(__builtin_FUNCSIG()), const char *>);
#endif
static_assert(is_same<decltype(__builtin_source_location()), const std::source_location::public_impl_alias *>);

// test noexcept
static_assert(noexcept(__builtin_LINE()));
static_assert(noexcept(__builtin_COLUMN()));
static_assert(noexcept(__builtin_FILE()));
static_assert(noexcept(__builtin_FILE_NAME()));
static_assert(noexcept(__builtin_FUNCTION()));
#ifdef MS
static_assert(noexcept(__builtin_FUNCSIG()));
#endif
static_assert(noexcept(__builtin_source_location()));

//===----------------------------------------------------------------------===//
//                            __builtin_LINE()
//===----------------------------------------------------------------------===//

namespace test_line {
static_assert(SL::current().line() == __LINE__);
static_assert(SL::current().line() == CURRENT_FROM_MACRO().line());

static constexpr SL GlobalS = SL::current();

static_assert(GlobalS.line() == __LINE__ - 2);

// clang-format off
constexpr bool test_line_fn() {
  constexpr SL S = SL::current();
  static_assert(S.line() == (__LINE__ - 1), "");
  // The start of the call expression to `current()` begins at the token `SL`
  constexpr int ExpectLine = __LINE__ + 3;
  constexpr SL S2
  =
  SL // Call expression starts here
  ::
  current
  (

  )
  ;
  static_assert(S2.line() == ExpectLine, "");

  static_assert(
          FORWARD(
             __builtin_LINE
            (
            )
          )
    == __LINE__ - 1, "");
  static_assert(\
\
  __builtin_LINE()\
\
  == __LINE__ - 2, "");
  static_assert(\
          _\
_builtin_LINE()
          == __LINE__ - 2, "");

  return true;
}
// clang-format on
static_assert(test_line_fn());

static_assert(__builtin_LINE() == __LINE__, "");

constexpr int baz() { return 101; }

constexpr int test_line_fn_simple(int z = baz(), int x = __builtin_LINE()) {
  return x;
}
void bar() {
  static_assert(test_line_fn_simple() == __LINE__, "");
  static_assert(test_line_fn_simple() == __LINE__, "");
}

struct CallExpr {
  constexpr int operator()(int x = __builtin_LINE()) const { return x; }
};
constexpr CallExpr get_call() { return CallExpr{}; }
static_assert(get_call()() == __LINE__, "");

template <class T>
constexpr bool test_line_fn_template(T Expect, int L = __builtin_LINE()) {
  return Expect == L;
}
static_assert(test_line_fn_template(__LINE__));

struct InMemInit {
  constexpr bool check(int expect) const {
    return info.line() == expect;
  }
  SL info = SL::current();
  InMemInit() = default;
  constexpr InMemInit(int) {}
};
static_assert(InMemInit{}.check(__LINE__ - 3), "");
static_assert(InMemInit{42}.check(__LINE__ - 3), "");

template <class T, class U = SL>
struct InMemInitTemplate {
  constexpr bool check(int expect) const {
    return info.line() == expect;
  }
  U info = U::current();
  InMemInitTemplate() = default;
  constexpr InMemInitTemplate(T) {}
  constexpr InMemInitTemplate(T, T) : info(U::current()) {}
  template <class V = U> constexpr InMemInitTemplate(T, T, T, V info = U::current())
      : info(info) {}
};
void test_mem_init_template() {
  constexpr int line_offset = 8;
  static_assert(InMemInitTemplate<int>{}.check(__LINE__ - line_offset), "");
  static_assert(InMemInitTemplate<unsigned>{42}.check(__LINE__ - line_offset), "");
  static_assert(InMemInitTemplate<unsigned>{42, 42}.check(__LINE__ - line_offset), "");
  static_assert(InMemInitTemplate<unsigned>{42, 42, 42}.check(__LINE__), "");
}

struct AggInit {
  int x;
  int y = __builtin_LINE();
  constexpr bool check(int expect) const {
    return y == expect;
  }
};
constexpr AggInit AI{42};
static_assert(AI.check(__LINE__ - 1), "");

template <class T, class U = SL>
struct AggInitTemplate {
  constexpr bool check(int expect) const {
    return expect == info.line();
  }
  T x;
  U info = U::current();
};

template <class T, class U = SL>
constexpr U test_fn_template(T, U u = U::current()) {
  return u;
}
void fn_template_tests() {
  static_assert(test_fn_template(42).line() == __LINE__, "");
}

struct TestMethodTemplate {
  template <class T, class U = SL, class U2 = SL>
  constexpr U get(T, U u = U::current(), U2 u2 = identity(U2::current())) const {
    assert(u.line() == u2.line());
    return u;
  }
};
void method_template_tests() {
  static_assert(TestMethodTemplate{}.get(42).line() == __LINE__, "");
}

struct InStaticInit {
  static constexpr int LINE = __LINE__;
  static constexpr const int x1 = __builtin_LINE();
  static constexpr const int x2 = identity(__builtin_LINE());
  static const int x3;
  const int x4 = __builtin_LINE();
  int x5 = __builtin_LINE();
};
const int InStaticInit::x3 = __builtin_LINE();
static_assert(InStaticInit::x1 == InStaticInit::LINE + 1, "");
static_assert(InStaticInit::x2 == InStaticInit::LINE + 2, "");

template <class T, int N = __builtin_LINE(), int Expect = -1>
constexpr void check_fn_template_param(T) {
  constexpr int RealExpect = Expect == -1 ? __LINE__ - 2 : Expect;
  static_assert(N == RealExpect);
}
template void check_fn_template_param(int);
template void check_fn_template_param<long, 42, 42>(long);

#line 100
struct AggBase {
#line 200
  int x = __builtin_LINE();
  int y = __builtin_LINE();
  int z = __builtin_LINE();
};
#line 300
struct AggDer : AggBase {
};
#line 400
static_assert(AggDer{}.x == 400, "");

struct ClassBase {
#line 400
  int x = __builtin_LINE();
  int y = 0;
  int z = 0;
#line 500
  ClassBase() = default;
  constexpr ClassBase(int yy, int zz = __builtin_LINE())
      : y(yy), z(zz) {}
};
struct ClassDer : ClassBase {
#line 600
  ClassDer() = default;
  constexpr ClassDer(int yy) : ClassBase(yy) {}
  constexpr ClassDer(int yy, int zz) : ClassBase(yy, zz) {}
};
#line 700
static_assert(ClassDer{}.x == 500, "");
static_assert(ClassDer{42}.x == 501, "");
static_assert(ClassDer{42}.z == 601, "");
static_assert(ClassDer{42, 42}.x == 501, "");

struct ClassAggDer : AggBase {
#line 800
  ClassAggDer() = default;
  constexpr ClassAggDer(int, int x = __builtin_LINE()) : AggBase{x} {}
};
static_assert(ClassAggDer{}.x == 100, "");

} // namespace test_line

//===----------------------------------------------------------------------===//
//                            __builtin_FILE()
//===----------------------------------------------------------------------===//

namespace test_file {
constexpr const char *test_file_simple(const char *__f = __builtin_FILE()) {
  return __f;
}
void test_function() {
#line 900
  static_assert(is_equal(test_file_simple(), __FILE__));
  static_assert(is_equal(SLF::test_function().file(), __FILE__), "");
  static_assert(is_equal(SLF::test_function_template(42).file(), __FILE__), "");

  static_assert(is_equal(SLF::test_function_indirect().file(), SLF::global_info.file()), "");
  static_assert(is_equal(SLF::test_function_template_indirect(42).file(), SLF::global_info.file()), "");

  static_assert(test_file_simple() != nullptr);
  static_assert(!is_equal(test_file_simple(), "source_location.cpp"));
}

void test_class() {
#line 315
  using SLF::TestClass;
  constexpr TestClass Default;
  constexpr TestClass InParam{42};
  constexpr TestClass Template{42, 42};
  constexpr auto *F = Default.info.file();
  constexpr auto Char = F[0];
  static_assert(is_equal(Default.info.file(), SLF::FILE), "");
  static_assert(is_equal(InParam.info.file(), SLF::FILE), "");
  static_assert(is_equal(InParam.ctor_info.file(), __FILE__), "");
}

void test_aggr_class() {
  using Agg = SLF::AggrClass<>;
  constexpr Agg Default{};
  constexpr Agg InitOne{42};
  static_assert(is_equal(Default.init_info.file(), __FILE__), "");
  static_assert(is_equal(InitOne.init_info.file(), __FILE__), "");
}

} // namespace test_file

//===----------------------------------------------------------------------===//
//                            __builtin_FILE_NAME()
//===----------------------------------------------------------------------===//

namespace test_file_name {
constexpr const char *test_file_name_simple(
  const char *__f = __builtin_FILE_NAME()) {
  return __f;
}
void test_function() {
#line 900
  static_assert(is_equal(test_file_name_simple(), __FILE_NAME__));
  static_assert(is_equal(SLF::test_function_filename(), __FILE_NAME__), "");
  static_assert(is_equal(SLF::test_function_filename_template(42),
                         __FILE_NAME__), "");

  static_assert(is_equal(SLF::test_function_filename_indirect(),
                         SLF::global_info_filename), "");
  static_assert(is_equal(SLF::test_function_filename_template_indirect(42),
                         SLF::global_info_filename), "");

  static_assert(test_file_name_simple() != nullptr);
  static_assert(is_equal(test_file_name_simple(), "source_location.cpp"));
}

void test_class() {
#line 315
  using SLF::TestClass;
  constexpr TestClass Default;
  constexpr TestClass InParam{42};
  constexpr TestClass Template{42, 42};
  constexpr auto *F = Default.info_file_name;
  constexpr auto Char = F[0];
  static_assert(is_equal(Default.info_file_name, SLF::FILE_NAME), "");
  static_assert(is_equal(InParam.info_file_name, SLF::FILE_NAME), "");
  static_assert(is_equal(InParam.ctor_info_file_name, __FILE_NAME__), "");
}

void test_aggr_class() {
  using Agg = SLF::AggrClass<>;
  constexpr Agg Default{};
  constexpr Agg InitOne{42};
  static_assert(is_equal(Default.init_info_file_name, __FILE_NAME__), "");
  static_assert(is_equal(InitOne.init_info_file_name, __FILE_NAME__), "");
}

} // namespace test_file_name

//===----------------------------------------------------------------------===//
//                            __builtin_FUNCTION()
//===----------------------------------------------------------------------===//

namespace test_func {

constexpr const char *test_func_simple(const char *__f = __builtin_FUNCTION()) {
  return __f;
}
constexpr const char *get_function() {
  return __func__;
}
constexpr bool test_function() {
  return is_equal(__func__, test_func_simple()) &&
         !is_equal(get_function(), test_func_simple());
}
static_assert(test_function());

template <class T, class U = SL>
constexpr Pair<U, U> test_func_template(T, U u = U::current()) {
  static_assert(is_equal(__PRETTY_FUNCTION__, U::current().function()));
  return {u, U::current()};
}
template <class T>
void func_template_tests() {
  constexpr auto P = test_func_template(42);
  //static_assert(is_equal(P.first.function(), __func__), "");
  //static_assert(!is_equal(P.second.function(), __func__), "");
}
template void func_template_tests<int>();

template <class = int, class T = SL>
struct TestCtor {
  T info = T::current();
  T ctor_info;
  TestCtor() = default;
  template <class U = SL>
  constexpr TestCtor(int, U u = U::current()) : ctor_info(u) {}
};
void ctor_tests() {
  constexpr TestCtor<> Default;
  constexpr TestCtor<> Template{42};
  static const char *XYZZY = Template.info.function();
  static_assert(is_equal(Default.info.function(), "test_func::TestCtor<>::TestCtor() [T = std::source_location]"));
  static_assert(is_equal(Default.ctor_info.function(), ""));
  static_assert(is_equal(Template.info.function(), "test_func::TestCtor<>::TestCtor(int, U) [T = std::source_location, U = std::source_location]"));
  static_assert(is_equal(Template.ctor_info.function(), __PRETTY_FUNCTION__));
}

constexpr SL global_sl = SL::current();
static_assert(is_equal(global_sl.function(), ""));

} // namespace test_func

//===----------------------------------------------------------------------===//
//                            __builtin_FUNCSIG()
//===----------------------------------------------------------------------===//

#ifdef MS
namespace test_funcsig {

constexpr const char *test_funcsig_simple(const char *f = __builtin_FUNCSIG()) {
  return f;
}
constexpr const char *get_funcsig() {
  return __FUNCSIG__;
}
constexpr bool test_funcsig() {
  return is_equal(__FUNCSIG__, test_funcsig_simple()) &&
         !is_equal(get_funcsig(), test_funcsig_simple());
}
static_assert(test_funcsig());

template <class T>
constexpr Pair<const char*, const char*> test_funcsig_template(T, const char* f = __builtin_FUNCSIG()) {
  return {f, __builtin_FUNCSIG()};
}
template <class T>
void func_template_tests() {
  constexpr auto P = test_funcsig_template(42);
  static_assert(is_equal(P.first, __FUNCSIG__), "");
  static_assert(!is_equal(P.second, __FUNCSIG__), "");
}
template void func_template_tests<int>();

template <class = int, class T = const char*>
struct TestCtor {
  T funcsig = __builtin_FUNCSIG();
  T ctor_funcsig;
  TestCtor() = default;
  template <class F = const char*>
  constexpr TestCtor(int, F f = __builtin_FUNCSIG()) : ctor_funcsig(f) {}
};
void ctor_tests() {
  constexpr TestCtor<> Template{42};
  static_assert(is_equal(Template.funcsig, "__cdecl test_funcsig::TestCtor<>::TestCtor(int, F) [T = const char *, F = const char *]"));
  static_assert(is_equal(Template.ctor_funcsig, __FUNCSIG__));
}

constexpr const char* global_funcsig = __builtin_FUNCSIG();
static_assert(is_equal(global_funcsig, ""));

} // namespace test_funcsig
#endif

//===----------------------------------------------------------------------===//
//                            __builtin_COLUMN()
//===----------------------------------------------------------------------===//

namespace test_column {

// clang-format off
constexpr bool test_column_fn() {
  constexpr SL S = SL::current();
  static_assert(S.line() == (__LINE__ - 1), "");
  constexpr int Indent = 4;
  {
    // The start of the call expression to `current()` begins at the token `SL`
    constexpr int ExpectCol = Indent + 3;
    constexpr SL S2
     =
      SL // Call expression starts here
        ::
          current
                 (

                  )
                   ;
    static_assert(S2.column() == ExpectCol, "");
  }
  {
    constexpr int ExpectCol = 2;
    constexpr int C =
 __builtin_COLUMN // Expect call expression to start here
      ();
    static_assert(C == ExpectCol);
  }
  return true;
}
#line 420
static_assert(test_column_fn());

// Test that the column matches the start of the call expression 'SL::current()'
static_assert(SL::current().column() == __builtin_strlen("static_assert(S"));
struct TestClass {
  int x = __builtin_COLUMN();
   TestClass() = default; /* indented to 3 spaces for testing */
  constexpr TestClass(int, int o = __builtin_COLUMN()) : x(o) {}
};
struct TestAggClass {
  int x = __builtin_COLUMN();
};
constexpr bool test_class() {

  auto check = [](int V, const char* S, int indent = 4) {
    assert(V == (__builtin_strlen(S) + indent));
  };
  {
    TestClass t{};
    check(t.x, "   T", 0); // Start of default constructor decl.
  }
  {
    TestClass t1
            {42};
    check(t1.x, "TestClass t"); // Start of variable being constructed.
  }
  {
    TestAggClass t  { };
    check(t.x, "TestAggClass t  { }");
  }
  {
    TestAggClass t = { };
    check(t.x, "TestAggClass t = { }");
  }
  return true;
}
static_assert(test_class());
// clang-format on
} // namespace test_column

// Test [reflection.src_loc.creation]p2
//  >  The value should be affected by #line (C++14 16.4) in the same manner as
//  >  for __LINE__ and __FILE__.
namespace test_pragma_line {
constexpr int StartLine = 42;
#line 42
static_assert(__builtin_LINE() == StartLine);
static_assert(__builtin_LINE() == StartLine + 1);
static_assert(SL::current().line() == StartLine + 2);
#line 44 "test_file.c"
static_assert(is_equal("test_file.c", __FILE__));
static_assert(is_equal("test_file.c", __builtin_FILE()));
static_assert(is_equal("test_file.c", __builtin_FILE_NAME()));
static_assert(is_equal("test_file.c", SL::current().file()));
static_assert(is_equal("test_file.c", SLF::test_function().file()));
static_assert(is_equal(SLF::FILE, SLF::test_function_indirect().file()));
} // end namespace test_pragma_line

namespace test_out_of_line_init {
#line 4000 "test_out_of_line_init.cpp"
constexpr unsigned get_line(unsigned n = __builtin_LINE()) { return n; }
constexpr const char *get_file(const char *f = __builtin_FILE()) { return f; }
constexpr const char *get_func(const char *f = __builtin_FUNCTION()) { return f; }
#line 4100 "A.cpp"
struct A {
  int n = __builtin_LINE();
  int n2 = get_line();
  const char *f = __builtin_FILE();
  const char *f2 = get_file();
  const char *func = __builtin_FUNCTION();
  const char *func2 = get_func();
  SL info = SL::current();
};
#line 4200 "B.cpp"
struct B {
  A a = {};
};
#line 4300 "test_passed.cpp"
constexpr B b = {};
static_assert(b.a.n == 4300, "");
static_assert(b.a.n2 == 4300, "");
static_assert(b.a.info.line() == 4300, "");
static_assert(is_equal(b.a.f, "test_passed.cpp"));
static_assert(is_equal(b.a.f2, "test_passed.cpp"));
static_assert(is_equal(b.a.info.file(), "test_passed.cpp"));
static_assert(is_equal(b.a.func, ""));
static_assert(is_equal(b.a.func2, ""));
static_assert(is_equal(b.a.info.function(), ""));

constexpr bool test_in_func() {
#line 4400 "test_func_passed.cpp"
  constexpr B b = {};
  static_assert(b.a.n == 4400, "");
  static_assert(b.a.n2 == 4400, "");
  static_assert(b.a.info.line() == 4400, "");
  static_assert(is_equal(b.a.f, "test_func_passed.cpp"));
  static_assert(is_equal(b.a.f2, "test_func_passed.cpp"));
  static_assert(is_equal(b.a.info.file(), "test_func_passed.cpp"));
  static_assert(is_equal(b.a.func, "test_in_func"));
  static_assert(is_equal(b.a.func2, "test_in_func"));
  static_assert(is_equal(b.a.info.function(), "bool test_out_of_line_init::test_in_func()"));
  return true;
}
static_assert(test_in_func());

} // end namespace test_out_of_line_init

namespace test_global_scope {
#line 5000 "test_global_scope.cpp"
constexpr unsigned get_line(unsigned n = __builtin_LINE()) { return n; }
constexpr const char *get_file(const char *f = __builtin_FILE()) { return f; }
constexpr const char *get_func(const char *f = __builtin_FUNCTION()) { return f; }
#line 5100
struct InInit {
  unsigned l = get_line();
  const char *f = get_file();
  const char *func = get_func();

#line 5200 "in_init.cpp"
  constexpr InInit() {}
};
#line 5300
constexpr InInit II;

static_assert(II.l == 5200, "");
static_assert(is_equal(II.f, "in_init.cpp"));
static_assert(is_equal(II.func, "InInit"));

#line 5400
struct AggInit {
  unsigned l = get_line();
  const char *f = get_file();
  const char *func = get_func();
};
#line 5500 "brace_init.cpp"
constexpr AggInit AI = {};
static_assert(AI.l == 5500);
static_assert(is_equal(AI.f, "brace_init.cpp"));
static_assert(is_equal(AI.func, ""));

} // namespace test_global_scope

namespace TestFuncInInit {
#line 6000 "InitClass.cpp"
struct Init {
  SL info;
#line 6100 "InitCtor.cpp"
  constexpr Init(SL info = SL::current()) : info(info) {}
};
#line 6200 "InitGlobal.cpp"
constexpr Init I;
static_assert(I.info.line() == 6200);
static_assert(is_equal(I.info.file(), "InitGlobal.cpp"));

} // namespace TestFuncInInit

namespace TestConstexprContext {
#line 7000 "TestConstexprContext.cpp"
  constexpr const char* foo() { return __builtin_FILE(); }
#line 7100 "Bar.cpp"
  constexpr const char* bar(const char* x = foo()) { return x; }
  constexpr bool test() {
    static_assert(is_equal(bar(), "TestConstexprContext.cpp"));
    return true;
  }
  static_assert(test());
}

namespace Lambda {
#line 8000 "TestLambda.cpp"
constexpr int nested_lambda(int l = []{
  return SL::current().line();
}()) {
  return l;
}
static_assert(nested_lambda() == __LINE__ - 4);

constexpr int lambda_param(int l = [](int l = SL::current().line()) {
  return l;
}()) {
  return l;
}
static_assert(lambda_param() == __LINE__);


}

constexpr int compound_literal_fun(int a =
                  (int){ SL::current().line() }
) { return a ;}
static_assert(compound_literal_fun() == __LINE__);

struct CompoundLiteral {
  int a = (int){ SL::current().line() };
};
static_assert(CompoundLiteral{}.a == __LINE__);


// FIXME
// Init captures are subexpressions of the lambda expression
// so according to the standard immediate invocations in init captures
// should be evaluated at the call site.
// However Clang does not yet implement this as it would introduce
// a fair bit of complexity.
// We intend to implement that functionality once we find real world
// use cases that require it.
constexpr int test_init_capture(int a =
                [b = SL::current().line()] { return b; }()) {
  return a;
}
#if defined(USE_CONSTEVAL) && !defined(NEW_INTERP)
static_assert(test_init_capture() == __LINE__ - 4);
#else
static_assert(test_init_capture() == __LINE__ );
#endif

namespace check_immediate_invocations_in_templates {

template <typename T = int>
struct G {
    T line = __builtin_LINE();
};
template <typename T>
struct S {
    int i = G<T>{}.line;
};
static_assert(S<int>{}.i != // intentional new line
              S<int>{}.i);

template <typename T>
constexpr int f(int i = G<T>{}.line) {
    return i;
}

static_assert(f<int>() != // intentional new line
              f<int>());
}

#ifdef PAREN_INIT
namespace GH63903 {
struct S {
    int _;
    int i = SL::current().line();
    int j = __builtin_LINE();
};
// Ensure parent aggregate initialization is consistent with brace
// aggregate initialization.
// Note: consteval functions are evaluated where they are used.
static_assert(S(0).i == __builtin_LINE());
static_assert(S(0).i == S{0}.i);
static_assert(S(0).j == S{0}.j);
static_assert(S(0).j == S{0}.i);
}
#endif

namespace GH78128 {

template<int N>
constexpr int f() {
  return N;
}

template<typename T>
void foo() {
  constexpr auto* F1 = std::source_location::current().function();
  static_assert(__builtin_strlen(F1) == f<__builtin_strlen(F1)>());

  constexpr auto* F2 = __builtin_FUNCTION();
  static_assert(__builtin_strlen(F2) == f<__builtin_strlen(F2)>());

#ifdef MS
  constexpr auto* F3 = __builtin_FUNCSIG();
  static_assert(__builtin_strlen(F3) == f<__builtin_strlen(F3)>());
#endif
}

void test() {
  foo<int>();
}

}

namespace GH80630 {

#define GH80630_LAMBDA \
    []( char const* fn ) { \
        static constexpr std::source_location loc = std::source_location::current(); \
        return &loc; \
    }( std::source_location::current().function() )

auto f( std::source_location const* loc = GH80630_LAMBDA ) {
    return loc;
}

auto g() {
    return f();
}

}
