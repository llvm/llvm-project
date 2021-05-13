// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core,debug.ExprInspection -verify %s
//
// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core,debug.ExprInspection -verify \
// RUN:   -triple i386-pc-win32 -fms-compatibility -fms-extensions -DANALYZER_MS %s

template <typename T>
void clang_analyzer_dump(const T *);
void clang_analyzer_warnIfReached();

template <typename T, auto Value, typename U>
void func(U param) {
  clang_analyzer_dump(__func__);
  clang_analyzer_dump(__FUNCTION__);
  clang_analyzer_dump(__PRETTY_FUNCTION__);
  // expected-warning@-3 {{&Element{"func",0 S64b,char}}}
  // expected-warning@-3 {{&Element{"func",0 S64b,char}}}
  // expected-warning@-3 {{&Element{"void func(U) [T = Class, Value = 42ULL, U = char]",0 S64b,char}}}

#ifdef ANALYZER_MS
  clang_analyzer_dump(__FUNCDNAME__);
  clang_analyzer_dump(L__FUNCTION__);
  clang_analyzer_dump(__FUNCSIG__);
  clang_analyzer_dump(L__FUNCSIG__);
  // expected-warning@-4 {{&Element{"??$func@UClass@?1??foo@@YAXXZ@$0CK@D@@YAXD@Z",0 S64b,char}}}
  // expected-warning@-4 {{&Element{L"func",0 S64b,wchar_t}}}
  // expected-warning@-4 {{&Element{"void __cdecl func(U) [T = Class, Value = 42ULL, U = char]",0 S64b,char}}}
  // expected-warning@-4 {{&Element{L"void __cdecl func(U) [T = Class, Value = 42ULL, U = char]",0 S64b,wchar_t}}}
#endif
}

void foo() {
  clang_analyzer_dump(__func__);
  clang_analyzer_dump(__FUNCTION__);
  clang_analyzer_dump(__PRETTY_FUNCTION__);
  // expected-warning@-3 {{&Element{"foo",0 S64b,char}}}
  // expected-warning@-3 {{&Element{"foo",0 S64b,char}}}
  // expected-warning@-3 {{&Element{"void foo()",0 S64b,char}}}

#ifdef ANALYZER_MS
  clang_analyzer_dump(__FUNCDNAME__);
  clang_analyzer_dump(L__FUNCTION__);
  clang_analyzer_dump(__FUNCSIG__);
  clang_analyzer_dump(L__FUNCSIG__);
  // expected-warning@-4 {{&Element{"?foo@@YAXXZ",0 S64b,char}}}
  // expected-warning@-4 {{&Element{L"foo",0 S64b,wchar_t}}}
  // expected-warning@-4 {{&Element{"void __cdecl foo(void)",0 S64b,char}}}
  // expected-warning@-4 {{&Element{L"void __cdecl foo(void)",0 S64b,wchar_t}}}
#endif

  func<struct Class, 42ull>('b'); // instantiate template
}

struct A {
  A() {
    clang_analyzer_dump(__func__);
    clang_analyzer_dump(__FUNCTION__);
    clang_analyzer_dump(__PRETTY_FUNCTION__);
    // expected-warning@-3 {{&Element{"A",0 S64b,char}}}
    // expected-warning@-3 {{&Element{"A",0 S64b,char}}}
    // expected-warning@-3 {{&Element{"A::A()",0 S64b,char}}}

#ifdef ANALYZER_MS
    clang_analyzer_dump(__FUNCDNAME__);
    clang_analyzer_dump(L__FUNCTION__);
    clang_analyzer_dump(__FUNCSIG__);
    clang_analyzer_dump(L__FUNCSIG__);
    // expected-warning@-4 {{&Element{"??0A@@QAE@XZ",0 S64b,char}}}
    // expected-warning@-4 {{&Element{L"A",0 S64b,wchar_t}}}
    // expected-warning@-4 {{&Element{"__thiscall A::A(void)",0 S64b,char}}}
    // expected-warning@-4 {{&Element{L"__thiscall A::A(void)",0 S64b,wchar_t}}}
#endif
  }
  ~A() {
    clang_analyzer_dump(__func__);
    clang_analyzer_dump(__FUNCTION__);
    clang_analyzer_dump(__PRETTY_FUNCTION__);
    // expected-warning@-3 {{&Element{"~A",0 S64b,char}}}
    // expected-warning@-3 {{&Element{"~A",0 S64b,char}}}
    // expected-warning@-3 {{&Element{"A::~A()",0 S64b,char}}}

#ifdef ANALYZER_MS
    clang_analyzer_dump(__FUNCDNAME__);
    clang_analyzer_dump(L__FUNCTION__);
    clang_analyzer_dump(__FUNCSIG__);
    clang_analyzer_dump(L__FUNCSIG__);
    // expected-warning@-4 {{&Element{"??1A@@QAE@XZ",0 S64b,char}}}
    // expected-warning@-4 {{&Element{L"~A",0 S64b,wchar_t}}}
    // expected-warning@-4 {{&Element{"__thiscall A::~A(void)",0 S64b,char}}}
    // expected-warning@-4 {{&Element{L"__thiscall A::~A(void)",0 S64b,wchar_t}}}
#endif
  }

  template <typename> int dependent() {
    // We should not analyze dependent functions.
    // Such functions have no function name of predefined expressions such as: '__func__' etc.
    clang_analyzer_warnIfReached(); // no-warning
  }
};
