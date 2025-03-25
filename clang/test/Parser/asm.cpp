// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

int foo1 asm ("bar1");
int foo2 asm (L"bar2"); // expected-error {{cannot use wide string literal in 'asm'}}
int foo3 asm (u8"bar3"); // expected-error {{cannot use unicode string literal in 'asm'}}
int foo4 asm (u"bar4"); // expected-error {{cannot use unicode string literal in 'asm'}}
int foo5 asm (U"bar5"); // expected-error {{cannot use unicode string literal in 'asm'}}
int foo6 asm ("bar6"_x); // expected-error {{string literal with user-defined suffix cannot be used here}}
int foo6 asm ("" L"bar7"); // expected-error {{cannot use wide string literal in 'asm'}}

void f() {
  [[]] asm("");
  [[gnu::deprecated]] asm(""); // expected-warning {{'deprecated' attribute ignored}}
}


#if !__has_extension(gnu_asm_constexpr_strings)
#error Extension 'gnu_asm_constexpr_strings' should be available by default
#endif

struct string_view {
  int S;
  const char* D;
  constexpr string_view(const char* Str) : S(__builtin_strlen(Str)), D(Str) {}
  constexpr string_view(int Size, const char* Str) : S(Size), D(Str) {}
  constexpr int size() const {
      return S;
  }
  constexpr const char* data() const {
      return D;
  }
};

// Neither gcc nor clang support expressions in label
int foo1 asm ((string_view("test"))); // expected-error {{expected string literal in 'asm'}}
int func() asm ((string_view("test"))); // expected-error {{expected string literal in 'asm'}}


void f2() {
  asm(string_view("")); // expected-error {{expected string literal or parenthesized constant expression in 'asm'}}
  asm("" : string_view("")); // expected-error {{expected string literal or parenthesized constant expression in 'asm'}}
  asm("" : : string_view("")); // expected-error {{expected string literal or parenthesized constant expression in 'asm'}}
  asm("" : : : string_view("")); // expected-error {{expected ')'}}
  asm("" :: string_view("")); // expected-error {{expected string literal or parenthesized constant expression in 'asm'}}
  asm(::string_view("")); // expected-error {{expected string literal or parenthesized constant expression in 'asm'}}

  int i;

  asm((string_view("")));
  asm((::string_view("")));
  asm("" : (::string_view("+g")) (i));
  asm("" : (::string_view("+g"))); // expected-error {{expected '(' after 'asm operand'}}
  asm("" : (::string_view("+g")) (i) : (::string_view("g")) (0));
  asm("" : (::string_view("+g")) (i) : (::string_view("g"))); // expected-error {{expected '(' after 'asm operand'}}
  asm("" : (::string_view("+g")) (i) : (::string_view("g")) (0) : (string_view("memory")));


  asm((0)); // expected-error {{the expression in this asm operand must be a string literal or an object with 'data()' and 'size()' member functions}}
}
