// RUN: %clang_cc1 -fsyntax-only -std=c++1y -DCXX1Y -Wc++98-compat-pedantic -verify %s -DCXX1Y2
// RUN: %clang_cc1 -fsyntax-only -std=c++1y -DCXX1Y -Wc++98-compat -Werror %s -DCXX1Y2
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -Werror %s
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Werror %s -DCXX98

// RUN: %clang_cc1 -fsyntax-only -std=c++1y -Wc++98-compat-pedantic -verify %s -Wno-pre-c++14-compat-pedantic -DCXX1Y2

// -Wc++98-compat-pedantic warns on C++11 features which we accept without a
// warning in C++98 mode.

#line 32767 // ok
#line 32768 // expected-warning {{#line number greater than 32767 is incompatible with C++98}}

#define VA_MACRO(x, ...) x // expected-warning {{variadic macros are incompatible with C++98}}
VA_MACRO(,x) // expected-warning {{empty macro arguments are incompatible with C++98}}

; // expected-warning {{extra ';' outside of a function is incompatible with C++98}}

enum Enum {
  Enum_value, // expected-warning {{commas at the end of enumerator lists are incompatible with C++98}}
};

void *dlsym();
void (*FnPtr)() = (void(*)())dlsym(); // expected-warning {{cast between pointer-to-function and pointer-to-object is incompatible with C++98}}
void *FnVoidPtr = (void*)&dlsym; // expected-warning {{cast between pointer-to-function and pointer-to-object is incompatible with C++98}}

struct ConvertToInt {
  operator int();
};
int *ArraySizeConversion = new int[ConvertToInt()];
#ifdef CXX1Y2
// expected-warning@-2 {{implicit conversion from array size expression of type 'ConvertToInt' to integral type 'size_t' is incompatible with C++98}}
#else
// expected-warning@-4 {{implicit conversion from array size expression of type 'ConvertToInt' to integral type 'int' is incompatible with C++98}}
#endif

template<typename T> class ExternTemplate {};
extern template class ExternTemplate<int>; // expected-warning {{extern templates are incompatible with C++98}}

long long ll1 = // expected-warning {{'long long' is incompatible with C++98}}
         -42LL; // expected-warning {{'long long' is incompatible with C++98}}
unsigned long long ull1 = // expected-warning {{'long long' is incompatible with C++98}}
                   42ULL; // expected-warning {{'long long' is incompatible with C++98}}

int k = 0b1001;
#ifdef CXX1Y
// expected-warning@-2 {{binary integer literals are incompatible with C++ standards before C++14}}
#endif

namespace CopyCtorIssues {
  struct Private {
    Private();
  private:
    Private(const Private&); // expected-note {{declared private here}}
  };
  struct NoViable {
    NoViable(); // expected-note {{not viable}}
    NoViable(NoViable&); // expected-note {{not viable}}
  };
  struct Ambiguous {
    Ambiguous();
    Ambiguous(const Ambiguous &, int = 0); // expected-note {{candidate}}
    Ambiguous(const Ambiguous &, double = 0); // expected-note {{candidate}}
  };
  struct Deleted {
    Private p; // expected-note {{implicitly deleted}}
  };

  const Private &a = Private(); // expected-warning {{copying variable of type 'CopyCtorIssues::Private' when binding a reference to a temporary would invoke an inaccessible constructor in C++98}}
  const NoViable &b = NoViable(); // expected-warning {{copying variable of type 'CopyCtorIssues::NoViable' when binding a reference to a temporary would find no viable constructor in C++98}}
#if !CXX98
  const Ambiguous &c = Ambiguous(); // expected-warning {{copying variable of type 'CopyCtorIssues::Ambiguous' when binding a reference to a temporary would find ambiguous constructors in C++98}}
#endif
  const Deleted &d = Deleted(); // expected-warning {{copying variable of type 'CopyCtorIssues::Deleted' when binding a reference to a temporary would invoke a deleted constructor in C++98}}
}
