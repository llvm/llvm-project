// Test this without pch.
// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -include %S/cxx-templates.h -verify %s -ast-dump -o -
// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -include %S/cxx-templates.h %s -emit-llvm -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -x c++-header -emit-pch -o %t %S/cxx-templates.h
// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -include-pch %t -verify %s -ast-dump  -o -
// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -include-pch %t %s -emit-llvm -o - | FileCheck %s

// expected-no-diagnostics

// CHECK: define weak_odr void @_ZN2S4IiE1mEv
// CHECK: define linkonce_odr void @_ZN2S3IiE1mEv

struct A {
  typedef int type;
  static void my_f();
  template <typename T>
  static T my_templf(T x) { return x; }
};

void test(const int (&a6)[17]) {
  int x = templ_f<int, 5>(3);
  
  S<char, float>::templ();
  S<int, char>::partial();
  S<int, float>::explicit_special();
  
  Dep<A>::Ty ty;
  Dep<A> a;
  a.f();
  
  S3<int> s3;
  s3.m();

  TS5 ts(0);

  S6<const int[17]>::t2 b6 = a6;
}

template struct S4<int>;

S7<int[5]> s7_5;

namespace ZeroLengthExplicitTemplateArgs {
  template void f<X>(X*);
}

// This used to overwrite memory and crash.
namespace Test1 {
  struct StringHasher {
    template<typename T, char Converter(T)> static inline unsigned createHash(const T*, unsigned) {
      return 0;
    }
  };

  struct CaseFoldingHash {
    static inline char foldCase(char) {
      return 0;
    }

    static unsigned hash(const char* data, unsigned length) {
      return StringHasher::createHash<char, foldCase>(data, length);
    }
  };
}

template< typename D >
Foo< D >& Foo< D >::operator=( const Foo& other )
{
   return *this;
}

namespace TestNestedExpansion {
  struct Int {
    Int(int);
    friend Int operator+(Int, Int);
  };
  Int &g(Int, int, double);
  Int &test = NestedExpansion<char, char, char>().f(0, 1, 2, Int(3), 4, 5.0);
}
