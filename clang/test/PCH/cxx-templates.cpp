// Test this without pch.
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -include %S/cxx-templates.h -verify %s -ast-dump -o -
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -include %S/cxx-templates.h %s -emit-llvm -o - -DNO_ERRORS | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -x c++-header -emit-pch -o %t %S/cxx-templates.h
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -include-pch %t -verify %s -ast-dump  -o -
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -include-pch %t %s -emit-llvm -o - -error-on-deserialized-decl doNotDeserialize -DNO_ERRORS | FileCheck %s

// Test with modules.
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -fmodules -x c++-header -emit-pch -o %t %S/cxx-templates.h
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -fmodules -include-pch %t -verify %s -ast-dump  -o -
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fexceptions -fmodules -include-pch %t %s -emit-llvm -o - -error-on-deserialized-decl doNotDeserialize -DNO_ERRORS -fmodules-ignore-macro=NO_ERRORS | FileCheck %s

// Test with pch and delayed template parsing.
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fdelayed-template-parsing -fexceptions -x c++-header -emit-pch -o %t %S/cxx-templates.h
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fdelayed-template-parsing -fexceptions -include-pch %t -verify %s -ast-dump  -o -
// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fcxx-exceptions -fdelayed-template-parsing -fexceptions -include-pch %t %s -emit-llvm -o - -DNO_ERRORS | FileCheck %s

// CHECK: define weak_odr {{.*}}void @_ZN2S4IiE1mEv
// CHECK: define linkonce_odr {{.*}}void @_ZN2S3IiE1mEv

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

namespace rdar13135282 {
  void test() {
    __mt_alloc<> mt = __mt_alloc<>();
  }
}

void CallDependentSpecializedFunc(DependentSpecializedFuncClass<int> &x) {
  DependentSpecializedFunc(x);
}

namespace cyclic_module_load {
  extern std::valarray<int> x;
  std::valarray<int> y(x);
}

#ifndef NO_ERRORS
// expected-error@cxx-templates.h:305 {{incomplete}}
template int local_extern::f<int[]>(); // expected-note {{in instantiation of}}
#endif
template int local_extern::g<int[]>();

namespace MemberSpecializationLocation {
#ifndef NO_ERRORS
  // expected-note@cxx-templates.h:* {{previous}}
  template<> float A<int>::n; // expected-error {{redeclaration of 'n' with a different type}}
#endif
  int k = A<int>::n;
}

// https://bugs.llvm.org/show_bug.cgi?id=34728
namespace PR34728 {
int test() {
  // Verify with several TemplateParmDecl kinds, using PCH (incl. modules).
  int z1 = func1(/*ignored*/2.718);
  int z2 = func2(/*ignored*/3.142);
  int tmp3 = 30;
  Container<int> c = func3(tmp3);
  int z3 = c.item;

  // Return value is meaningless.  Just "use" all these values to avoid
  // warning about unused vars / values.
  return z1 + z2 + z3;
}
} // end namespace PR34728
