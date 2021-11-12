// RUN: %clang_cc1 -std=gnu++11 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -std=gnu++11 -ast-dump %s | FileCheck %s

namespace attribute_aligned {
  template<int N>
  struct X {
    char c[1] __attribute__((__aligned__((N)))); // expected-error {{alignment is not a power of 2}}
  };

  template <bool X> struct check {
    int check_failed[X ? 1 : -1]; // expected-error {{array with a negative size}}
  };

  template <int N> struct check_alignment {
    typedef check<N == sizeof(X<N>)> t; // expected-note {{in instantiation}}
  };

  check_alignment<1>::t c1;
  check_alignment<2>::t c2;
  check_alignment<3>::t c3; // expected-note 2 {{in instantiation}}
  check_alignment<4>::t c4;

  template<unsigned Size, unsigned Align>
  class my_aligned_storage
  {
    __attribute__((aligned(Align))) char storage[Size];
  };
  
  template<typename T>
  class C {
  public:
    C() {
      static_assert(sizeof(t) == sizeof(T), "my_aligned_storage size wrong");
      static_assert(alignof(t) == alignof(T), "my_aligned_storage align wrong"); // expected-warning{{'alignof' applied to an expression is a GNU extension}}
    }
    
  private:
    my_aligned_storage<sizeof(T), alignof(T)> t;
  };
  
  C<double> cd;
}

namespace PR9049 {
  extern const void *CFRetain(const void *ref);

  template<typename T> __attribute__((cf_returns_retained))
  inline T WBCFRetain(T aValue) { return aValue ? (T)CFRetain(aValue) : (T)0; }


  extern void CFRelease(const void *ref);

  template<typename T>
  inline void WBCFRelease(__attribute__((cf_consumed)) T aValue) { if(aValue) CFRelease(aValue); }
}

// CHECK: FunctionTemplateDecl {{.*}} HasAnnotations
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_FOO"
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_BAR"
// CHECK: FunctionDecl {{.*}} HasAnnotations
// CHECK:   TemplateArgument type 'int'
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_FOO"
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_BAR"
template<typename T> [[clang::annotate("ANNOTATE_FOO"), clang::annotate("ANNOTATE_BAR")]] void HasAnnotations();
void UseAnnotations() { HasAnnotations<int>(); }

namespace preferred_name {
  int x [[clang::preferred_name("frank")]]; // expected-error {{expected a type}}
  int y [[clang::preferred_name(int)]]; // expected-warning {{'preferred_name' attribute only applies to class templates}}
  struct [[clang::preferred_name(int)]] A; // expected-warning {{'preferred_name' attribute only applies to class templates}}
  template<typename T> struct [[clang::preferred_name(int)]] B; // expected-error {{argument 'int' to 'preferred_name' attribute is not a typedef for a specialization of 'B'}}
  template<typename T> struct C;
  using X = C<int>; // expected-note {{'X' declared here}}
  typedef C<float> Y;
  using Z = const C<double>; // expected-note {{'Z' declared here}}
  template<typename T> struct [[clang::preferred_name(C<int>)]] C; // expected-error {{argument 'C<int>' to 'preferred_name' attribute is not a typedef for a specialization of 'C'}}
  template<typename T> struct [[clang::preferred_name(X), clang::preferred_name(Y)]] C;
  template<typename T> struct [[clang::preferred_name(const X)]] C; // expected-error {{argument 'const preferred_name::X'}}
  template<typename T> struct [[clang::preferred_name(Z)]] C; // expected-error {{argument 'preferred_name::Z' (aka 'const C<double>')}}
  template<typename T> struct C {};

  // CHECK: ClassTemplateDecl {{.*}} <line:[[@LINE-10]]:{{.*}} C
  // CHECK:   ClassTemplateSpecializationDecl {{.*}} struct C definition
  // CHECK:     TemplateArgument type 'int'
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     PreferredNameAttr {{.*}} preferred_name::X
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     CXXRecordDecl
  // CHECK:   ClassTemplateSpecializationDecl {{.*}} struct C definition
  // CHECK:     TemplateArgument type 'float'
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     PreferredNameAttr {{.*}} preferred_name::Y
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     CXXRecordDecl
  // CHECK:   ClassTemplateSpecializationDecl {{.*}} struct C definition
  // CHECK:     TemplateArgument type 'double'
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     CXXRecordDecl

  // Check this doesn't cause us to instantiate the same attribute multiple times.
  C<float> *cf1;
  C<float> *cf2;

  void f(C<int> a, C<float> b, C<double> c) {
    auto p = a;
    auto q = b;
    auto r = c;
    p.f(); // expected-error {{no member named 'f' in 'preferred_name::X'}}
    q.f(); // expected-error {{no member named 'f' in 'preferred_name::Y'}}
    r.f(); // expected-error {{no member named 'f' in 'preferred_name::C<double>'}}
  }

  template<typename T> struct D;
  using DInt = D<int>;
  template<typename T> struct __attribute__((__preferred_name__(DInt))) D {};
  template struct D<int>;
  int use_dint = D<int>().get(); // expected-error {{no member named 'get' in 'preferred_name::DInt'}}

  template<typename T> struct MemberTemplate {
    template<typename U> struct Iter;
    using iterator = Iter<T>;
    using const_iterator = Iter<const T>;
    template<typename U>
    struct [[clang::preferred_name(iterator),
             clang::preferred_name(const_iterator)]] Iter {};
  };
  template<typename T> T desugar(T);
  auto it = desugar(MemberTemplate<int>::Iter<const int>());
  int n = it; // expected-error {{no viable conversion from 'preferred_name::MemberTemplate<int>::const_iterator' to 'int'}}

  template<int A, int B, typename ...T> struct Foo;
  template<typename ...T> using Bar = Foo<1, 2, T...>;
  template<int A, int B, typename ...T> struct [[clang::preferred_name(::preferred_name::Bar<T...>)]] Foo {};
  Foo<1, 2, int, float>::nosuch x; // expected-error {{no type named 'nosuch' in 'preferred_name::Bar<int, float>'}}
}
::preferred_name::Foo<1, 2, int, float>::nosuch x; // expected-error {{no type named 'nosuch' in 'preferred_name::Bar<int, float>'}}
