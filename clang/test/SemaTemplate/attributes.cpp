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
