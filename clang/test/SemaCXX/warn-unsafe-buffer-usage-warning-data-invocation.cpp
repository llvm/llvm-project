// RUN: %clang_cc1 -std=c++20  -Wno-all -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fblocks -include %s -verify %s

// RUN: %clang -x c++ -frtti -fsyntax-only -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -std=c++11 -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -std=c++20 -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// CHECK-NOT: [-Wunsafe-buffer-usage]

#include <stdint.h>
#ifndef INCLUDED
#define INCLUDED
#pragma clang system_header

// no spanification warnings for system headers
#else

namespace std {
  class type_info;
  class bad_cast;
  class bad_typeid;
}
using size_t = __typeof(sizeof(int));
void *malloc(size_t);

void foo(int v) {
}

void foo(int *p){}

namespace std{
  template <typename T> class span {

  T *elements;
 
  span(T *, unsigned){}

  public:

  constexpr span<T> subspan(size_t offset, size_t count) const {
    return span<T> (elements+offset, count); // expected-warning{{unsafe pointer arithmetic}}
  }

  constexpr T* data() const noexcept {
    return elements;
  }

 
  constexpr T* hello() const noexcept {
   return elements;
  }
};
 
 template <typename T> class span_duplicate {
  span_duplicate(T *, unsigned){}

  T array[10];

  public:

  T* data() {
    return array;
  }

};
}

using namespace std;

class A {
  int a, b, c;
};

class B {
  int a, b, c;
};

struct Base {
   virtual ~Base() = default;
};

struct Derived: Base {
  int d;
};

void cast_without_data(int *ptr) {
 A *a = (A*) ptr;
 float *p = (float*) ptr;
}

void warned_patterns(std::span<int> span_ptr, std::span<Base> base_span, span<int> span_without_qual) {
    A *a1 = (A*)span_ptr.data(); // expected-warning{{unsafe invocation of span::data}}
    a1 = (A*)span_ptr.data(); // expected-warning{{unsafe invocation of span::data}}

    a1 = (A*)(span_ptr.data()); // expected-warning{{unsafe invocation of span::data}}
    A *a2 = (A*) (span_without_qual.data()); // expected-warning{{unsafe invocation of span::data}}

    a2 = (A*) span_without_qual.data(); // expected-warning{{unsafe invocation of span::data}}

     // TODO:: Should we warn when we cast from base to derived type?
     Derived *b = dynamic_cast<Derived*> (base_span.data());// expected-warning{{unsafe invocation of span::data}}

    // TODO:: This pattern is safe. We can add special handling for it, if we decide this
    // is the recommended fixit for the unsafe invocations.
    A *a3 = (A*)span_ptr.subspan(0, sizeof(A)).data(); // expected-warning{{unsafe invocation of span::data}}
}

void not_warned_patterns(std::span<A> span_ptr, std::span<Base> base_span) {
    int *p = (int*) span_ptr.data(); // Cast to a smaller type
  
    B *b = (B*) span_ptr.data(); // Cast to a type of same size.

    p = (int*) span_ptr.data();
    A *a = (A*) span_ptr.hello(); // Invoking other methods.
   
     intptr_t k = (intptr_t) span_ptr.data();
    k = (intptr_t) (span_ptr.data());
}

// We do not want to warn about other types
void other_classes(std::span_duplicate<int> span_ptr) {
    int *p;
    A *a = (A*)span_ptr.data();
    a = (A*)span_ptr.data(); 
}

// Potential source for false negatives

A false_negatives(std::span<int> span_pt, span<A> span_A) {
  int *ptr = span_pt.data();

  A *a1 = (A*)ptr; //TODO: We want to warn here eventually.

  A *a2= span_A.data();
  return *a2; // TODO: Can cause OOB if span_pt is empty

}
#endif
