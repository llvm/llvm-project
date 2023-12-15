// RUN: %clang_cc1 -std=c++20  -Wno-all -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fblocks -include %s -verify %s

// RUN: %clang -x c++ -frtti -fsyntax-only -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -std=c++11 -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -std=c++20 -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// CHECK-NOT: [-Wunsafe-buffer-usage]

#ifndef INCLUDED
#define INCLUDED
#pragma clang system_header

// no spanification warnings for system headers
void foo(...);  // let arguments of `foo` to hold testing expressions
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

class span {

 int array[10];
 
 public:

 int *data() {
   return array;
 }
};

class A {
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

void warned_patterns(std::span<int> span_ptr, std::span<Base> base_span) {
    int *p;
    A *a = (A*)span_ptr.data(); // expected-warning{{unsafe invocation of span::data}}
    a = (A*)span_ptr.data(); // expected-warning{{unsafe invocation of span::data}}
   
    // TODO:: Should we warn when we cast from base to derived type?
    Derived *b = dynamic_cast<Derived*> (base_span.data());// expected-warning{{unsafe invocation of span::data}}

   // TODO:: This pattern is safe. We can add special handling for it, if we decide this
   // is the recommended fixit for the unsafe invocations.
   a = (A*)span_ptr.subspan(0, sizeof(A)).data(); // expected-warning{{unsafe invocation of span::data}}
}

void not_warned_patterns(std::span<A> span_ptr, std::span<Base> base_span) {
    int *p = (int*)span_ptr.data();
    p = (int*)span_ptr.data();
    A *a = (A*) span_ptr.hello();
}

// We do not want to warn about other types
void other_classes(std::span_duplicate<int> span_ptr, span sp) {
    int *p;
    A *a = (A*)span_ptr.data();
    a = (A*)span_ptr.data(); 

    a = (A*)sp.data();
}

// Potential source for false negatives

void false_negatives(std::span<int> span_pt) {
  int *ptr = span_pt.data();
  A *a = (A*)ptr; //TODO: We want to warn here eventually.

  int k = *ptr; // TODO: Can cause OOB if span_pt is empty

}
#endif
