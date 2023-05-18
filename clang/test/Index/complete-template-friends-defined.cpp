// The run lines are below, because this test is line- and
// column-number sensitive.

namespace N {
  template<typename T> struct A {
    template<typename U> friend class B;
  };

  template<typename T> struct B { };
}

void foo() {
  N::A<int> a1;
  N::A<int> a2;
}

namespace M {
  template<typename T> struct C {
    template<typename U> friend struct C;
  };
}

void bar() {
  M::C<int> c1;
  M::C<int> c2;
}

// RUN: c-index-test -code-completion-at=%s:14:6 %s | FileCheck -check-prefix=CHECK-ACCESS-1 %s
// CHECK-ACCESS-1: ClassTemplate:{TypedText A}{LeftAngle <}{Placeholder typename T}{RightAngle >} (50)
// CHECK-ACCESS-1: ClassTemplate:{TypedText B}{LeftAngle <}{Placeholder typename T}{RightAngle >} (50)

// RUN: c-index-test -code-completion-at=%s:25:6 %s | FileCheck -check-prefix=CHECK-ACCESS-2 %s
// CHECK-ACCESS-2: ClassTemplate:{TypedText C}{LeftAngle <}{Placeholder typename T}{RightAngle >} (50)
