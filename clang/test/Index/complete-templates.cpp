// Tests are line- and column-sensive, so run lines are below.

template<typename T>
class X {
  X();
  X(const X&);
  
  template<typename U> X(U);
};

template<typename T> void f(T);

void test() {
  
}

// RUN: c-index-test -code-completion-at=%s:14:2 %s | FileCheck %s
// CHECK: FunctionTemplate:{ResultType void}{TypedText f}{LeftParen (}{Placeholder T}{RightParen )} (50)
// CHECK: ClassTemplate:{TypedText X}{LeftAngle <}{Placeholder typename T}{RightAngle >} (50)
