// RUN: %clang_cc1 -fsyntax-only -std=c++03 -verify -ast-dump %s > %t-03
// RUN: FileCheck --input-file=%t-03 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -ast-dump %s > %t-11
// RUN: FileCheck --input-file=%t-11 %s
// RUN: FileCheck --input-file=%t-11 %s --check-prefix=CHECK-CXX11

// http://llvm.org/PR7905
namespace PR7905 {
struct S; // expected-note {{forward declaration}}
void foo1() {
  (void)(S[]) {{3}}; // expected-error {{array has incomplete element type}}
}

template <typename T> struct M { T m; };
void foo2() {
  (void)(M<short> []) {{3}};
}
}

// Check compound literals mixed with C++11 list-initialization.
namespace brace_initializers {
  struct POD {
    int x, y;
  };
  struct HasCtor {
    HasCtor(int x, int y);
  };
  struct HasDtor {
    int x, y;
    ~HasDtor();
  };
  struct HasCtorDtor {
    HasCtorDtor(int x, int y);
    ~HasCtorDtor();
  };

  void test() {
    (void)(POD){1, 2};
    // CHECK-NOT: CXXBindTemporaryExpr {{.*}} 'brace_initializers::POD'
    // CHECK: CompoundLiteralExpr {{.*}} 'brace_initializers::POD'
    // CHECK-NEXT: InitListExpr {{.*}} 'brace_initializers::POD'
    // CHECK-NEXT: IntegerLiteral {{.*}} 1{{$}}
    // CHECK-NEXT: IntegerLiteral {{.*}} 2{{$}}

    (void)(HasDtor){1, 2};
    // CHECK: CXXBindTemporaryExpr {{.*}} 'brace_initializers::HasDtor'
    // CHECK-NEXT: CompoundLiteralExpr {{.*}} 'brace_initializers::HasDtor'
    // CHECK-NEXT: InitListExpr {{.*}} 'brace_initializers::HasDtor'
    // CHECK-NEXT: IntegerLiteral {{.*}} 1{{$}}
    // CHECK-NEXT: IntegerLiteral {{.*}} 2{{$}}

#if __cplusplus >= 201103L
    (void)(HasCtor){1, 2};
    // CHECK-CXX11-NOT: CXXBindTemporaryExpr {{.*}} 'brace_initializers::HasCtor'
    // CHECK-CXX11: CompoundLiteralExpr {{.*}} 'brace_initializers::HasCtor'
    // CHECK-CXX11-NEXT: CXXTemporaryObjectExpr {{.*}} 'brace_initializers::HasCtor'
    // CHECK-CXX11-NEXT: IntegerLiteral {{.*}} 1{{$}}
    // CHECK-CXX11-NEXT: IntegerLiteral {{.*}} 2{{$}}

    (void)(HasCtorDtor){1, 2};
    // CHECK-CXX11: CXXBindTemporaryExpr {{.*}} 'brace_initializers::HasCtorDtor'
    // CHECK-CXX11-NEXT: CompoundLiteralExpr {{.*}} 'brace_initializers::HasCtorDtor'
    // CHECK-CXX11-NEXT: CXXTemporaryObjectExpr {{.*}} 'brace_initializers::HasCtorDtor'
    // CHECK-CXX11-NEXT: IntegerLiteral {{.*}} 1{{$}}
    // CHECK-CXX11-NEXT: IntegerLiteral {{.*}} 2{{$}}
#endif
  }

  struct PrivateDtor {
    int x, y;
  private:
    ~PrivateDtor(); // expected-note {{declared private here}}
  };

  void testPrivateDtor() {
    (void)(PrivateDtor){1, 2}; // expected-error {{temporary of type 'brace_initializers::PrivateDtor' has private destructor}}
  }
}

// This doesn't necessarily need to be an error, but CodeGen can't handle it
// at the moment.
int PR17415 = (int){PR17415}; // expected-error {{initializer element is not a compile-time constant}}

// Make sure we accept this.  (Not sure if we actually should... but we do
// at the moment.)
template<unsigned> struct Value { };
template<typename T>
int &check_narrowed(Value<sizeof((T){1.1})>);

#if __cplusplus >= 201103L
// Compound literals in global lambdas have automatic storage duration
// and are not subject to the constant-initialization rules.
int computed_with_lambda = [] {
  int x = 5;
  int result = ((int[]) { x, x + 2, x + 4, x + 6 })[0];
  return result;
}();
#endif
