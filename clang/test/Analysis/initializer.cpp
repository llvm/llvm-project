// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++11
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++17
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++11\
// RUN:   -DTEST_INLINABLE_ALLOCATORS
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++17\
// RUN:   -DTEST_INLINABLE_ALLOCATORS

void clang_analyzer_eval(bool);

#include "Inputs/system-header-simulator-cxx.h"

class A {
  int x;
public:
  A();
};

A::A() : x(0) {
  clang_analyzer_eval(x == 0); // expected-warning{{TRUE}}
}


class DirectMember {
  int x;
public:
  DirectMember(int value) : x(value) {}

  int getX() { return x; }
};

void testDirectMember() {
  DirectMember obj(3);
  clang_analyzer_eval(obj.getX() == 3); // expected-warning{{TRUE}}
}


class IndirectMember {
  struct {
    int x;
  };
public:
  IndirectMember(int value) : x(value) {}

  int getX() { return x; }
};

void testIndirectMember() {
  IndirectMember obj(3);
  clang_analyzer_eval(obj.getX() == 3); // expected-warning{{TRUE}}
}


struct DelegatingConstructor {
  int x;
  DelegatingConstructor(int y) { x = y; }
  DelegatingConstructor() : DelegatingConstructor(42) {}
};

void testDelegatingConstructor() {
  DelegatingConstructor obj;
  clang_analyzer_eval(obj.x == 42); // expected-warning{{TRUE}}
}


struct RefWrapper {
  RefWrapper(int *p) : x(*p) {}
  RefWrapper(int &r) : x(r) {}
  int &x;
};

void testReferenceMember() {
  int *p = 0;
  RefWrapper X(p); // expected-warning@-7 {{Dereference of null pointer}}
}

void testReferenceMember2() {
  int *p = 0;
  RefWrapper X(*p); // expected-warning {{Forming reference to null pointer}}
}


extern "C" char *strdup(const char *);

class StringWrapper {
  char *str;
public:
  StringWrapper(const char *input) : str(strdup(input)) {} // no-warning
};


// PR15070 - Constructing a type containing a non-POD array mistakenly
// tried to perform a bind instead of relying on the CXXConstructExpr,
// which caused a cast<> failure in RegionStore.
namespace DefaultConstructorWithCleanups {
  class Element {
  public:
    int value;

    class Helper {
    public:
      ~Helper();
    };
    Element(Helper h = Helper());
  };
  class Wrapper {
  public:
    Element arr[2];

    Wrapper();
  };

  Wrapper::Wrapper() /* initializers synthesized */ {}

  int test() {
    Wrapper w;
    return w.arr[0].value; // no-warning
  }
}

namespace DefaultMemberInitializers {
  struct Wrapper {
    int value = 42;

    Wrapper() {}
    Wrapper(int x) : value(x) {}
    Wrapper(bool) {}
  };

  void test() {
    Wrapper w1;
    clang_analyzer_eval(w1.value == 42); // expected-warning{{TRUE}}

    Wrapper w2(50);
    clang_analyzer_eval(w2.value == 50); // expected-warning{{TRUE}}

    Wrapper w3(false);
    clang_analyzer_eval(w3.value == 42); // expected-warning{{TRUE}}
  }

  struct StringWrapper {
    const char s[4] = "abc";
    const char *p = "xyz";

    StringWrapper(bool) {}
  };

  void testString() {
    StringWrapper w(true);
    clang_analyzer_eval(w.s[1] == 'b'); // expected-warning{{TRUE}}
    clang_analyzer_eval(w.p[1] == 'y'); // expected-warning{{TRUE}}
  }
}

namespace ReferenceInitialization {
  struct OtherStruct {
    OtherStruct(int i);
    ~OtherStruct();
  };

  struct MyStruct {
    MyStruct(int i);
    MyStruct(OtherStruct os);

    void method() const;
  };

  void referenceInitializeLocal() {
    const MyStruct &myStruct(5);
    myStruct.method(); // no-warning
  }

  void referenceInitializeMultipleLocals() {
    const MyStruct &myStruct1(5), myStruct2(5), &myStruct3(5);
    myStruct1.method(); // no-warning
    myStruct2.method(); // no-warning
    myStruct3.method(); // no-warning
  }

  void referenceInitializeLocalWithCleanup() {
    const MyStruct &myStruct(OtherStruct(5));
    myStruct.method(); // no-warning
  }
};

namespace PR31592 {
struct C {
   C() : f("}") { } // no-crash
   const char(&f)[2];
};
}

namespace CXX_initializer_lists {
struct C {
  C(std::initializer_list<int *> list);
};
void testPointerEscapeIntoLists() {
  C empty{}; // no-crash

  // Do not warn that 'x' leaks. It might have been deleted by
  // the destructor of 'c'.
  int *x = new int;
  C c{x}; // no-warning
}

void testPassListsWithExplicitConstructors() {
  (void)(std::initializer_list<int>){12}; // no-crash
}
}

namespace CXX17_aggregate_construction {
struct A {
  A();
};

struct B: public A {
};

struct C: public B {
};

struct D: public virtual A {
};

// In C++17, classes B and C are aggregates, so they will be constructed
// without actually calling their trivial constructor. Used to crash.
void foo() {
  B b = {}; // no-crash
  const B &bl = {}; // no-crash
  B &&br = {}; // no-crash

  C c = {}; // no-crash
  const C &cl = {}; // no-crash
  C &&cr = {}; // no-crash

  D d = {}; // no-crash

#if __cplusplus >= 201703L
  C cd = {{}}; // no-crash
  const C &cdl = {{}}; // no-crash
  C &&cdr = {{}}; // no-crash

  const B &bll = {{}}; // no-crash
  const B &bcl = C({{}}); // no-crash
  B &&bcr = C({{}}); // no-crash
#endif
}
} // namespace CXX17_aggregate_construction

namespace newexpr_init_list_initialization {
template <class FirstT, class... Rest>
void escape(FirstT first, Rest... args);

struct S {
  int foo;
  int bar;
};
void none_designated() {
  S *s = new S{13,1};
  clang_analyzer_eval(13 == s->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == s->bar); // expected-warning{{TRUE}}
  delete s;
}
void none_designated_swapped() {
  S *s = new S{1,13};
  clang_analyzer_eval(1 == s->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == s->bar); // expected-warning{{TRUE}}
  delete s;
}
void one_designated_one_not() {
  S *s = new S{ 1, .bar = 13 };
  clang_analyzer_eval(1 == s->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(13 == s->bar); // expected-warning{{TRUE}}
  delete s;
}
void all_designated() {
  S *s = new S{
      .foo = 13,
      .bar = 1,
  };
  clang_analyzer_eval(13 == s->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == s->bar); // expected-warning{{TRUE}}
  delete s;
}
void non_designated_array_of_aggr_struct() {
  S *s = new S[2] { {1, 2}, {3, 4} };
  clang_analyzer_eval(1 == s[0].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == s[0].bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == s[1].foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(4 == s[1].bar); // expected-warning{{TRUE}}
  delete[] s;
}

struct WithGaps {
  int foo;
  int bar;
  int baz;
};
void out_of_order_designated_initializers_with_gaps() {
  WithGaps *s = new WithGaps{
    .foo = 13,
    .baz = 1,
  };
  clang_analyzer_eval(13 == s->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == s->bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == s->baz); // expected-warning{{TRUE}}
  delete s;
}

// https://eel.is/c++draft/dcl.init.aggr#note-6:
// Static data members, non-static data members of anonymous
// union members, and unnamed bit-fields are not considered
// elements of the aggregate.
struct NonConsideredFields {
  int i;
  static int s;
  int j;
  int :17;
  int k;
};
void considered_fields_initd() {
  auto S = new NonConsideredFields { 1, 2, 3 };
  clang_analyzer_eval(1 == S->i); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == S->j); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == S->k); // expected-warning{{TRUE}}
  delete S;
}

#if __cplusplus >= 201703L
enum Enum : int {
};
void list_init_enum() {
  Enum *E = new Enum{53};
  clang_analyzer_eval(53 == *E); // expected-warning{{TRUE}}
  delete E;
}
#endif // __cplusplus >= 201703L

class PubClass {
public:
  int foo;
  int bar;
};
void public_class_designated_initializers() {
  S *s = new S{
      .foo = 13,
      .bar = 1,
  };
  clang_analyzer_eval(13 == s->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(1 == s->bar); // expected-warning{{TRUE}}
  delete s;
}

union UnionTestTy {
  int x;
  char y;
};
void new_expr_aggr_init_union_no_designator() {
  UnionTestTy *u = new UnionTestTy{};
  clang_analyzer_eval(0 == u->x); // expected-warning{{UNKNOWN}} FIXME: should be TRUE
  clang_analyzer_eval(u->y); // expected-warning{{UNKNOWN}} FIXME: should be undefined, warning
  delete u;
}
void new_expr_aggr_init_union_designated_first_field() {
  UnionTestTy *u = new UnionTestTy{ .x = 14 };
  clang_analyzer_eval(14 == u->x); // expected-warning{{UNKNOWN}} FIXME: should be TRUE
  clang_analyzer_eval(u->y); // expected-warning{{UNKNOWN}} FIXME: should be undefined, warning
  delete u;
}
void new_expr_aggr_init_union_designated_non_first_field() {
  UnionTestTy *u = new UnionTestTy{ .y = 3 };
  clang_analyzer_eval(3 == u->y); // expected-warning{{UNKNOWN}} FIXME: should be TRUE
  clang_analyzer_eval(u->x); // expected-warning{{UNKNOWN}} FIXME: should be undefined, warning
  delete u;
}

union UnionTestTyWithDefaultMemberInit {
  int x;
  char y = 14;
};
void union_with_default_member_init_empty_init_list() {
  auto U = new UnionTestTyWithDefaultMemberInit{};
  // clang_analyzer_eval(14 == U->y); // FIXME: Should be true
  clang_analyzer_eval(U->x); // expected-warning{{UNKNOWN}} FIXME: should be undefined, warning
  delete U;
}

struct Inner {
  int bar;
};
struct Nested {
  int foo;
  Inner inner;
  int baz;
};
void nested_aggregates() {
  auto N = new Nested{};
  clang_analyzer_eval(0 == N->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N->baz); // expected-warning{{TRUE}}

  auto N1 = new Nested{1};
  clang_analyzer_eval(1 == N1->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N1->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N1->baz); // expected-warning{{TRUE}}

  auto N2 = new Nested{.baz = 14};
  clang_analyzer_eval(0 == N2->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N2->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(14 == N2->baz); // expected-warning{{TRUE}}

  auto N3 = new Nested{1,2,3};
  clang_analyzer_eval(1 == N3->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == N3->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N3->baz); // expected-warning{{TRUE}}

  auto N4 = new Nested{1, {}, 3};
  clang_analyzer_eval(1 == N4->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N4->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N4->baz); // expected-warning{{TRUE}}

  auto N5 = new Nested{{},{},{}};
  clang_analyzer_eval(0 == N5->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N5->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(0 == N5->baz); // expected-warning{{TRUE}}

  auto N6 = new Nested{1, {.bar = 2}, 3};
  clang_analyzer_eval(1 == N6->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == N6->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N6->baz); // expected-warning{{TRUE}}

  auto N7 = new Nested{1, {2}, 3};
  clang_analyzer_eval(1 == N7->foo); // expected-warning{{TRUE}}
  clang_analyzer_eval(2 == N7->inner.bar); // expected-warning{{TRUE}}
  clang_analyzer_eval(3 == N7->baz); // expected-warning{{TRUE}}

  escape(N,N1,N2,N3,N4,N5,N6,N7);
}
} // namespace newexpr_init_list_initialization

namespace placement_new_initializer_list_arg {
struct S {
  int x;
};
void aggregate_struct() {
  S s;
  S *s_ptr = new (&s) S{1};
  clang_analyzer_eval(1 == s_ptr->x); // expected-warning{{TRUE}}

  S vi;
  S *vi_ptr = new (&vi) S{};
  clang_analyzer_eval(0 == vi_ptr->x); // expected-warning{{TRUE}}

  S di;
  S *di_ptr = new (&di) S;
  int z = di_ptr->x + 1; // expected-warning{{The left operand of '+' is a garbage value}}
}
void initialize_non_zeroth_element(S arr[2]) {
  S *s = new (&arr[1]) S{1};
  clang_analyzer_eval(1 == s->x); // expected-warning{{TRUE}}
}
void initialize_non_zeroth_argument_pointers(S *arr[2]) {
  arr[1] = new (arr[1]) S{1};
  clang_analyzer_eval(1 == arr[1]->x); // expected-warning{{TRUE}}
}
} // namespace placement_new_initializer_list_arg

namespace CXX17_transparent_init_list_exprs {
class A {};

class B: private A {};

B boo();
void foo1() {
  B b { boo() }; // no-crash
}

class C: virtual public A {};

C coo();
void foo2() {
  C c { coo() }; // no-crash
}

B foo_recursive() {
  B b { foo_recursive() };
}
} // namespace CXX17_transparent_init_list_exprs

namespace skip_vbase_initializer_side_effects {
int glob;
struct S {
  S() { ++glob; }
};

struct A {
  A() {}
  A(S s) {}
};

struct B : virtual A {
  B() : A(S()) {}
};

struct C : B {
  C() {}
};

void foo() {
  glob = 0;
  B b;
  clang_analyzer_eval(glob == 1); // expected-warning{{TRUE}}
  C c; // no-crash
  clang_analyzer_eval(glob == 1); // expected-warning{{TRUE}}
}
} // namespace skip_vbase_initializer_side_effects

namespace dont_skip_vbase_initializers_in_most_derived_class {
struct A {
  static int a;
  A() { a = 0; }
  A(int x) { a = x; }
};

struct B {
  static int b;
  B() { b = 0; }
  B(int y) { b = y; }
};

struct C : virtual A {
  C() : A(1) {}
};
struct D : C, virtual B {
  D() : B(2) {}
};

void testD() {
  D d;
  clang_analyzer_eval(A::a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(B::b == 2); // expected-warning{{TRUE}}
}

struct E : virtual B, C {
  E() : B(2) {}
};

void testE() {
  E e;
  clang_analyzer_eval(A::a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(B::b == 2); // expected-warning{{TRUE}}
}

struct F : virtual A, virtual B {
  F() : A(1) {}
};
struct G : F {
  G(): B(2) {}
};

void testG() {
  G g;
  clang_analyzer_eval(A::a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(B::b == 2); // expected-warning{{TRUE}}
}

struct H : virtual B, virtual A {
  H(): A(1) {}
};
struct I : H {
  I(): B(2) {}
};

void testI() {
  I i;
  clang_analyzer_eval(A::a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(B::b == 2); // expected-warning{{TRUE}}
}
} // namespace dont_skip_vbase_initializers_in_most_derived_class

namespace elementwise_copy_small_array_from_post_initializer_of_cctor {
struct String {
  String(const String &) {}
};

struct MatchComponent {
  unsigned numbers[2];
  String prerelease;
  MatchComponent(MatchComponent const &) = default;
};

MatchComponent get();
void consume(MatchComponent const &);

MatchComponent parseMatchComponent() {
  MatchComponent component = get();
  component.numbers[0] = 10;
  component.numbers[1] = 20;
  return component; // We should have no stack addr escape warning here.
}

void top() {
  consume(parseMatchComponent());
}
} // namespace elementwise_copy_small_array_from_post_initializer_of_cctor

namespace gh147686 {
// The problem reported in https://github.com/llvm/llvm-project/issues/147686
// is sensitive to the initializer form: using parenthesis to initialize m_ptr
// resulted in crashes when analyzing *m_ptr = '\0'; but using braces is fine.

struct A {
  A() : m_ptr(m_buf) { *m_ptr = '\0'; } // no-crash
  A(int overload) : m_ptr{m_buf} { *m_ptr = '\0'; }
  A(char src) : m_ptr(m_buf) { *m_ptr = src; } // no-crash
  A(char src, int overload) : m_ptr{m_buf} { *m_ptr = src; }
  char m_buf[64] = {0};
  char * m_ptr;
};

void test1() {
  A a;
  clang_analyzer_eval(a.m_buf[0] == 0); // expected-warning{{TRUE}}
  // FIXME The next eval should result in TRUE.
  clang_analyzer_eval(*a.m_ptr == 0); // expected-warning{{UNKNOWN}}
}

void test2() {
  A a(314);
  clang_analyzer_eval(a.m_buf[0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(*a.m_ptr == 0); // expected-warning{{TRUE}}
}

void test3() {
  A a(0);
  clang_analyzer_eval(a.m_buf[0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(*a.m_ptr == 0); // expected-warning{{TRUE}}
}

void test3Bis(char arg) {
  A a(arg);
  // FIXME This test should behave like test3.
  clang_analyzer_eval(a.m_buf[0] == arg); // expected-warning{{FALSE}} // expected-warning{{TRUE}}
  clang_analyzer_eval(*a.m_ptr == arg); // expected-warning{{UNKNOWN}}
}

void test4(char arg) {
  A a(arg, 314);
  clang_analyzer_eval(a.m_buf[0] == arg); // expected-warning{{TRUE}}
  clang_analyzer_eval(*a.m_ptr == arg); // expected-warning{{TRUE}}
}

} // namespace gh147686
