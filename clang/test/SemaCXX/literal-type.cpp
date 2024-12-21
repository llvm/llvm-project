// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s


static_assert(__is_literal(int), "fail");
static_assert(__is_literal_type(int), "fail"); // alternate spelling for GCC
static_assert(__is_literal(void*), "fail");
enum E { E1 };
static_assert(__is_literal(E), "fail");
static_assert(__is_literal(decltype(E1)), "fail");
typedef int IAR[10];
static_assert(__is_literal(IAR), "fail");
typedef int Vector __attribute__((vector_size(16)));
typedef int VectorExt __attribute__((ext_vector_type(4)));
static_assert(__is_literal(Vector), "fail");
static_assert(__is_literal(VectorExt), "fail");

// C++0x [basic.types]p10:
//   A type is a literal type if it is:
//    [...]
//    -- a class type that has all of the following properties:
//        -- it has a trivial destructor
//           [P0784R7 changed the condition to "constexpr destructor" in C++20]
//        -- every constructor call and full-expression in the
//           brace-or-equal-initializers for non-static data members (if an) is
//           a constant expression,
//        -- it is an aggregate type or has at least one constexpr constructor
//           or constructor template that is not a copy or move constructor, and
//           [DR1452 adds class types with trivial default constructors to
//            this list]
//        -- it has all non-static data members and base classes of literal
//           types
struct Empty {};
struct LiteralType {
  int x;
  E e;
  IAR arr;
  Empty empty;
  int method();
};
struct HasDtor { ~HasDtor(); };

class NonAggregate { int x; };
struct NonLiteral { NonLiteral(); };
struct HasNonLiteralBase : NonLiteral {};
struct HasNonLiteralMember { HasDtor x; };

static_assert(__is_literal(Empty), "fail");
static_assert(__is_literal(LiteralType), "fail");
static_assert(__is_literal(NonAggregate), "fail");
static_assert(!__is_literal(NonLiteral), "fail");
static_assert(!__is_literal(HasDtor), "fail");
static_assert(!__is_literal(HasNonLiteralBase), "fail");
static_assert(!__is_literal(HasNonLiteralMember), "fail");

// DR1361 removes the brace-or-equal-initializer bullet so that we can allow:
extern int f(); // expected-note {{here}}
struct HasNonConstExprMemInit {
  int x = f(); // expected-note {{non-constexpr function}}
  constexpr HasNonConstExprMemInit() {} // expected-error {{never produces a constant expression}}
  constexpr HasNonConstExprMemInit(int y) : x(y) {} // ok
};
static_assert(__is_literal(HasNonConstExprMemInit), "fail");

class HasConstExprCtor {
  int x;
public:
  constexpr HasConstExprCtor(int x) : x(x) {}
};
template <typename T> class HasConstExprCtorTemplate {
  T x;
public:
  template <typename U> constexpr HasConstExprCtorTemplate(U y) : x(y) {}
};
template <typename T> class HasConstExprCtorT {
  constexpr HasConstExprCtorT(T) {}
};
static_assert(__is_literal(HasConstExprCtor), "fail");
static_assert(__is_literal(HasConstExprCtorTemplate<int>), "fail");
static_assert(__is_literal(HasConstExprCtorT<NonLiteral>), "fail");


#if __cplusplus >= 202003L
namespace GH77924 {

struct A { A(); };
template <class T>
struct opt {
  union Data {
      constexpr Data() : x{} {}
      constexpr ~Data() {}
      char x;
      T data;
  };

  constexpr opt() : data{} {}
  constexpr ~opt() { if (engaged) data.data.~T(); }
  Data data;
  bool engaged = false;
};

consteval void foo() {
  opt<A> a;
}

void test() {
  foo();
}

}
#endif

#if __cplusplus >= 201103L
namespace GH85550 {
struct HasDefaultCtorAndNonConstexprDtor {
  constexpr HasDefaultCtorAndNonConstexprDtor() = default;
  ~HasDefaultCtorAndNonConstexprDtor() {}
};

union UnionWithNonLiteralMember {
  HasDefaultCtorAndNonConstexprDtor x;
  int y;

  constexpr UnionWithNonLiteralMember() : x{} {}
};
#if __cplusplus >= 202002L
static_assert(__is_literal(UnionWithNonLiteralMember), "fail");
#else
static_assert(!__is_literal(UnionWithNonLiteralMember), "fail");
#endif

union UnionWithNonLiteralMemberExplicitDtor1 {
  HasDefaultCtorAndNonConstexprDtor x;
  int y;
  // expected-note@-2 {{destructor of 'UnionWithNonLiteralMemberExplicitDtor1' is implicitly deleted because variant field 'x' has a non-trivial destructor}}

  constexpr UnionWithNonLiteralMemberExplicitDtor1() : x{} {}
  ~UnionWithNonLiteralMemberExplicitDtor1() = default; // expected-warning {{explicitly defaulted destructor is implicitly deleted}}
  // expected-note@-1 {{replace 'default' with 'delete'}}
};
#if __cplusplus >= 202002L
static_assert(__is_literal(UnionWithNonLiteralMemberExplicitDtor1), "fail");
#else
static_assert(!__is_literal(UnionWithNonLiteralMemberExplicitDtor1), "fail");
#endif

union UnionWithNonLiteralMemberExplicitDtor2 {
  HasDefaultCtorAndNonConstexprDtor x;
  int y;

  constexpr UnionWithNonLiteralMemberExplicitDtor2() : x{} {}
  ~UnionWithNonLiteralMemberExplicitDtor2() = delete;
};
static_assert(!__is_literal(UnionWithNonLiteralMemberExplicitDtor2), "fail");

#if __cplusplus >= 202002L
union UnionWithNonLiteralMemberConstexprDtor1 {
  HasDefaultCtorAndNonConstexprDtor x;
  int y;
  // expected-note@-2 {{destructor of 'UnionWithNonLiteralMemberConstexprDtor1' is implicitly deleted because variant field 'x' has a non-trivial destructor}}

  constexpr UnionWithNonLiteralMemberConstexprDtor1() : x{} {}
  constexpr ~UnionWithNonLiteralMemberConstexprDtor1() = default; // expected-warning {{explicitly defaulted destructor is implicitly deleted}}
  // expected-note@-1 {{replace 'default' with 'delete'}}
};
static_assert(__is_literal(UnionWithNonLiteralMemberConstexprDtor1), "fail");

union UnionWithNonLiteralMemberConstexprDtor2 {
  HasDefaultCtorAndNonConstexprDtor x;
  int y;

  constexpr UnionWithNonLiteralMemberConstexprDtor2() : x{} {}
  constexpr ~UnionWithNonLiteralMemberConstexprDtor2() = delete;
};
static_assert(__is_literal(UnionWithNonLiteralMemberConstexprDtor2), "fail");
#endif
}
#endif
