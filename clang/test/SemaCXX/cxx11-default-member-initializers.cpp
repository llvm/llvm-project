// RUN: %clang_cc1 -std=c++11 -verify %s -pedantic
// RUN: %clang_cc1 -std=c++20 -verify %s -pedantic


namespace PR31692 {
  struct A {
    struct X { int n = 0; } x;
    // Trigger construction of X() from a SFINAE context. This must not mark
    // any part of X as invalid.
    static_assert(!__is_constructible(X), "");
    // Check that X::n is not marked invalid.
    double &r = x.n; // expected-error {{non-const lvalue reference to type 'double' cannot bind to a value of unrelated type 'int'}}
  };
  // A::X can now be default-constructed.
  static_assert(__is_constructible(A::X), "");
}


struct S {
} constexpr s;
struct C {
  C(S);
};
class MemInit {
  C m = s;
};

#if __cplusplus >= 202002L
// This test ensures cleanup expressions are correctly produced
// in the presence of default member initializers.
namespace PR136554 {
struct string {
  constexpr string(const char*) {};
  constexpr ~string();
};
struct S;
struct optional {
    template <typename U = S>
    constexpr optional(U &&) {}
};
struct S {
    string a;
    optional b;
    int defaulted = 0;
} test {
    "", {
        { "", 0 }
    }
};

// Ensure that the this pointer is
// transformed without crashing
consteval int immediate() { return 0;}
struct StructWithThisInInitializer {
  int member() const {
      return 0;
  }
  int m = member() + immediate();
  int m2 = this->member() + immediate();
};

template <typename T>
struct StructWithThisInInitializerTPL {
  template <typename U>
  int member() const {
      return 0;
  }
  int m = member<int>() + immediate();
  int m2 = this->member<int>() + immediate();
};

void test_this() {
  (void)StructWithThisInInitializer{};
  (void)StructWithThisInInitializerTPL<int>{};
}

struct ReferenceToNestedMembers {
  int m;
  int a = ((void)immediate(), m); // ensure g is found in the correct scope
  int b = ((void)immediate(), this->m); // ensure g is found in the correct scope
};
struct ReferenceToNestedMembersTest {
 void* m = nullptr;
 ReferenceToNestedMembers j{0};
} test_reference_to_nested_members;

}


namespace odr_in_unevaluated_context {
template <typename e, bool = __is_constructible(e)> struct f {
    using type = bool;
};

template <class k, f<k>::type = false> int l;
int m;
struct p {
  // This used to crash because m is first marked odr used
  // during parsing, but subsequently used in an unevaluated context
  // without being transformed.
  int o = m;
  p() {}
};

int i = l<p>;
}

#endif
