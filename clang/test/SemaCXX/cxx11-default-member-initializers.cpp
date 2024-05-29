// RUN: %clang_cc1 -std=c++11 -verify %s -pedantic
// RUN: %clang_cc1 -std=c++11 -verify %s -pedantic -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++20 -verify %s -pedantic
// RUN: %clang_cc1 -std=c++20 -verify %s -pedantic -fexperimental-new-constant-interpreter


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

namespace std {
typedef decltype(sizeof(int)) size_t;

// libc++'s implementation
template <class _E> class initializer_list {
  const _E *__begin_;
  size_t __size_;

  initializer_list(const _E *__b, size_t __s) : __begin_(__b), __size_(__s) {}

public:
  typedef _E value_type;
  typedef const _E &reference;
  typedef const _E &const_reference;
  typedef size_t size_type;

  typedef const _E *iterator;
  typedef const _E *const_iterator;

  initializer_list() : __begin_(nullptr), __size_(0) {}

  size_t size() const { return __size_; }
  const _E *begin() const { return __begin_; }
  const _E *end() const { return __begin_ + __size_; }
};
} // namespace std

#if __cplusplus >= 201703L
namespace test_rebuild {
template <typename T, int> class C {
public:
  C(std::initializer_list<T>);
};

template <typename T> using Ptr = __remove_pointer(T) *;
template <typename T> C(T) -> C<Ptr<T>, sizeof(T)>;

class A {
public:
  template <typename T1, typename T2> T1 *some_func(T2 &&);
};

struct B : A {
  // Test CXXDefaultInitExpr rebuild issue in 
  // https://github.com/llvm/llvm-project/pull/87933
  int *ar = some_func<int>(C{some_func<int>(0)});
  B() {}
};

int TestBody_got;
template <int> class Vector {
public:
  Vector(std::initializer_list<int>);
};
template <typename... Ts> Vector(Ts...) -> Vector<sizeof...(Ts)>;
class ProgramBuilder {
public:
  template <typename T, typename ARGS> int *create(ARGS);
};

struct TypeTest : ProgramBuilder {
  int *str_f16 = create<int>(Vector{0});
  TypeTest() {}
};
class TypeTest_Element_Test : TypeTest {
  void TestBody();
};
void TypeTest_Element_Test::TestBody() {
  int *expect = str_f16;
  &TestBody_got != expect; // expected-warning {{inequality comparison result unused}}
}
} //  namespace test_rebuild
#endif // __cplusplus >= 201703L

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
