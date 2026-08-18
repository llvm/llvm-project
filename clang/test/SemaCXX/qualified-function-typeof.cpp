// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

using F = void() const;

struct S {
  using F = void() const;
};

void testTypeOf() {
  using T1 = __typeof__(F);
  using T2 = __typeof__(S::F);
  using T3 = __typeof_unqual__(F);
  using T4 = __typeof_unqual__(S::F);
}

template <typename T>
struct TypeIdentity {
  using type = T;
};

void testTemplateArg() {
  using U1 = TypeIdentity<F>::type;
  using U2 = TypeIdentity<S::F>::type;
}

namespace std { class type_info; }

namespace test_other_contexts {
  void test() {
    auto &a = typeid(F); // expected-error {{type operand 'F' (aka 'void () const') of 'typeid' cannot have 'const' qualifier}}
    auto &b = typeid(void() const); // expected-error {{type operand 'void () const' of 'typeid' cannot have 'const' qualifier}}

    unsigned s1 = sizeof(F); // expected-error {{invalid application of 'sizeof' to a function type}}
    unsigned s2 = sizeof(void() const); // expected-error {{invalid application of 'sizeof' to a function type}}
    unsigned a1 = alignof(F); // expected-error {{invalid application of 'alignof' to a function type}}
    unsigned a2 = alignof(void() const); // expected-error {{invalid application of 'alignof' to a function type}}

    auto c1 = (F)nullptr; // expected-error {{C-style cast from 'std::nullptr_t' to 'F' (aka 'void () const') is not allowed}}
    auto c2 = (void() const)nullptr; // expected-error {{C-style cast from 'std::nullptr_t' to 'void () const' is not allowed}}

    auto n1 = new F; // expected-error {{non-member function of type 'F' (aka 'void () const') cannot have 'const' qualifier}} \
                     // expected-error {{cannot allocate function type 'void ()' with new}}
    auto n2 = new (void() const); // expected-error {{non-member function cannot have 'const' qualifier}} \
                                  // expected-error {{cannot allocate function type 'void ()' with new}}
  }
}
