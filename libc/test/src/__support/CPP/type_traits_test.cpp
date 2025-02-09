//===-- Unittests for type_traits -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

// TODO: Split this file if it becomes too big.

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

class Class {};
union Union {};
struct Struct {};
enum Enum {};
enum class EnumClass {};

using UnqualObjectTypes = testing::TypeList<int, float, Class, Union, Struct>;

TYPED_TEST(LlvmLibcTypeTraitsTest, add_lvalue_reference, UnqualObjectTypes) {
  // non-ref cv, adds ref
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<T>, T &>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<const T>, const T &>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<volatile T>, volatile T &>));
  EXPECT_TRUE((
      is_same_v<add_lvalue_reference_t<const volatile T>, const volatile T &>));

  // pointer cv, adds ref
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<T *>, T *&>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<const T *>, const T *&>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<volatile T *>, volatile T *&>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<const volatile T *>,
                         const volatile T *&>));

  // ref cv, returns same type
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<T &>, T &>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<const T &>, const T &>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<volatile T &>, volatile T &>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<const volatile T &>,
                         const volatile T &>));
}

TEST(LlvmLibcTypeTraitsTest, add_lvalue_reference_void) {
  // void cannot be referenced
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<void>, void>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<const void>, const void>));
  EXPECT_TRUE(
      (is_same_v<add_lvalue_reference_t<volatile void>, volatile void>));
  EXPECT_TRUE((is_same_v<add_lvalue_reference_t<const volatile void>,
                         const volatile void>));
}

TYPED_TEST(LlvmLibcTypeTraitsTest, add_pointer, UnqualObjectTypes) {
  // object types -> pointer type
  EXPECT_TRUE((is_same_v<add_pointer_t<T>, T *>));
  EXPECT_TRUE((is_same_v<add_pointer_t<const T>, const T *>));
  EXPECT_TRUE((is_same_v<add_pointer_t<volatile T>, volatile T *>));
  EXPECT_TRUE((is_same_v<add_pointer_t<const volatile T>, const volatile T *>));

  // pointer types -> pointer type
  EXPECT_TRUE((is_same_v<add_pointer_t<T *>, T **>));
  EXPECT_TRUE((is_same_v<add_pointer_t<const T *>, const T **>));
  EXPECT_TRUE((is_same_v<add_pointer_t<volatile T *>, volatile T **>));
  EXPECT_TRUE(
      (is_same_v<add_pointer_t<const volatile T *>, const volatile T **>));

  // reference type -> pointer type
  EXPECT_TRUE((is_same_v<add_pointer_t<T &>, T *>));
  EXPECT_TRUE((is_same_v<add_pointer_t<const T &>, const T *>));
  EXPECT_TRUE((is_same_v<add_pointer_t<volatile T &>, volatile T *>));
  EXPECT_TRUE(
      (is_same_v<add_pointer_t<const volatile T &>, const volatile T *>));
}

TEST(LlvmLibcTypeTraitsTest, add_pointer_void) {
  // void -> pointer type
  EXPECT_TRUE((is_same_v<add_pointer_t<void>, void *>));
  EXPECT_TRUE((is_same_v<add_pointer_t<const void>, const void *>));
  EXPECT_TRUE((is_same_v<add_pointer_t<volatile void>, volatile void *>));
  EXPECT_TRUE(
      (is_same_v<add_pointer_t<const volatile void>, const volatile void *>));
}

TYPED_TEST(LlvmLibcTypeTraitsTest, add_rvalue_reference, UnqualObjectTypes) {

  // non-ref cv, adds ref
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<T>, T &&>));
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<const T>, const T &&>));
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<volatile T>, volatile T &&>));
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<const volatile T>,
                         const volatile T &&>));

  // ref cv, returns same type
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<T &>, T &>));
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<const T &>, const T &>));
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<volatile T &>, volatile T &>));
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<const volatile T &>,
                         const volatile T &>));
}

TEST(LlvmLibcTypeTraitsTest, add_rvalue_reference_void) {
  // void cannot be referenced
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<void>, void>));
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<const void>, const void>));
  EXPECT_TRUE(
      (is_same_v<add_rvalue_reference_t<volatile void>, volatile void>));
  EXPECT_TRUE((is_same_v<add_rvalue_reference_t<const volatile void>,
                         const volatile void>));
}

TEST(LlvmLibcTypeTraitsTest, aligned_storage) {
  struct S {
    int a, b;
  };
  aligned_storage_t<sizeof(S), alignof(S)> buf;
  EXPECT_EQ(alignof(decltype(buf)), alignof(S));
  EXPECT_EQ(sizeof(buf), sizeof(S));
}

TEST(LlvmLibcTypeTraitsTest, bool_constant) {
  EXPECT_TRUE((bool_constant<true>::value));
  EXPECT_FALSE((bool_constant<false>::value));
}

TEST(LlvmLibcTypeTraitsTest, conditional_t) {
  EXPECT_TRUE((is_same_v<conditional_t<true, int, float>, int>));
  EXPECT_TRUE((is_same_v<conditional_t<false, int, float>, float>));
}

TEST(LlvmLibcTypeTraitsTest, decay) {
  EXPECT_TRUE((is_same_v<decay_t<int>, int>));

  // array decay
  EXPECT_TRUE((is_same_v<decay_t<int[2]>, int *>));
  EXPECT_TRUE((is_same_v<decay_t<int[2]>, int *>));
  EXPECT_TRUE((is_same_v<decay_t<int[2][4]>, int(*)[4]>));

  // cv ref decay
  EXPECT_TRUE((is_same_v<decay_t<int &>, int>));
  EXPECT_TRUE((is_same_v<decay_t<const int &>, int>));
  EXPECT_TRUE((is_same_v<decay_t<volatile int &>, int>));
  EXPECT_TRUE((is_same_v<decay_t<const volatile int &>, int>));
}

// TODO enable_if

TEST(LlvmLibcTypeTraitsTest, false_type) { EXPECT_FALSE((false_type::value)); }

TEST(LlvmLibcTypeTraitsTest, integral_constant) {
  EXPECT_EQ((integral_constant<int, 4>::value), 4);
}

namespace invoke_detail {

enum State { INIT = 0, A_APPLY_CALLED, B_APPLY_CALLED };

struct A {
  State state = INIT;
  virtual ~A() {}
  virtual void apply() { state = A_APPLY_CALLED; }
};

struct B : public A {
  virtual ~B() {}
  virtual void apply() override { state = B_APPLY_CALLED; }
};

void free_function() {}
int free_function_return_5() { return 5; }
int free_function_passtrough(int value) { return value; }

struct Delegate {
  int (*ptr)(int) = &free_function_passtrough;
};

template <int tag> struct Tag {
  static constexpr int value = tag;
};

struct Functor {
  auto operator()() & { return Tag<0>(); }
  auto operator()() const & { return Tag<1>(); }
  auto operator()() && { return Tag<2>(); }
  auto operator()() const && { return Tag<3>(); }

  const Tag<0> &operator()(const Tag<0> &a) { return a; }
  const Tag<0> &&operator()(const Tag<0> &&a) { return cpp::move(a); }
  Tag<1> operator()(Tag<1> a) { return a; }
};

} // namespace invoke_detail

TEST(LlvmLibcTypeTraitsTest, invoke) {
  using namespace invoke_detail;
  { // member function call
    A a;
    EXPECT_EQ(a.state, INIT);
    invoke(&A::apply, a);
    EXPECT_EQ(a.state, A_APPLY_CALLED);
  }
  { // overriden member function call
    B b;
    EXPECT_EQ(b.state, INIT);
    invoke(&A::apply, b);
    EXPECT_EQ(b.state, B_APPLY_CALLED);
  }
  { // free function
    invoke(&free_function);
    EXPECT_EQ(invoke(&free_function_return_5), 5);
    EXPECT_EQ(invoke(&free_function_passtrough, 1), 1);
  }
  { // pointer member function call
    Delegate d;
    EXPECT_EQ(invoke(&Delegate::ptr, d, 2), 2);
  }
  { // Functor with several ref qualifier
    Functor f;
    const Functor cf;
    EXPECT_EQ(invoke(f).value, 0);
    EXPECT_EQ(invoke(cf).value, 1);
    EXPECT_EQ(invoke(move(f)).value, 2);
    EXPECT_EQ(invoke(move(cf)).value, 3);
  }
  { // lambda
    EXPECT_EQ(invoke([]() -> int { return 2; }), 2);
    EXPECT_EQ(invoke([](int value) -> int { return value; }, 1), 1);

    const auto lambda = [](int) { return 0; };
    EXPECT_EQ(invoke(lambda, 1), 0);
  }
}

TEST(LlvmLibcTypeTraitsTest, invoke_result) {
  using namespace invoke_detail;
  EXPECT_TRUE((is_same_v<invoke_result_t<void (A::*)(), A>, void>));
  EXPECT_TRUE((is_same_v<invoke_result_t<void (A::*)(), B>, void>));
  EXPECT_TRUE((is_same_v<invoke_result_t<void (*)()>, void>));
  EXPECT_TRUE((is_same_v<invoke_result_t<int (*)()>, int>));
  EXPECT_TRUE((is_same_v<invoke_result_t<int (*)(int), int>, int>));
  EXPECT_TRUE((
      is_same_v<invoke_result_t<int (*Delegate::*)(int), Delegate, int>, int>));
  // Functor with several ref qualifiers
  EXPECT_TRUE((is_same_v<invoke_result_t<Functor &>, Tag<0>>));
  EXPECT_TRUE((is_same_v<invoke_result_t<Functor const &>, Tag<1>>));
  EXPECT_TRUE((is_same_v<invoke_result_t<Functor &&>, Tag<2>>));
  EXPECT_TRUE((is_same_v<invoke_result_t<Functor const &&>, Tag<3>>));
  // Functor with several arg qualifiers
  EXPECT_TRUE(
      (is_same_v<invoke_result_t<Functor &&, Tag<0> &>, const Tag<0> &>));
  EXPECT_TRUE((is_same_v<invoke_result_t<Functor, Tag<0>>, const Tag<0> &&>));
  EXPECT_TRUE((is_same_v<invoke_result_t<Functor, Tag<1>>, Tag<1>>));
  {
    auto lambda = []() {};
    EXPECT_TRUE((is_same_v<invoke_result_t<decltype(lambda)>, void>));
  }
  {
    auto lambda = []() { return 0; };
    EXPECT_TRUE((is_same_v<invoke_result_t<decltype(lambda)>, int>));
  }
  {
    auto lambda = [](int) -> double { return 0; };
    EXPECT_TRUE((is_same_v<invoke_result_t<decltype(lambda), int>, double>));
  }
}

using IntegralAndFloatingTypes =
    testing::TypeList<bool, char, short, int, long, long long, unsigned char,
                      unsigned short, unsigned int, unsigned long,
                      unsigned long long, float, double, long double>;

TYPED_TEST(LlvmLibcTypeTraitsTest, is_arithmetic, IntegralAndFloatingTypes) {
  EXPECT_TRUE((is_arithmetic_v<T>));
  EXPECT_TRUE((is_arithmetic_v<const T>));
  EXPECT_TRUE((is_arithmetic_v<volatile T>));
  EXPECT_TRUE((is_arithmetic_v<const volatile T>));

  EXPECT_FALSE((is_arithmetic_v<T *>));
  EXPECT_FALSE((is_arithmetic_v<T &>));
}

TEST(LlvmLibcTypeTraitsTest, is_arithmetic_non_integral) {
  EXPECT_FALSE((is_arithmetic_v<Union>));
  EXPECT_FALSE((is_arithmetic_v<Class>));
  EXPECT_FALSE((is_arithmetic_v<Struct>));
  EXPECT_FALSE((is_arithmetic_v<Enum>));
}

TEST(LlvmLibcTypeTraitsTest, is_array) {
  EXPECT_FALSE((is_array_v<int>));
  EXPECT_FALSE((is_array_v<float>));
  EXPECT_FALSE((is_array_v<Struct>));
  EXPECT_FALSE((is_array_v<int *>));

  EXPECT_TRUE((is_array_v<Class[]>));
  EXPECT_TRUE((is_array_v<Union[4]>));
}

TEST(LlvmLibcTypeTraitsTest, is_base_of) {
  struct A {};
  EXPECT_TRUE((is_base_of_v<A, A>));

  // Test public, protected and private inheritance.
  struct B : public A {};
  EXPECT_TRUE((is_base_of_v<A, B>));
  EXPECT_FALSE((is_base_of_v<B, A>));

  struct C : protected A {};
  EXPECT_TRUE((is_base_of_v<A, C>));
  EXPECT_FALSE((is_base_of_v<C, A>));

  struct D : private A {};
  EXPECT_TRUE((is_base_of_v<A, D>));
  EXPECT_FALSE((is_base_of_v<D, A>));

  // Test inheritance chain.
  struct E : private B {};
  EXPECT_TRUE((is_base_of_v<A, E>));
}

TEST(LlvmLibcTypeTraitsTest, is_class) {
  EXPECT_TRUE((is_class_v<Struct>));
  EXPECT_TRUE((is_class_v<Class>));

  // Pointer or ref do not qualify.
  EXPECT_FALSE((is_class_v<Class *>));
  EXPECT_FALSE((is_class_v<Class &>));

  // Neither other types.
  EXPECT_FALSE((is_class_v<Union>));
  EXPECT_FALSE((is_class_v<int>));
  EXPECT_FALSE((is_class_v<EnumClass>));
}

TYPED_TEST(LlvmLibcTypeTraitsTest, is_const, UnqualObjectTypes) {
  EXPECT_FALSE((is_const_v<T>));
  EXPECT_TRUE((is_const_v<const T>));

  using Aliased = const T;
  EXPECT_TRUE((is_const_v<Aliased>));
}

// TODO is_convertible

TYPED_TEST(LlvmLibcTypeTraitsTest, is_destructible, UnqualObjectTypes) {
  EXPECT_TRUE((is_destructible_v<T>));
}
TEST(LlvmLibcTypeTraitsTest, is_destructible_no_destructor) {
  struct S {
    ~S() = delete;
  };
  EXPECT_FALSE((is_destructible_v<S>));
}

TYPED_TEST(LlvmLibcTypeTraitsTest, is_enum, UnqualObjectTypes) {
  EXPECT_FALSE((is_enum_v<T>));
}
TEST(LlvmLibcTypeTraitsTest, is_enum_enum) {
  EXPECT_TRUE((is_enum_v<Enum>));
  EXPECT_TRUE((is_enum_v<EnumClass>));
}

// TODO is_floating_point

// TODO is_function

// TODO is_integral

// TODO is_lvalue_reference

// TODO is_member_pointer

// TODO is_null_pointer

TEST(LlvmLibcTypeTraitsTest, is_object) {
  EXPECT_TRUE((is_object_v<int>));      // scalar
  EXPECT_TRUE((is_object_v<Struct[]>)); // array
  EXPECT_TRUE((is_object_v<Union>));    // union
  EXPECT_TRUE((is_object_v<Class>));    // class

  // pointers are still objects
  EXPECT_TRUE((is_object_v<int *>));       // scalar
  EXPECT_TRUE((is_object_v<Struct(*)[]>)); // array
  EXPECT_TRUE((is_object_v<Union *>));     // union
  EXPECT_TRUE((is_object_v<Class *>));     // class

  // reference are not objects
  EXPECT_FALSE((is_object_v<int &>));       // scalar
  EXPECT_FALSE((is_object_v<Struct(&)[]>)); // array
  EXPECT_FALSE((is_object_v<Union &>));     // union
  EXPECT_FALSE((is_object_v<Class &>));     // class

  // not an object
  EXPECT_FALSE((is_object_v<void>));
}

// TODO is_pointer

// TODO is_reference

// TODO is_rvalue_reference

// TODO is_same

// TODO is_scalar

// TODO is_signed

// TODO is_trivially_constructible

// TODO is_trivially_copyable

// TODO is_trivially_destructible

// TODO is_union

// TODO is_unsigned

// TODO is_void

// TODO make_signed

// TODO make_unsigned

// TODO remove_all_extents

// TODO remove_cv

// TODO remove_cvref

// TODO remove_extent

// TODO remove_reference

TEST(LlvmLibcTypeTraitsTest, true_type) { EXPECT_TRUE((true_type::value)); }

struct CompilerLeadingPadded {
  char b;
  int a;
};

struct CompilerTrailingPadded {
  int a;
  char b;
};

struct alignas(long long) ManuallyPadded {
  int b;
  char padding[sizeof(long long) - sizeof(int)];
};

TEST(LlvmLibcTypeTraitsTest, has_unique_object_representations) {
  EXPECT_TRUE(has_unique_object_representations<int>::value);
  EXPECT_FALSE(has_unique_object_representations_v<CompilerLeadingPadded>);
  EXPECT_FALSE(has_unique_object_representations_v<CompilerTrailingPadded>);
  EXPECT_TRUE(has_unique_object_representations_v<ManuallyPadded>);
}

// TODO type_identity

// TODO void_t

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL
