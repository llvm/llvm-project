//===-- OverridePureVirtualsTests.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

class OverridePureVirtualsTests : public TweakTest {
protected:
  OverridePureVirtualsTests() : TweakTest("OverridePureVirtuals") {}
};

TEST_F(OverridePureVirtualsTests, MinimalUnavailable) {
  EXPECT_UNAVAILABLE("class ^C {};");
}

TEST_F(OverridePureVirtualsTests, MinimalAvailable) {
  EXPECT_AVAILABLE(R"cpp(
class B { public: virtual void Foo() = 0; };
class ^C : public B {};
)cpp");
}

TEST_F(OverridePureVirtualsTests, UnavailableWhenOverriden) {
  EXPECT_UNAVAILABLE(
      R"cpp(
class B {
public:
  virtual void foo() = 0;
};

class ^D : public B {
public:
  void foo() override;
};
)cpp");
}

TEST_F(OverridePureVirtualsTests, AvailabilityNoOverride) {
  EXPECT_AVAILABLE(R"cpp(
class Base {
public:
virtual ~Base() = default;
virtual void F1() = 0;
virtual void F2() = 0;
};

class ^Derived : public Base {
public:
};

)cpp");
}

TEST_F(OverridePureVirtualsTests, AvailabilityPartiallyOverridden) {
  EXPECT_AVAILABLE(R"cpp(
class Base {
public:
virtual ~Base() = default;
virtual void F1() = 0;
virtual void F2() = 0;
};

class ^Derived : public Base {
public:
void F1() override;
};
)cpp");
}

TEST_F(OverridePureVirtualsTests, EmptyDerivedClass) {
  const char *Before = R"cpp(
class Base {
public:
virtual ~Base() = default;
virtual void F1() = 0;
virtual void F2(int P1, const int &P2) = 0;
};

class ^Derived : public Base {};
)cpp";
  const auto *Expected = R"cpp(
class Base {
public:
virtual ~Base() = default;
virtual void F1() = 0;
virtual void F2(int P1, const int &P2) = 0;
};

class Derived : public Base {
public:
  void F1() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `F1` is not implemented.");
  }

  void F2(int P1, const int &P2) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `F2` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, SingleBaseClassPartiallyImplemented) {
  auto Applied = apply(
      R"cpp(
class Base {
public:
virtual ~Base() = default;
virtual void F1() = 0;
virtual void F2() = 0;
};

class ^Derived : public Base {
public:
  void F1() override;
};
)cpp");

  const auto *Expected = R"cpp(
class Base {
public:
virtual ~Base() = default;
virtual void F1() = 0;
virtual void F2() = 0;
};

class Derived : public Base {
public:
  void F2() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `F2` is not implemented.");
  }

  void F1() override;
};
)cpp";
  EXPECT_EQ(Applied, Expected) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, MultipleDirectBaseClasses) {
  const char *Before = R"cpp(
class Base1 {
public:
  virtual void func1() = 0;
};
class Base2 {
protected:
  virtual bool func2(char c) const = 0;
};

class ^Derived : public Base1, public Base2 {};
)cpp";
  const auto *Expected = R"cpp(
class Base1 {
public:
  virtual void func1() = 0;
};
class Base2 {
protected:
  virtual bool func2(char c) const = 0;
};

class Derived : public Base1, public Base2 {
public:
  void func1() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `func1` is not implemented.");
  }
protected:
  bool func2(char c) const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `func2` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, UnnamedParametersInBase) {
  const char *Before = R"cpp(
struct S {};
class Base {
public:
  virtual void func(int, const S&, char*) = 0;
};

class ^Derived : public Base {};
)cpp";

  const auto *Expected = R"cpp(
struct S {};
class Base {
public:
  virtual void func(int, const S&, char*) = 0;
};

class Derived : public Base {
public:
  void func(int, const S &, char *) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `func` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, DiamondInheritance) {
  const char *Before = R"cpp(
class Top {
public:
  virtual ~Top() = default;
  virtual void diamond_func() = 0;
};
class Left : virtual public Top {};
class Right : virtual public Top {};
class ^Bottom : public Left, public Right {};
)cpp";
  const auto *Expected = R"cpp(
class Top {
public:
  virtual ~Top() = default;
  virtual void diamond_func() = 0;
};
class Left : virtual public Top {};
class Right : virtual public Top {};
class Bottom : public Left, public Right {
public:
  void diamond_func() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `diamond_func` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, MixedAccessSpecifiers) {
  const char *Before = R"cpp(
class Base {
public:
  virtual void pub_func() = 0;
  virtual void pub_func2(char) const = 0;
protected:
  virtual int prot_func(int x) const = 0;
};

class ^Derived : public Base {
  int member; // Existing member
public:
  Derived(int m) : member(m) {}
};
)cpp";
  const auto *Expected = R"cpp(
class Base {
public:
  virtual void pub_func() = 0;
  virtual void pub_func2(char) const = 0;
protected:
  virtual int prot_func(int x) const = 0;
};

class Derived : public Base {
  int member; // Existing member
public:
  void pub_func() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `pub_func` is not implemented.");
  }

  void pub_func2(char) const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `pub_func2` is not implemented.");
  }

  Derived(int m) : member(m) {}

protected:
  int prot_func(int x) const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `prot_func` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, OutOfOrderMixedAccessSpecifiers) {
  const char *Before = R"cpp(
class Base {
public:
  virtual void pub_func() = 0;
  virtual void pub_func2(char) const = 0;
protected:
  virtual int prot_func(int x) const = 0;
};

class ^Derived : public Base {
  int member; // Existing member
protected:
  void foo();
public:
  Derived(int m) : member(m) {}
};
)cpp";
  const auto *Expected = R"cpp(
class Base {
public:
  virtual void pub_func() = 0;
  virtual void pub_func2(char) const = 0;
protected:
  virtual int prot_func(int x) const = 0;
};

class Derived : public Base {
  int member; // Existing member
protected:
  int prot_func(int x) const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `prot_func` is not implemented.");
  }

  void foo();
public:
  void pub_func() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `pub_func` is not implemented.");
  }

  void pub_func2(char) const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `pub_func2` is not implemented.");
  }

  Derived(int m) : member(m) {}
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, MultiAccessSpecifiersOverride) {
  constexpr auto Before = R"cpp(
class Base {
public:
  virtual void foo() = 0;
protected:
  virtual void bar() = 0;
};

class ^Derived : public Base {};
)cpp";

  constexpr auto Expected = R"cpp(
class Base {
public:
  virtual void foo() = 0;
protected:
  virtual void bar() = 0;
};

class Derived : public Base {
public:
  void foo() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `foo` is not implemented.");
  }
protected:
  void bar() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `bar` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, AccessSpecifierAlreadyExisting) {
  const char *Before = R"cpp(
class Base {
public:
  virtual void func1() = 0;
};

class ^Derived : public Base {
public:
};
)cpp";

  const auto *Expected = R"cpp(
class Base {
public:
  virtual void func1() = 0;
};

class Derived : public Base {
public:
  void func1() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `func1` is not implemented.");
  }

};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, ConstexprSpecifier) {
  ExtraArgs.push_back("-std=c++20");

  constexpr auto Before = R"cpp(
class B {
public:
  constexpr virtual int getValue() const = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  constexpr virtual int getValue() const = 0;
};

class D : public B {
public:
  constexpr int getValue() const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `getValue` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, ConstevalSpecifier) {
  ExtraArgs.push_back("-std=c++20");

  constexpr auto Before = R"cpp(
class B {
public:
  virtual consteval float calculate() = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  virtual consteval float calculate() = 0;
};

class D : public B {
public:
  consteval float calculate() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `calculate` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, LValueRefQualifier) {
  constexpr auto Before = R"cpp(
class B {
public:
  virtual void process() & = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  virtual void process() & = 0;
};

class D : public B {
public:
  void process() & override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `process` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, RValueRefQualifier) {
  constexpr auto Before = R"cpp(
class B {
public:
  virtual bool isValid() && = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  virtual bool isValid() && = 0;
};

class D : public B {
public:
  bool isValid() && override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `isValid` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, SimpleTrailingReturnType) {
  constexpr auto Before = R"cpp(
class B {
public:
  virtual auto getStatus() -> bool = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  virtual auto getStatus() -> bool = 0;
};

class D : public B {
public:
  auto getStatus() -> bool override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `getStatus` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, ConstexprLValueRefAndTrailingReturn) {
  ExtraArgs.push_back("-std=c++20");

  constexpr auto Before = R"cpp(
class B {
public:
  constexpr virtual auto getData() & -> const char * = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  constexpr virtual auto getData() & -> const char * = 0;
};

class D : public B {
public:
  constexpr auto getData() & -> const char * override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `getData` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, ConstevalRValueRefAndTrailingReturn) {
  ExtraArgs.push_back("-std=c++20");

  constexpr auto Before = R"cpp(
class B {
public:
  virtual consteval auto foo() && -> double = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  virtual consteval auto foo() && -> double = 0;
};

class D : public B {
public:
  consteval auto foo() && -> double override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `foo` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, CombinedFeaturesWithTrailingReturnTypes) {
  ExtraArgs.push_back("-std=c++20");

  constexpr auto Before = R"cpp(
class B {
public:
  virtual auto f1() & -> int = 0;
  constexpr virtual auto f2() && -> int = 0;
  virtual consteval auto f3() -> int = 0;
  virtual auto f4() const & -> char = 0;
  constexpr virtual auto f5() const && -> bool = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  virtual auto f1() & -> int = 0;
  constexpr virtual auto f2() && -> int = 0;
  virtual consteval auto f3() -> int = 0;
  virtual auto f4() const & -> char = 0;
  constexpr virtual auto f5() const && -> bool = 0;
};

class D : public B {
public:
  auto f1() & -> int override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f1` is not implemented.");
  }

  constexpr auto f2() && -> int override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f2` is not implemented.");
  }

  consteval auto f3() -> int override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f3` is not implemented.");
  }

  auto f4() const & -> char override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f4` is not implemented.");
  }

  constexpr auto f5() const && -> bool override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f5` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, DefaultParameters) {
  ExtraArgs.push_back("-std=c++20");

  constexpr auto Before = R"cpp(
class B {
public:
  virtual void foo(int var = 0) = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  virtual void foo(int var = 0) = 0;
};

class D : public B {
public:
  void foo(int var = 0) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `foo` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, VariadicMethods) {
  constexpr auto Before = R"cpp(
class Base {
public:
  virtual void foo(int, ...) = 0;
};

class De^riv : Base {};
)cpp";

  constexpr auto Expected = R"cpp(
class Base {
public:
  virtual void foo(int, ...) = 0;
};

class Deriv : Base {
public:
  void foo(int, ...) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `foo` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, ExceptionSpecifiers) {
  constexpr auto Before = R"cpp(
class Base {
public:
  virtual void f1() noexcept = 0;
  virtual void f2() noexcept(sizeof(int) > 2) = 0;
  virtual void f3() throw() = 0;
};

class De^riv : Base {};
)cpp";

  constexpr auto Expected = R"cpp(
class Base {
public:
  virtual void f1() noexcept = 0;
  virtual void f2() noexcept(sizeof(int) > 2) = 0;
  virtual void f3() throw() = 0;
};

class Deriv : Base {
public:
  void f1() noexcept override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f1` is not implemented.");
  }

  void f2() noexcept(sizeof(int) > 2) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f2` is not implemented.");
  }

  void f3() throw() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f3` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, QualifiedNames) {
  constexpr auto Before = R"cpp(
namespace foo { struct S{}; namespace bar { struct S2{}; } }

class B {
public:
  virtual foo::S foo(int var = 0) = 0;
  virtual foo::bar::S2 bar(int var = 0) = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
namespace foo { struct S{}; namespace bar { struct S2{}; } }

class B {
public:
  virtual foo::S foo(int var = 0) = 0;
  virtual foo::bar::S2 bar(int var = 0) = 0;
};

class D : public B {
public:
  foo::S foo(int var = 0) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `foo` is not implemented.");
  }

  foo::bar::S2 bar(int var = 0) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `bar` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, NestedQualifiedNames) {
  constexpr auto Before = R"cpp(
namespace A {
namespace B {
class S {};
class Base {
public:
  class T {};
  virtual S bar(S &s) = 0;
  virtual T foo(T &t) = 0;
};
}

namespace C {
class D^eriv : B::Base {};
}
}
)cpp";

  constexpr auto Expected = R"cpp(
namespace A {
namespace B {
class S {};
class Base {
public:
  class T {};
  virtual S bar(S &s) = 0;
  virtual T foo(T &t) = 0;
};
}

namespace C {
class Deriv : B::Base {
public:
  B::S bar(B::S &s) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `bar` is not implemented.");
  }

  B::Base::T foo(B::Base::T &t) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `foo` is not implemented.");
  }
};
}
}
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, NestedQualifiedNamesWithTrailingReturnTypes) {
  constexpr auto Before = R"cpp(
namespace A {
namespace B {
class S {};
class Base {
public:
  class T {};
  virtual auto bar(S &s) -> S = 0;
  virtual auto foo(T &t) -> T = 0;
};
}

namespace C {
class D^eriv : B::Base {};
}
}
)cpp";

  constexpr auto Expected = R"cpp(
namespace A {
namespace B {
class S {};
class Base {
public:
  class T {};
  virtual auto bar(S &s) -> S = 0;
  virtual auto foo(T &t) -> T = 0;
};
}

namespace C {
class Deriv : B::Base {
public:
  auto bar(B::S &s) -> B::S override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `bar` is not implemented.");
  }

  auto foo(B::Base::T &t) -> B::Base::T override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `foo` is not implemented.");
  }
};
}
}
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, Operators) {
  constexpr auto Before = R"cpp(
class B {
public:
  virtual ~B() = default;
  virtual float & operator()(int i) = 0;
  virtual const float & operator()(int i) const = 0;
  virtual float & operator[](int i) = 0;
  virtual bool operator!() = 0;
  virtual float operator-() = 0;
  virtual B & operator++() = 0;
  virtual B operator++(int) = 0;
  virtual B operator+(const B &o) const = 0;
  virtual B & operator=(const B &o) = 0;
  virtual B & operator+=(const B &o) = 0;
  virtual B & operator=(B &&o) noexcept = 0;
  virtual bool operator==(const B &o) const = 0;
  virtual B* operator->() const = 0;
  virtual B & operator*() = 0;
  virtual operator bool() const = 0;
};

class ^D : public B {};
)cpp";

  constexpr auto Expected = R"cpp(
class B {
public:
  virtual ~B() = default;
  virtual float & operator()(int i) = 0;
  virtual const float & operator()(int i) const = 0;
  virtual float & operator[](int i) = 0;
  virtual bool operator!() = 0;
  virtual float operator-() = 0;
  virtual B & operator++() = 0;
  virtual B operator++(int) = 0;
  virtual B operator+(const B &o) const = 0;
  virtual B & operator=(const B &o) = 0;
  virtual B & operator+=(const B &o) = 0;
  virtual B & operator=(B &&o) noexcept = 0;
  virtual bool operator==(const B &o) const = 0;
  virtual B* operator->() const = 0;
  virtual B & operator*() = 0;
  virtual operator bool() const = 0;
};

class D : public B {
public:
  float & operator()(int i) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator()` is not implemented.");
  }

  const float & operator()(int i) const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator()` is not implemented.");
  }

  float & operator[](int i) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator[]` is not implemented.");
  }

  bool operator!() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator!` is not implemented.");
  }

  float operator-() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator-` is not implemented.");
  }

  B & operator++() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator++` is not implemented.");
  }

  B operator++(int) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator++` is not implemented.");
  }

  B operator+(const B &o) const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator+` is not implemented.");
  }

  B & operator=(const B &o) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator=` is not implemented.");
  }

  B & operator+=(const B &o) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator+=` is not implemented.");
  }

  B & operator=(B &&o) noexcept override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator=` is not implemented.");
  }

  bool operator==(const B &o) const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator==` is not implemented.");
  }

  B * operator->() const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator->` is not implemented.");
  }

  B & operator*() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator*` is not implemented.");
  }

  operator bool() const override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `operator bool` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTests, Attributes) {
  ExtraArgs = {"-fopenmp", "--target=x86_64-unknown-linux-gnu"};
  constexpr auto Before = R"cpp(
struct Base {
  virtual int f0() = 0;
  [ [gnu::hot] ] [ [gnu::const] ] [ [nodiscard] ] virtual int f1() = 0;
  [ [gnu::const, gnu::hot, nodiscard] ] virtual int f2() = 0;
  __attribute__((annotate("foo"))) __attribute__((cold)) virtual int f3() = 0;
#pragma omp declare simd
  [ [nodiscard] ] virtual int f4() = 0;
  virtual int __attribute__((ms_abi)) f5() = 0;
  virtual int f6() asm("sym") = 0;
  virtual int f7([ [maybe_unused] ] int x) = 0;
};

struct De^riv : Base {
};
)cpp";

  constexpr auto Expected = R"cpp(
struct Base {
  virtual int f0() = 0;
  [ [gnu::hot] ] [ [gnu::const] ] [ [nodiscard] ] virtual int f1() = 0;
  [ [gnu::const, gnu::hot, nodiscard] ] virtual int f2() = 0;
  __attribute__((annotate("foo"))) __attribute__((cold)) virtual int f3() = 0;
#pragma omp declare simd
  [ [nodiscard] ] virtual int f4() = 0;
  virtual int __attribute__((ms_abi)) f5() = 0;
  virtual int f6() asm("sym") = 0;
  virtual int f7([ [maybe_unused] ] int x) = 0;
};

struct Deriv : Base {

public:
  int f0() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f0` is not implemented.");
  }

  [[gnu::hot]] [[gnu::const]] [[nodiscard("")]] int f1() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f1` is not implemented.");
  }

  [[gnu::const]] [[gnu::hot]] [[nodiscard("")]] int f2() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f2` is not implemented.");
  }

  __attribute__((annotate("foo"))) __attribute__((cold)) int f3() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f3` is not implemented.");
  }

#pragma omp declare simd
  [[nodiscard("")]] int f4() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f4` is not implemented.");
  }

  int __attribute__((ms_abi)) f5() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f5` is not implemented.");
  }

  int f6() override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f6` is not implemented.");
  }

  int f7([[maybe_unused]] int x) override {
    // TODO: Implement this pure virtual method.
    static_assert(false, "Method `f7` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

} // namespace
} // namespace clangd
} // namespace clang
