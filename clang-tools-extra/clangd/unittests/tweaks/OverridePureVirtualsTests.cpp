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

  void F2(int P1, const int & P2) override {
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

} // namespace
} // namespace clangd
} // namespace clang
