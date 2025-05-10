//===-- AddPureVirtualOverrideTest.cpp --------------------------*- C++ -*-===//
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

class OverridePureVirtualsTest : public TweakTest {
protected:
  OverridePureVirtualsTest() : TweakTest("OverridePureVirtuals") {}
};

TEST_F(OverridePureVirtualsTest, MinimalUnavailable) {
  EXPECT_UNAVAILABLE("class ^C {};");
}

TEST_F(OverridePureVirtualsTest, MinimalAvailable) {
  EXPECT_AVAILABLE(R"cpp(
class B { public: virtual void Foo() = 0; };
class ^C : public B {};
)cpp");
}

TEST_F(OverridePureVirtualsTest, Availability) {
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

TEST_F(OverridePureVirtualsTest, EmptyDerivedClass) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `F1` is not implemented.");
  }
  void F2(int P1, const int & P2) override {
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `F2` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTest, SingleBaseClassPartiallyImplemented) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `F2` is not implemented.");
  }

  void F1() override;
};
)cpp";
  EXPECT_EQ(Applied, Expected) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTest, MultipleDirectBaseClasses) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `func1` is not implemented.");
  }

protected:
  bool func2(char c) const override {
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `func2` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTest, UnnamedParametersInBase) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `func` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTest, DiamondInheritance) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `diamond_func` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTest, MixedAccessSpecifiers) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `pub_func` is not implemented.");
  }
  void pub_func2(char) const override {
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `pub_func2` is not implemented.");
  }

  Derived(int m) : member(m) {}

protected:
  int prot_func(int x) const override {
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `prot_func` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTest, OutOfOrderMixedAccessSpecifiers) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `prot_func` is not implemented.");
  }

  void foo();
public:
  void pub_func() override {
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `pub_func` is not implemented.");
  }
  void pub_func2(char) const override {
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `pub_func2` is not implemented.");
  }

  Derived(int m) : member(m) {}
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTest, MultiAccessSpecifiersOverride) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `foo` is not implemented.");
  }

protected:
  void bar() override {
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `bar` is not implemented.");
  }
};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

TEST_F(OverridePureVirtualsTest, AccessSpecifierAlreadyExisting) {
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
    // TODO: Implement this pure virtual method
    static_assert(false, "Method `func1` is not implemented.");
  }

};
)cpp";
  auto Applied = apply(Before);
  EXPECT_EQ(Expected, Applied) << "Applied result:\n" << Applied;
}

} // namespace
} // namespace clangd
} // namespace clang
