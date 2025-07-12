// RUN: %check_clang_tidy -std=c++11 -check-suffix=11 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++14 -check-suffix=11,14 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++17 -check-suffix=11,14,17 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++20 -check-suffix=11,14,17,20 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffix=11,14,17,20,23 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing

// RUN: %check_clang_tidy -std=c++11 -check-suffix=11,11-CLT %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConservativeLiteralType: false}}" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++14 -check-suffix=11,11-CLT,14,14-CLT %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConservativeLiteralType: false}}" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++17 -check-suffix=11,11-CLT,14,14-CLT,17,17-CLT %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConservativeLiteralType: false}}" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++20 -check-suffix=11,11-CLT,14,14-CLT,17,17-CLT,20,20-CLT %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConservativeLiteralType: false}}" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffix=11,14,17,20,23 %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConservativeLiteralType: false}}" -- -fno-delayed-template-parsing

namespace {
namespace my {
  struct point {
    constexpr point() {}
    int get_x() const { return x; }
    // CHECK-MESSAGES-11: :[[@LINE-1]]:9: warning: function 'get_x' can be declared 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES-11: constexpr int get_x() const { return x; }
    int x;
    int y;
  };

  struct point2 {
    point2();
    int get_x() const { return x; }
    int x;
  };
} // namespace my
} // namespace

namespace function {
  struct Empty {};

  struct Base {
    virtual void virt() = 0;
  };
  struct Derived : Base {
    Derived() {}
    void virt() override {}
  };

  static void f1() {}
  // CHECK-MESSAGES-23: :[[@LINE-1]]:15: warning: function 'f1' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr void f1() {}

  static int f2() { return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f2' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f2() { return 0; }

  static int f3(int x) { return x; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f3' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f3(int x) { return x; }

  static int f4(Empty x) { return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f4' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f4(Empty x) { return 0; }

  static int f5(Empty x) { return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f5' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f5(Empty x) { return 0; }

  static int f6(Empty x) { ; return 0; }
  // CHECK-MESSAGES-14: :[[@LINE-1]]:14: warning: function 'f6' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-14: static constexpr int f6(Empty x) { ; return 0; }

  static int f7(Empty x) { static_assert(0 == 0, ""); return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f7' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f7(Empty x) { static_assert(0 == 0, ""); return 0; }

  static int f8(Empty x) { using my_int = int; return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f8' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f8(Empty x) { using my_int = int; return 0; }

  static int f9(Empty x) { using my::point; return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f9' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f9(Empty x) { using my::point; return 0; }

  static int f10(Empty x) { return 10; return 0; }
  // CHECK-MESSAGES-14: :[[@LINE-1]]:14: warning: function 'f10' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-14: static constexpr int f10(Empty x) { return 10; return 0; }

  static int f11(Empty x) { if (true) return 10; return 0; }
  // CHECK-MESSAGES-14: :[[@LINE-1]]:14: warning: function 'f11' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-14: static constexpr int f11(Empty x) { if (true) return 10; return 0; }

  static int f12(Empty x) { label: ; goto label; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f12' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f12(Empty x) { label: ; goto label; return 0; }
  static int f13(Empty x) { try { throw 0; } catch(int) {}; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f13' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f13(Empty x) { try { throw 0; } catch(int) {}; return 0; }
  static int f14(Empty x) { asm ("mov %rax, %rax"); }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f14' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f14(Empty x) { asm ("mov %rax, %rax"); }
  static int f15(Empty x) { int y; return 0; }
  // CHECK-MESSAGES-20: :[[@LINE-1]]:14: warning: function 'f15' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-20: static constexpr int f15(Empty x) { int y; return 0; }
  static int f16(Empty x) { static int y = 0; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f16' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f16(Empty x) { static int y = 0; return 0; }
  static int f17(Empty x) { thread_local int y = 0; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f17' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f17(Empty x) { thread_local int y = 0; return 0; }
  static int f18(Empty x) { [](){ label: ; goto label; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f18' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f18(Empty x) { [](){ label: ; goto label; return 0;  }; return 0; }
  static int f19(Empty x) { [](){ try { throw 0; } catch(int) {}; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f19' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f19(Empty x) { [](){ try { throw 0; } catch(int) {}; return 0;  }; return 0; }
  static int f20(Empty x) { [](){ asm ("mov %rax, %rax");  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f20' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f20(Empty x) { [](){ asm ("mov %rax, %rax");  }; return 0; }
  static int f21(Empty x) { [](){ int y; return 0;  }; return 0; }
  // CHECK-MESSAGES-20: :[[@LINE-1]]:14: warning: function 'f21' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-20: static constexpr int f21(Empty x) { [](){ int y; return 0;  }; return 0; }
  static int f22(Empty x) { [](){ static int y = 0; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f22' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f22(Empty x) { [](){ static int y = 0; return 0;  }; return 0; }
  static int f23(Empty x) { [](){ thread_local int y = 0; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f23' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f23(Empty x) { [](){ thread_local int y = 0; return 0;  }; return 0; }

  static int f24(Empty x) { return [](){ return 0; }(); }
  // CHECK-MESSAGES-17: :[[@LINE-1]]:14: warning: function 'f24' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-17: static constexpr int f24(Empty x) { return [](){ return 0; }(); }

  static int f25(Empty x) { new int; return 0; }
  // CHECK-MESSAGES-20: :[[@LINE-1]]:14: warning: function 'f25' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-20: static constexpr int f25(Empty x) { new int; return 0; }

  struct Range0To10 {
    struct iterator {
      int operator*() const;
      void operator++();
      friend bool operator!=(const iterator&lhs, const iterator&rhs) { return lhs.i == rhs.i; }
      int i;
    };
    iterator begin() const;
    iterator end() const;
  };
  static int f26(Empty x) {
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f26' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f26(Empty x) {
  // CHECK-MESSAGES-20-CLT: :[[@LINE-3]]:14: warning: function 'f26' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-20-CLT: static constexpr int f26(Empty x) {
    auto R = Range0To10{};
    for (const int i: R) { }
    return 0;
  }

  const auto f27 = [](int X){ return X + 1; }(10);
  // CHECK-MESSAGES-17: :[[@LINE-1]]:14: warning: variable 'f27' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-17: constexpr  auto f27 = [](int X){ return X + 1; }(10);

  [[nodiscard]] static int f28() { return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:28: warning: function 'f28' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: {{\[\[nodiscard\]\]}} static constexpr int f28() { return 0; }
} // namespace function
namespace function_non_literal {
  struct NonLiteral{
    NonLiteral();
    ~NonLiteral();
    int &ref;
  };

  struct Base {
    virtual void virt() = 0;
  };
  struct Derived : Base {
    Derived() {}
    void virt() override {}
  };

  static void f1() {}
  // CHECK-MESSAGES-23: :[[@LINE-1]]:15: warning: function 'f1' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr void f1() {}

  static int f2() { return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f2' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f2() { return 0; }

  static int f3(int x) { return x; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f3' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f3(int x) { return x; }

  static int f4(NonLiteral x) { return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f4' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f4(NonLiteral x) { return 0; }

  static int f5(NonLiteral x) { return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f5' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f5(NonLiteral x) { return 0; }

  static int f6(NonLiteral x) { ; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f6' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f6(NonLiteral x) { ; return 0; }

  static int f7(NonLiteral x) { static_assert(0 == 0, ""); return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f7' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f7(NonLiteral x) { static_assert(0 == 0, ""); return 0; }

  static int f8(NonLiteral x) { using my_int = int; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f8' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f8(NonLiteral x) { using my_int = int; return 0; }

  static int f9(NonLiteral x) { using my::point; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f9' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f9(NonLiteral x) { using my::point; return 0; }

  static int f10(NonLiteral x) { return 10; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f10' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f10(NonLiteral x) { return 10; return 0; }

  static int f11(NonLiteral x) { if (true) return 10; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f11' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f11(NonLiteral x) { if (true) return 10; return 0; }

  static int f12(NonLiteral x) { label: ; goto label; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f12' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f12(NonLiteral x) { label: ; goto label; return 0; }
  static int f13(NonLiteral x) { try { throw 0; } catch(int) {}; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f13' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f13(NonLiteral x) { try { throw 0; } catch(int) {}; return 0; }
  static int f14(NonLiteral x) { asm ("mov %rax, %rax"); }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f14' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f14(NonLiteral x) { asm ("mov %rax, %rax"); }
  static int f15(NonLiteral x) { int y; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f15' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f15(NonLiteral x) { int y; return 0; }
  static int f16(NonLiteral x) { static int y = 0; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f16' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f16(NonLiteral x) { static int y = 0; return 0; }
  static int f17(NonLiteral x) { thread_local int y = 0; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f17' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f17(NonLiteral x) { thread_local int y = 0; return 0; }
  static int f18(NonLiteral x) { [](){ label: ; goto label; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f18' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f18(NonLiteral x) { [](){ label: ; goto label; return 0;  }; return 0; }
  static int f19(NonLiteral x) { [](){ try { throw 0; } catch(int) {}; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f19' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f19(NonLiteral x) { [](){ try { throw 0; } catch(int) {}; return 0;  }; return 0; }
  static int f20(NonLiteral x) { [](){ asm ("mov %rax, %rax");  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f20' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f20(NonLiteral x) { [](){ asm ("mov %rax, %rax");  }; return 0; }
  static int f21(NonLiteral x) { [](){ int y; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f21' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f21(NonLiteral x) { [](){ int y; return 0;  }; return 0; }
  static int f22(NonLiteral x) { [](){ static int y = 0; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f22' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f22(NonLiteral x) { [](){ static int y = 0; return 0;  }; return 0; }
  static int f23(NonLiteral x) { [](){ thread_local int y = 0; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f23' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f23(NonLiteral x) { [](){ thread_local int y = 0; return 0;  }; return 0; }

  static int f24(NonLiteral x) { return [](){ return 0; }(); }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f24' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f24(NonLiteral x) { return [](){ return 0; }(); }

  static int f25(NonLiteral x) { new int; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f25' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f25(NonLiteral x) { new int; return 0; }

  struct Range0To10 {
    struct iterator {
      int operator*() const { return i; }
      void operator++() { ++i; }
      friend bool operator!=(const iterator&lhs, const iterator&rhs) { return lhs.i == rhs.i; }
      int i;
    };
    iterator begin() const { return { 0 }; }
    iterator end() const { return { 10 }; }
  };
  static int f26(NonLiteral x) {
    auto R = Range0To10{};
    for (const int i: R) { }
    return 0;
  }
  // CHECK-MESSAGES-23: :[[@LINE-5]]:14: warning: function 'f26' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f26(NonLiteral x) {
} // namespace function_non_literal
namespace function_non_literal_ref {
  struct NonLiteral{
    NonLiteral();
    ~NonLiteral();
    int &ref;
  };

  struct Base {
    virtual void virt() = 0;
  };
  struct Derived : Base {
    Derived() {}
    void virt() override {}
  };

  static void f1() {}
  // CHECK-MESSAGES-23: :[[@LINE-1]]:15: warning: function 'f1' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr void f1() {}

  static int f2() { return 0; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f2' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f2() { return 0; }

  static int f3(int x) { return x; }
  // CHECK-MESSAGES-11: :[[@LINE-1]]:14: warning: function 'f3' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11: static constexpr int f3(int x) { return x; }

  static int f4(NonLiteral& x) { return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f4' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f4(NonLiteral& x) { return 0; }
  // CHECK-MESSAGES-11-CLT: :[[@LINE-3]]:14: warning: function 'f4' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11-CLT: static constexpr int f4(NonLiteral& x) { return 0; }

  static int f5(NonLiteral& x) { return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f5' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f5(NonLiteral& x) { return 0; }
  // CHECK-MESSAGES-11-CLT: :[[@LINE-3]]:14: warning: function 'f5' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11-CLT: static constexpr int f5(NonLiteral& x) { return 0; }

  static int f6(NonLiteral& x) { ; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f6' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f6(NonLiteral& x) { ; return 0; }
  // CHECK-MESSAGES-14-CLT: :[[@LINE-3]]:14: warning: function 'f6' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-14-CLT: static constexpr int f6(NonLiteral& x) { ; return 0; }

  static int f7(NonLiteral& x) { static_assert(0 == 0, ""); return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f7' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f7(NonLiteral& x) { static_assert(0 == 0, ""); return 0; }
  // CHECK-MESSAGES-11-CLT: :[[@LINE-3]]:14: warning: function 'f7' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11-CLT: static constexpr int f7(NonLiteral& x) { static_assert(0 == 0, ""); return 0; }

  static int f8(NonLiteral& x) { using my_int = int; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f8' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f8(NonLiteral& x) { using my_int = int; return 0; }
  // CHECK-MESSAGES-11-CLT: :[[@LINE-3]]:14: warning: function 'f8' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11-CLT: static constexpr int f8(NonLiteral& x) { using my_int = int; return 0; }

  static int f9(NonLiteral& x) { using my::point; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f9' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f9(NonLiteral& x) { using my::point; return 0; }
  // CHECK-MESSAGES-11-CLT: :[[@LINE-3]]:14: warning: function 'f9' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-11-CLT: static constexpr int f9(NonLiteral& x) { using my::point; return 0; }

  static int f10(NonLiteral& x) { return 10; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f10' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f10(NonLiteral& x) { return 10; return 0; }
  // CHECK-MESSAGES-14-CLT: :[[@LINE-3]]:14: warning: function 'f10' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-14-CLT: static constexpr int f10(NonLiteral& x) { return 10; return 0; }

  static int f11(NonLiteral& x) { if (true) return 10; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f11' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f11(NonLiteral& x) { if (true) return 10; return 0; }
  // CHECK-MESSAGES-14-CLT: :[[@LINE-3]]:14: warning: function 'f11' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-14-CLT: static constexpr int f11(NonLiteral& x) { if (true) return 10; return 0; }

  static int f12(NonLiteral& x) { label: ; goto label; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f12' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f12(NonLiteral& x) { label: ; goto label; return 0; }
  static int f13(NonLiteral& x) { try { throw 0; } catch(int) {}; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f13' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f13(NonLiteral& x) { try { throw 0; } catch(int) {}; return 0; }
  static int f14(NonLiteral& x) { asm ("mov %rax, %rax"); }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f14' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f14(NonLiteral& x) { asm ("mov %rax, %rax"); }
  static int f15(NonLiteral& x) { int y; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f15' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f15(NonLiteral& x) { int y; return 0; }
  // CHECK-MESSAGES-20-CLT: :[[@LINE-3]]:14: warning: function 'f15' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-20-CLT: static constexpr int f15(NonLiteral& x) { int y; return 0; }
  static int f16(NonLiteral& x) { static int y = 0; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f16' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f16(NonLiteral& x) { static int y = 0; return 0; }
  static int f17(NonLiteral& x) { thread_local int y = 0; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f17' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f17(NonLiteral& x) { thread_local int y = 0; return 0; }
  static int f18(NonLiteral& x) { [](){ label: ; goto label; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f18' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f18(NonLiteral& x) { [](){ label: ; goto label; return 0;  }; return 0; }
  static int f19(NonLiteral& x) { [](){ try { throw 0; } catch(int) {}; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f19' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f19(NonLiteral& x) { [](){ try { throw 0; } catch(int) {}; return 0;  }; return 0; }
  static int f20(NonLiteral& x) { [](){ asm ("mov %rax, %rax");  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f20' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f20(NonLiteral& x) { [](){ asm ("mov %rax, %rax");  }; return 0; }
  static int f21(NonLiteral& x) { [](){ int y; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f21' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f21(NonLiteral& x) { [](){ int y; return 0;  }; return 0; }
  // CHECK-MESSAGES-20-CLT: :[[@LINE-3]]:14: warning: function 'f21' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-20-CLT: static constexpr int f21(NonLiteral& x) { [](){ int y; return 0;  }; return 0; }
  static int f22(NonLiteral& x) { [](){ static int y = 0; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f22' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f22(NonLiteral& x) { [](){ static int y = 0; return 0;  }; return 0; }
  static int f23(NonLiteral& x) { [](){ thread_local int y = 0; return 0;  }; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f23' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f23(NonLiteral& x) { [](){ thread_local int y = 0; return 0;  }; return 0; }

  static int f24(NonLiteral& x) { return [](){ return 0; }(); }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f24' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f24(NonLiteral& x) { return [](){ return 0; }(); }
  // CHECK-MESSAGES-17-CLT: :[[@LINE-3]]:14: warning: function 'f24' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-17-CLT: static constexpr int f24(NonLiteral& x) { return [](){ return 0; }(); }

  static int f25(NonLiteral& x) { new int; return 0; }
  // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'f25' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f25(NonLiteral& x) { new int; return 0; }
  // CHECK-MESSAGES-20-CLT: :[[@LINE-3]]:14: warning: function 'f25' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-20-CLT: static constexpr int f25(NonLiteral& x) { new int; return 0; }

  struct Range0To10 {
    struct iterator {
      int operator*() const { return i; }
      void operator++() { ++i; }
      friend bool operator!=(const iterator&lhs, const iterator&rhs) { return lhs.i == rhs.i; }
      int i;
    };
    iterator begin() const { return { 0 }; }
    iterator end() const { return { 10 }; }
  };
  static int f26(NonLiteral& x) {
    auto R = Range0To10{};
    for (const int i: R) { }
    return 0;
  }
  // CHECK-MESSAGES-23: :[[@LINE-5]]:14: warning: function 'f26' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-23: static constexpr int f26(NonLiteral& x) {
  // CHECK-MESSAGES-20-CLT: :[[@LINE-7]]:14: warning: function 'f26' can be declared 'constexpr' [modernize-use-constexpr]
  // CHECK-FIXES-20-CLT: static constexpr int f26(NonLiteral& x) {

  template <typename> void f27() {
    [](int N) { N; };
  }

} // namespace function_non_literal_ref

template <typename T>
static T forwardDeclared();

template <typename T>
static T forwardDeclared() { return T{}; }
// CHECK-MESSAGES-11: :[[@LINE-1]]:10: warning: function 'forwardDeclared' can be declared 'constexpr' [modernize-use-constexpr]
// CHECK-FIXES-11: template <typename T>
// CHECK-FIXES-11: static constexpr T forwardDeclared();
// CHECK-FIXES-11: template <typename T>
// CHECK-FIXES-11: static constexpr T forwardDeclared() { return T{}; }

static void useForwardDeclared() {
// CHECK-MESSAGES-23: :[[@LINE-1]]:13: warning: function 'useForwardDeclared' can be declared 'constexpr' [modernize-use-constexpr]
// CHECK-FIXES-23: static constexpr void useForwardDeclared() {
    forwardDeclared<int>() + forwardDeclared<double>() + forwardDeclared<char>();
}

namespace {
namespace variable {
    namespace literal_type {
        constexpr int f1() { return 0; }
        int g1() { return 0; }
        // CHECK-MESSAGES-11: :[[@LINE-1]]:13: warning: function 'g1' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: constexpr int g1() { return 0; }
        static constexpr int A1 = 0;
        static int B1 = 0;
        static const int C1 = 0;
        // CHECK-MESSAGES-11: :[[@LINE-1]]:26: warning: variable 'C1' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: static constexpr int C1 = 0;
        static const int D1 = f1();
        // CHECK-MESSAGES-11: :[[@LINE-1]]:26: warning: variable 'D1' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: static constexpr int D1 = f1();
        static const int E1 = g1();

        template <typename T>
        const T TemplatedVar1 = T{};
        // CHECK-MESSAGES-11: :[[@LINE-1]]:17: warning: variable 'TemplatedVar1' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: constexpr T TemplatedVar1 = T{};

        void h1() {
        // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'h1' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr void h1() {
            int a1 = 0;
            const int b1 = 1;
            // CHECK-MESSAGES-11: :[[@LINE-1]]:23: warning: variable 'b1' can be declared 'constexpr' [modernize-use-constexpr]
            // CHECK-FIXES-11: constexpr int b1 = 1;
            static int c1 = 2;
            static const int d1 = 3;

            static auto e1 = TemplatedVar1<int> + TemplatedVar1<unsigned int>;

            const auto check = [](const int & ref) { };
            // CHECK-MESSAGES-17: :[[@LINE-1]]:24: warning: variable 'check' can be declared 'constexpr' [modernize-use-constexpr]
            // CHECK-FIXES-17: constexpr  auto check = [](const int & ref) { };
            
            [[maybe_unused]] const int f1 = 4;
            // CHECK-MESSAGES-11: :[[@LINE-1]]:40: warning: variable 'f1' can be declared 'constexpr' [modernize-use-constexpr]
            // CHECK-FIXES-11: {{\[\[maybe_unused\]\]}} constexpr int f1 = 4;
        }
    } // namespace literal_type

    namespace non_literal_type {
        void unreferencedVolatile() {
        // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'unreferencedVolatile' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr void unreferencedVolatile() {
            const volatile int x = 0;
        }
        void referencedVolatile() {
        // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'referencedVolatile' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr void referencedVolatile() {
            const volatile int x = 0;
            int y = x;
        }
    } // namespace non_literal_type

    namespace struct_type {
        struct AStruct { int val; };
        constexpr AStruct f2() { return {}; }
        AStruct g2() { return {}; }
        // CHECK-MESSAGES-11: :[[@LINE-1]]:17: warning: function 'g2' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: constexpr AStruct g2() { return {}; }
        static constexpr AStruct A2 = {};
        static AStruct B2 = {};
        static const AStruct C2 = {};
        // CHECK-MESSAGES-11: :[[@LINE-1]]:30: warning: variable 'C2' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: static constexpr AStruct C2 = {};
        static const AStruct D2 = f2();
        // CHECK-MESSAGES-11: :[[@LINE-1]]:30: warning: variable 'D2' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: static constexpr AStruct D2 = f2();
        static const AStruct E2 = g2();
        void h2() {
        // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'h2' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr void h2() {
            AStruct a2{};
            const AStruct b2{};
            // CHECK-MESSAGES-11: :[[@LINE-1]]:27: warning: variable 'b2' can be declared 'constexpr' [modernize-use-constexpr]
            // CHECK-FIXES-11: constexpr AStruct b2{};
            static AStruct c2{};
            static const AStruct d2{};
        }
    } // namespace struct_type

    namespace struct_type_non_literal {
        struct AStruct { ~AStruct(); int val; };
        AStruct g3() { return {}; }
        // CHECK-MESSAGES-23: :[[@LINE-1]]:17: warning: function 'g3' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr AStruct g3() { return {}; }
        static AStruct B3 = {};
        static const AStruct C3 = {};
        static const AStruct E3 = g3();

        template <typename T>
        const T TemplatedVar2 = T{};
        template <typename T>
        const T TemplatedVar2B = T{};
        // CHECK-MESSAGES-11: :[[@LINE-1]]:17: warning: variable 'TemplatedVar2B' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: constexpr T TemplatedVar2B = T{};

        void h3() {
        // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'h3' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr void h3() {
            AStruct a3{};
            const AStruct b3{};
            static AStruct c3{};
            static const AStruct d3{};

            static auto e1 = TemplatedVar2<AStruct>;
            static auto f1 = TemplatedVar2B<AStruct>;
            static auto g1 = TemplatedVar2B<int>;
        }
    } // namespace struct_type_non_literal

    namespace struct_type_non_literal2 {
        struct AStruct { volatile int Val; };
        AStruct g4() { return {}; }
        // CHECK-MESSAGES-23: :[[@LINE-1]]:17: warning: function 'g4' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr AStruct g4() { return {}; }
        static AStruct B4 = {};
        static const AStruct C4 = {};
        static const AStruct E4 = g4();
        void h4() {
        // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'h4' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr void h4() {
            AStruct a4{};
            const AStruct b4{};
            static AStruct c4{};
            static const AStruct d4{};
        }
    } // namespace struct_type_non_literal2

    namespace struct_type_non_literal3 {
        struct AStruct { union { int val; float val5; }; };
        constexpr AStruct f5() { return {}; }
        AStruct g5() { return {}; }
        // CHECK-MESSAGES-11: :[[@LINE-1]]:17: warning: function 'g5' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: constexpr AStruct g5() { return {}; }
        static constexpr AStruct A5 = {};
        static AStruct B5 = {};
        static const AStruct C5 = {};
        // CHECK-MESSAGES-11: :[[@LINE-1]]:30: warning: variable 'C5' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: static constexpr AStruct C5 = {};
        static const AStruct D5 = f5();
        // CHECK-MESSAGES-11: :[[@LINE-1]]:30: warning: variable 'D5' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-11: static constexpr AStruct D5 = f5();
        static const AStruct E5 = g5();
        void h5() {
        // CHECK-MESSAGES-23: :[[@LINE-1]]:14: warning: function 'h5' can be declared 'constexpr' [modernize-use-constexpr]
        // CHECK-FIXES-23: constexpr void h5() {
            AStruct a5{};
            const AStruct b5{};
            // CHECK-MESSAGES-11: :[[@LINE-1]]:27: warning: variable 'b5' can be declared 'constexpr' [modernize-use-constexpr]
            // CHECK-FIXES-11: constexpr AStruct b5{};
            static AStruct c5{};
            static const AStruct d5{};
        }
    } // namespace struct_type_non_literal3
} // namespace variable
} // namespace

