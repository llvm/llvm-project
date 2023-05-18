// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// FIXME: test with non-std qualifiers

namespace move {
  struct Const {
    Const(const Const&&) = default; // expected-error {{the parameter for an explicitly-defaulted move constructor may not be const}}
    Const& operator=(const Const&&) = default; // expected-error {{the parameter for an explicitly-defaulted move assignment operator may not be const}}
  };

  struct Volatile {
    Volatile(volatile Volatile&&) = default; // expected-error {{the parameter for an explicitly-defaulted move constructor may not be volatile}}
    Volatile& operator=(volatile Volatile&&) = default; // expected-error {{the parameter for an explicitly-defaulted move assignment operator may not be volatile}}
  };

  struct AssignmentRet1 {
    AssignmentRet1&& operator=(AssignmentRet1&&) = default; // expected-error {{explicitly-defaulted move assignment operator must return 'AssignmentRet1 &'}}
  };

  struct AssignmentRet2 {
    const AssignmentRet2& operator=(AssignmentRet2&&) = default; // expected-error {{explicitly-defaulted move assignment operator must return 'AssignmentRet2 &'}}
  };

  struct ConstAssignment {
    ConstAssignment& operator=(ConstAssignment&&) const = default; // expected-error {{an explicitly-defaulted move assignment operator may not have 'const', 'constexpr' or 'volatile' qualifiers}}
  };
}

namespace copy {
  struct Volatile {
    Volatile(const volatile Volatile&) = default; // expected-error {{the parameter for an explicitly-defaulted copy constructor may not be volatile}}
    Volatile& operator=(const volatile Volatile&) = default; // expected-error {{the parameter for an explicitly-defaulted copy assignment operator may not be volatile}}
  };

  struct Const {
    Const(const Const&) = default;
    Const& operator=(const Const&) = default;
  };

  struct NonConst {
    NonConst(NonConst&) = default;
    NonConst& operator=(NonConst&) = default;
  };

  struct NonConst2 {
    NonConst2(NonConst2&);
    NonConst2& operator=(NonConst2&);
  };
  NonConst2::NonConst2(NonConst2&) = default;
  NonConst2 &NonConst2::operator=(NonConst2&) = default;

  struct NonConst3 {
    NonConst3(NonConst3&) = default;
    NonConst3& operator=(NonConst3&) = default;
    NonConst nc;
  };

  struct BadConst {
    BadConst(const BadConst&) = default; // expected-error {{is const, but}}
    BadConst& operator=(const BadConst&) = default; // expected-error {{is const, but}}
    NonConst nc; // makes implicit copy non-const
  };

  struct AssignmentRet1 {
    AssignmentRet1&& operator=(const AssignmentRet1&) = default; // expected-error {{explicitly-defaulted copy assignment operator must return 'AssignmentRet1 &'}}
  };

  struct AssignmentRet2 {
    const AssignmentRet2& operator=(const AssignmentRet2&) = default; // expected-error {{explicitly-defaulted copy assignment operator must return 'AssignmentRet2 &'}}
  };

  struct ConstAssignment {
    ConstAssignment& operator=(const ConstAssignment&) const = default; // expected-error {{an explicitly-defaulted copy assignment operator may not have 'const', 'constexpr' or 'volatile' qualifiers}}
  };
}
