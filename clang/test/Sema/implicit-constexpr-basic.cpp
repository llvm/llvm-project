// RUN: %clang_cc1 -verify=ALL_NORMAL,NORMAL14,BOTH14,ALL_PRE20,ALLNORMAL,NORMAL_PRE20,ALL -std=c++14 %s -fcolor-diagnostics
// RUN: %clang_cc1 -verify=IMPLICIT14,IMPLICIT_PRE20,BOTH14,ALL_PRE20,ALLIMPLICIT,ALL -fimplicit-constexpr -std=c++14 %s -fcolor-diagnostics

// RUN: %clang_cc1 -verify=ALL_NORMAL,NORMAL17,BOTH17,ALL_PRE20,ALLNORMAL,NORMAL_PRE20,ALL -std=c++17 %s -fcolor-diagnostics
// RUN: %clang_cc1 -verify=IMPLICIT17,IMPLICIT_PRE20,BOTH17,ALL_PRE20,ALLIMPLICIT,ALL -fimplicit-constexpr -std=c++17 %s -fcolor-diagnostics

// RUN: %clang_cc1 -verify=ALL_NORMAL,NORMAL20,BOTH20,ALLNORMAL,ALL -std=c++20 %s -fcolor-diagnostics
// RUN: %clang_cc1 -verify=IMPLICIT20,BOTH20,ALLIMPLICIT,ALL -fimplicit-constexpr -std=c++20 %s -fcolor-diagnostics

// RUN: %clang_cc1 -verify=ALL_NORMAL,NORMAL23,BOTH23,ALLNORMAL,ALL -std=c++23 %s -fcolor-diagnostics
// RUN: %clang_cc1 -verify=IMPLICIT23,BOTH23,ALLIMPLICIT,ALL -fimplicit-constexpr -std=c++23 %s -fcolor-diagnostics




// =============================================
// 1) simple uninlined function

bool noinline_fnc() { 
// ALL-note@-1 {{declared here}}
	return true;
}

constexpr bool result_noinline_fnc = noinline_fnc();
// ALL-error@-1 {{constexpr variable 'result_noinline_fnc' must be initialized by a constant expression}}
// ALLNORMAL-note@-2 {{non-constexpr function 'noinline_fnc' cannot be used in a constant expression}}
// ALLIMPLICIT-note@-3 {{non-inline function 'noinline_fnc' is not implicitly constexpr}}


// =============================================
// 2) simple inlined function

inline bool inline_fnc() { 
// ALLNORMAL-note@-1 {{declared here}}
	return true;
}

constexpr bool result_inline_fnc = inline_fnc();
// ALLNORMAL-error@-1 {{constexpr variable 'result_inline_fnc' must be initialized by a constant expression}}
// ALLNORMAL-note@-2 {{non-constexpr function 'inline_fnc' cannot be used in a constant expression}}


// =============================================
// 3) undefined uninlined function

bool noinline_undefined_fnc();
// ALL-note@-1 {{declared here}}

constexpr bool result_noinline_undefined_fnc = noinline_undefined_fnc();
// ALL-error@-1 {{constexpr variable 'result_noinline_undefined_fnc' must be initialized by a constant expression}}
// ALLNORMAL-note@-2 {{non-constexpr function 'noinline_undefined_fnc' cannot be used in a constant expression}}
// ALLIMPLICIT-note@-3 {{undefined function 'noinline_undefined_fnc' cannot be used in a constant expression}}


// =============================================
// 4) undefined inline function

inline bool inline_undefined_fnc();
// ALL-note@-1 {{declared here}}

constexpr bool result_inline_undefined_fnc = inline_undefined_fnc();
// ALL-error@-1 {{constexpr variable 'result_inline_undefined_fnc' must be initialized by a constant expression}}
// ALLNORMAL-note@-2 {{non-constexpr function 'inline_undefined_fnc' cannot be used in a constant expression}}
// ALLIMPLICIT-note@-3 {{undefined function 'inline_undefined_fnc' cannot be used in a constant expression}}

// =============================================
// 5) lambda function

auto lambda = [](int x) { return x > 0; };
// NORMAL14-note@-1 {{declared here}}

constexpr bool result_lambda = lambda(10);
// NORMAL14-error@-1 {{constexpr variable 'result_lambda' must be initialized by a constant expression}}
// NORMAL14-note@-2 {{non-constexpr function 'operator()' cannot be used in a constant expression}}


// =============================================
// 6) virtual functions

struct type {
  virtual bool dispatch() const noexcept {
    return false;
  }
};

struct child_of_type: type {
  bool dispatch() const noexcept override {
// NORMAL20-note@-1 {{declared here}}
// NORMAL23-note@-2 {{declared here}}
    return true;
  }
};

constexpr bool result_virtual = static_cast<const type &>(child_of_type{}).dispatch();
// ALL_NORMAL-error@-1 {{constexpr variable 'result_virtual' must be initialized by a constant expression}}
// NORMAL_PRE20-note@-2 {{cannot evaluate call to virtual function in a constant expression in C++ standards before C++20}}
// IMPLICIT_PRE20-error@-3 {{constexpr variable 'result_virtual' must be initialized by a constant expression}}
// IMPLICIT_PRE20-note@-4 {{cannot evaluate call to virtual function in a constant expression in C++ standards before C++20}}
// NORMAL20-note@-5 {{non-constexpr function 'dispatch' cannot be used in a constant expression}}
// NORMAL20-note@-6 {{declared here}}
// NORMAL23-note@-7 {{non-constexpr function 'dispatch' cannot be used in a constant expression}}
// NORMAL23-note@-8 {{declared here}}


#if defined(__cpp_constexpr) && __cpp_constexpr >= 201907L
static_assert(result_virtual == true, "virtual should work");
// ALL_NORMAL-error@-1 {{static assertion expression is not an integral constant expression}}
// ALL_NORMAL-note@-2 {{initializer of 'result_virtual' is not a constant expression}}
// IMPLICIT_PRE20-note@-3 {{initializer of 'result_virtual' is not a constant expression}}
#endif


