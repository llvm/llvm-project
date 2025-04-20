// RUN: %clang_cc1 -verify=NORMAL14,BOTH14,ALLNORMAL,ALL -std=c++14 %s -fcolor-diagnostics
// RUN: %clang_cc1 -verify=IMPLICIT14,BOTH14,ALLIMPLICIT,ALL -fimplicit-constexpr -std=c++14 %s -fcolor-diagnostics

// RUN: %clang_cc1 -verify=NORMAL17,BOTH20,ALLNORMAL,ALL -std=c++17 %s -fcolor-diagnostics
// RUN: %clang_cc1 -verify=IMPLICIT17,BOTH20,ALLIMPLICIT,ALL -fimplicit-constexpr -std=c++17 %s -fcolor-diagnostics

// RUN: %clang_cc1 -verify=NORMAL20,BOTH20,ALLNORMAL,ALL -std=c++20 %s -fcolor-diagnostics
// RUN: %clang_cc1 -verify=IMPLICIT20,BOTH20,ALLIMPLICIT,ALL -fimplicit-constexpr -std=c++20 %s -fcolor-diagnostics

// RUN: %clang_cc1 -verify=NORMAL23,BOTH23,ALLNORMAL,ALL -std=c++23 %s -fcolor-diagnostics
// RUN: %clang_cc1 -verify=IMPLICIT23,BOTH23,ALLIMPLICIT,ALL -fimplicit-constexpr -std=c++23 %s -fcolor-diagnostics

// ALLIMPLICIT-no-diagnostics

// =============================================
// 1) simple member function

struct simple_type {
  bool test() const {
    // ALLNORMAL-note@-1 {{declared here}}
    return true;
  }
};

constexpr bool result_simple_type = simple_type{}.test();
// ALLNORMAL-error@-1 {{constexpr variable 'result_simple_type' must be initialized by a constant expression}}
// ALLNORMAL-note@-2 {{non-constexpr function 'test' cannot be used in a constant expression}}

#ifdef __cpp_implicit_constexpr
static_assert(result_simple_type == true, "simple member function must work");
#endif


// =============================================
// 2) simple member function inside a template

template <typename T> struct template_type {
  bool test() const {
    // ALLNORMAL-note@-1 {{declared here}}
    return true;
  }
};

constexpr bool result_template_type = template_type<int>{}.test();
// ALLNORMAL-error@-1 {{constexpr variable 'result_template_type' must be initialized by a constant expression}}
// ALLNORMAL-note@-2 {{non-constexpr function 'test' cannot be used in a constant expression}}


// =============================================
// 3) template member function inside a template

template <typename T> struct template_template_type {
  template <typename Y> bool test() const {
    // ALLNORMAL-note@-1 {{declared here}}
    return true;
  }
};

constexpr bool result_template_template_type = template_template_type<int>{}.template test<long>();
// ALLNORMAL-error@-1 {{constexpr variable 'result_template_template_type' must be initialized by a constant expression}}
// ALLNORMAL-note@-2 {{non-constexpr function 'test<long>' cannot be used in a constant expression}}


#if defined(__cpp_explicit_this_parameter) && __cpp_explicit_this_parameter >= 202110L

// =============================================
// 3) explicit "this" function

struct explicit_this {
  template <typename Self> bool test(this const Self & self) {
    // ALLNORMAL-note@-1 {{declared here}}
    return self.ok;
  }
};

struct child: explicit_this {
  static constexpr bool ok = true;
};

constexpr bool result_explicit_this = child{}.test();
// ALLNORMAL-error@-1 {{constexpr variable 'result_explicit_this' must be initialized by a constant expression}}
// ALLNORMAL-note@-2 {{non-constexpr function 'test<child>' cannot be used in a constant expression}}

#endif
