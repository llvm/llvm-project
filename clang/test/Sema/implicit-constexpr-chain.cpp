// RUN: %clang_cc1 -verify=BEFORE -std=c++23 %s
// RUN: %clang_cc1 -verify=AFTER -std=c++23 %s -fimplicit-constexpr 

// AFTER-no-diagnostics

// FOLLOWING TWO EXAMPLES ESTABLISH THE `-fimplicit-constexpr` allows enter to constant evaluation for functions
// which would be disabled based on:
// [expr.const]#10.3 "an invocation of a non-constexpr function" (is not allowed)

// -------------------

inline int normal_function() {
  // BEFORE-note@-1 {{declared here}}
  return 42;
}

constinit auto cinit = normal_function();
// BEFORE-error@-1 {{variable does not have a constant initializer}}
// BEFORE-note@-2 {{required by 'constinit' specifier here}}
// BEFORE-note@-3 {{non-constexpr function 'normal_function' cannot be used in a constant expression}}

// -------------------

inline int still_normal_function() {
  // BEFORE-note@-1 {{declared here}}
  return 42;
}

constexpr auto cxpr = still_normal_function();
// BEFORE-error@-1 {{constexpr variable 'cxpr' must be initialized by a constant expression}}
// BEFORE-note@-2 {{non-constexpr function 'still_normal_function' cannot be used in a constant expression}}

// -------------------

// Following example shows calling non-constexpr marked function in 
// constant evaluated context is no longer an error.

struct type_with_nonconstexpr_static_function {
  static /* non-constexpr */ int square(int v) {
    // BEFORE-note@-1 {{declared here}}
    return v*v;
  }
  
  int value;
  constexpr type_with_nonconstexpr_static_function(int x): value{square(x)} { }
  // BEFORE-note@-1 {{non-constexpr function 'square' cannot be used in a constant expression}} 
  //                                                                      ^ (Hana's note: during evaluation)
};

constexpr auto force_ce = type_with_nonconstexpr_static_function{4};
// BEFORE-error@-1 {{constexpr variable 'force_ce' must be initialized by a constant expression}}
// BEFORE-note@-2 {{in call to 'type_with_nonconstexpr_static_function(4)'}}


// this is fine: as it's in runtime, where the constructor 
// is called in runtime, like a normal function, so it doesn't matter `square` is not constexpr
static auto static_var = type_with_nonconstexpr_static_function{4};



// -------------------

// Following example shows now you can call non-constexpr marked even 
// from consteval function initiated constant evaluation.

inline int runtime_only_function() {
  // BEFORE-note@-1 {{declared here}}
  return 11;
}

constexpr int constexpr_function() {
  return runtime_only_function();
  // BEFORE-note@-1 {{non-constexpr function 'runtime_only_function' cannot be used in a constant expression}}
}

consteval int consteval_function() {
  return constexpr_function();
  // BEFORE-note@-1 {{in call to 'constexpr_function()'}}
}

static int noncalled_runtime_function() {
  // we enter consteval context here to replace `consteval_function()` with a constant.
  // this is happen during parsing!!
  return consteval_function(); 
  // BEFORE-error@-1 {{call to consteval function 'consteval_function' is not a constant expression}}
  // BEFORE-note@-2 {{in call to 'consteval_function()'}}
}

