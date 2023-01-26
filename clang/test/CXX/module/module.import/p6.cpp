// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -x c++-header %t/bad-header-unit.h \
// RUN:  -emit-header-unit -o %t/bad-header-unit.pcm -verify

//--- bad-header-unit.h

inline int ok_foo () { return 0;}

static int ok_bar ();

int ok_decl ();

int bad_def () { return 2;}  // expected-error {{non-inline external definitions are not permitted in C++ header units}}

inline int ok_inline_var = 1;

static int ok_static_var;

int ok_var_decl;

int bad_var_definition = 3;  // expected-error {{non-inline external definitions are not permitted in C++ header units}}

class A {
public:
    // This is a declaration instead of definition.
    static const int value = 43; 
};

void deleted_fn_ok (void) = delete;

struct S {
   ~S() noexcept(false) = default;
private:
  S(S&);
};
S::S(S&) = default;
