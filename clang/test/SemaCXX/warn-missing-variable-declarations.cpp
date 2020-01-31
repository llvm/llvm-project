// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-variable-declarations -std=c++17 %s

// Variable declarations that should trigger a warning.
int vbad1; // expected-warning{{no previous extern declaration for non-static variable 'vbad1'}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}

int vbad2 = 10; // expected-warning{{no previous extern declaration for non-static variable 'vbad2'}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}

namespace x {
  int vbad3; // expected-warning{{no previous extern declaration for non-static variable 'vbad3'}}
  // expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}
}

// Variable declarations that should not trigger a warning.
static int vgood1;
extern int vgood2;
int vgood2;
static struct {
  int mgood1;
} vgood3;

// Functions should never trigger a warning.
void fgood1(void);
void fgood2(void) {
  int lgood1;
  static int lgood2;
}
static void fgood3(void) {
  int lgood3;
  static int lgood4;
}

// Structures, namespaces and classes should be unaffected.
struct sgood1 {
  int mgood2;
};
struct {
  int mgood3;
} sgood2;
class CGood1 {
  static int MGood1;
};
int CGood1::MGood1;
namespace {
  int mgood4;
}

class C {
  void test() {
    static int x = 0; // no-warn
  }
};

// There is also no need to use static in anonymous namespaces.
namespace {
  int vgood4;
}

inline int inline_var = 0;
const int const_var = 0;
constexpr int constexpr_var = 0;
inline constexpr int inline_constexpr_var = 0;
extern const int extern_const_var = 0; // expected-warning {{no previous extern declaration}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}
extern constexpr int extern_constexpr_var = 0; // expected-warning {{no previous extern declaration}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}

template<typename> int var_template = 0;
template<typename> constexpr int const_var_template = 0;
template<typename> static int static_var_template = 0;

template<typename T> int var_template<T*>;

template int var_template<int[1]>;
int use_var_template() { return var_template<int[2]>; }
template int var_template<int[3]>;
extern template int var_template<int[4]>;
template<> int var_template<int[5]>; // expected-warning {{no previous extern declaration}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}

// FIXME: We give this specialization internal linkage rather than inheriting
// the linkage from the template! We should not warn here.
template<> int static_var_template<int[5]>; // expected-warning {{no previous extern declaration}}
// expected-note@-1{{declare 'static' if the variable is not intended to be used outside of this translation unit}}
