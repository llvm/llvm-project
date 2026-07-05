// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fno-modules-error-recovery -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fno-modules-error-recovery -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11 -DIMPORT_DECLS

// expected-no-diagnostics

#ifdef IMPORT_DECLS
@import redecl_add_after_load_decls;
#else
typedef struct A B;
extern const int variable;
extern constexpr int function();
constexpr int test(bool b) { return b ? variable : function(); }

namespace N {
  typedef struct A B;
  extern const int variable;
  extern constexpr int function();
}
typedef N::B NB;
constexpr int N_test(bool b) { return b ? N::variable : N::function(); }

@import redecl_add_after_load_top;
typedef C::A CB;
constexpr int C_test(bool b) { return b ? C::variable : C::function(); }

struct D {
  struct A;
  static const int variable;
  static constexpr int function();
};
typedef D::A DB;
constexpr int D_test(bool b) { return b ? D::variable : D::function(); }
#endif

@import redecl_add_after_load;

B tu_struct_test;
constexpr int tu_variable_test = test(true);
constexpr int tu_function_test = test(false);

NB ns_struct_test;
constexpr int ns_variable_test = N_test(true);
constexpr int ns_function_test = N_test(false);

CB struct_struct_test;
constexpr int struct_variable_test = C_test(true);
constexpr int struct_function_test = C_test(false);

DB merged_struct_struct_test;
constexpr int merged_struct_variable_test = D_test(true);
constexpr int merged_struct_function_test = D_test(false);
