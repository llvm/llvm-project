// RUN: %check_clang_tidy %s readability-inconsistent-declaration-parameter-name %t -- \
// RUN:   -config="{CheckOptions: {readability-inconsistent-declaration-parameter-name.IgnoreMacros: false}}"

#define MACRO() \
  void f(int x)

struct S1 {
  MACRO();
  // CHECK-NOTES: :[[@LINE-1]]:3: warning: function 'S1::f' has a definition with different parameter names
  // CHECK-NOTES: :[[@LINE-5]]:8: note: expanded from macro 'MACRO'
  // CHECK-NOTES: :[[@LINE+4]]:10: note: the definition seen here
  // CHECK-NOTES: :[[@LINE-4]]:3: note: differing parameters are named here: ('x'), in definition: ('y')
  // CHECK-NOTES: :[[@LINE-8]]:8: note: expanded from macro 'MACRO'
};
void S1::f(int y) {}

struct S2 {
  int g() const;
  void set_g(int g);
  // CHECK-NOTES: :[[@LINE-1]]:8: warning: function 'S2::set_g' has a definition with different parameter names
  // CHECK-NOTES: :[[@LINE+14]]:1: note: the definition seen here
  // CHECK-NOTES: :[[@LINE+9]]:12: note: expanded from macro 'DEFINITION'
  // This one is unfortunate, but the location this points to is in a scratch
  // space, so it's not helpful to the user.
  // CHECK-NOTES: {{^}}note: expanded from here{{$}}
  // CHECK-NOTES: :[[@LINE-7]]:8: note: differing parameters are named here: ('g'), in definition: ('w')
};

#define DEFINITION(name, parameter)    \
  int S2::name() const { return 0; }   \
  void S2::set_##name(int parameter) { \
    (void)parameter;                   \
  }

DEFINITION(g, w)

//////////////////////////////////////////////////////

#define DECLARE_FUNCTION_WITH_PARAM_NAME(function_name, param_name) \
  void function_name(int param_name)

// CHECK-NOTES: :[[@LINE+1]]:34: warning: function 'macroFunction' has 1 other declaration with different parameter names [readability-inconsistent-declaration-parameter-name]
DECLARE_FUNCTION_WITH_PARAM_NAME(macroFunction, a);
// CHECK-NOTES: :[[@LINE+2]]:34: note: the 1st inconsistent declaration seen here
// CHECK-NOTES: :[[@LINE+1]]:34: note: differing parameters are named here: ('b'), in the other declaration: ('a')
DECLARE_FUNCTION_WITH_PARAM_NAME(macroFunction, b);
