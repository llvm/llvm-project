// RUN: %check_clang_tidy -std=c++11-or-later %s cppcoreguidelines-use-enum-class %t -- \
// RUN:   -config="{CheckOptions: {cppcoreguidelines-use-enum-class.IgnoreUnscopedEnumsInClasses: true}}" --

enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use 'enum class' instead

enum class EC {};

struct S {
  enum E {};
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead
  // Ignore unscoped enums in recordDecl
  enum class EC {};
};

enum ForwardE : int;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'ForwardE' is unscoped, use 'enum class' instead
enum class ForwardEC : int;
