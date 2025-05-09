// RUN: %check_clang_tidy -std=c++11-or-later %s cppcoreguidelines-use-enum-class %t -- -config="{CheckOptions: {cppcoreguidelines-use-enum-class.IgnoreUnscopedEnumsInClasses: true}}" --

enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]

enum class EC {};

struct S {
  enum E {};
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]
  // Ignore unscoped enums in recordDecl
  enum class EC {};
};
