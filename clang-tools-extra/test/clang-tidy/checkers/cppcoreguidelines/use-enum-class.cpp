// RUN: %check_clang_tidy -std=c++11-or-later -check-suffix=ALL,DEFAULT %s \
// RUN: cppcoreguidelines-use-enum-class %t --

// RUN: %check_clang_tidy -std=c++11-or-later -check-suffix=ALL %s \
// RUN: cppcoreguidelines-use-enum-class %t -- \
// RUN: -config="{CheckOptions: { \
// RUN: cppcoreguidelines-use-enum-class.IgnoreUnscopedEnumsInClasses: true \
// RUN: }}" --

enum E {};
// CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use 'enum class' instead

enum class EC {};

enum struct ES {};

struct S {
  enum E {};
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead
  enum class EC {};
};

class C {
  enum E {};
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead
  enum class EC {};
};

template<class T>
class TC {
  enum E {};
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead
  enum class EC {};
};

union U {
  enum E {};
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead
  enum class EC {};
};

namespace {
enum E {};
// CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use 'enum class' instead
enum class EC {};
} // namespace

namespace N {
enum E {};
// CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use 'enum class' instead
enum class EC {};
} // namespace N

template<enum ::EC>
static void foo();

enum ForwardE : int;
// CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: enum 'ForwardE' is unscoped, use 'enum class' instead

enum class ForwardEC : int;

enum struct ForwardES : int;
