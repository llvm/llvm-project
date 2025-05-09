// RUN: %check_clang_tidy -std=c++11-or-later %s cppcoreguidelines-use-enum-class %t

enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]

enum class EC {};

struct S {
  enum E {};
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]
  enum class EC {};
};

class C {
  enum E {};
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]
  enum class EC {};
};

template<class T>
class TC {
  enum E {};
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]
  enum class EC {};
};

union U {
  enum E {};
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]
  enum class EC {};
};

namespace {
enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]
enum class EC {};
} // namespace

namespace N {
enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]
enum class EC {};
} // namespace N

template<enum ::EC>
static void foo();

enum ForwardE : int;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'ForwardE' is unscoped, use 'enum class' instead [cppcoreguidelines-use-enum-class]
enum class ForwardEC : int;
