// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-enum-class %t

enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use enum class instead [modernize-use-enum-class]

enum class EC {};

struct S {
    enum E {};
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:12: warning: enum 'E' is unscoped, use enum class instead [modernize-use-enum-class]
    // Ignore unscoped enums in recordDecl
    enum class EC {};
};

class C {
    enum E {};
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:12: warning: enum 'E' is unscoped, use enum class instead [modernize-use-enum-class]
    // Ignore unscoped enums in recordDecl
    enum class EC {};
};

template<class T>
class TC {
    enum E {};
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:12: warning: enum 'E' is unscoped, use enum class instead [modernize-use-enum-class]
    // Ignore unscoped enums in recordDecl
    enum class EC {};
};

union U {
    enum E {};
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:12: warning: enum 'E' is unscoped, use enum class instead [modernize-use-enum-class]
    // Ignore unscoped enums in recordDecl
    enum class EC {};
};

namespace {
enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use enum class instead [modernize-use-enum-class]
enum class EC {};
} // namespace

namespace N {
enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' is unscoped, use enum class instead [modernize-use-enum-class]
enum class EC {};
} // namespace N

template<enum ::EC>
static void foo();

using enum S::E;
using enum S::EC;

enum ForwardE : int;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'ForwardE' is unscoped, use enum class instead [modernize-use-enum-class]

enum class ForwardEC : int;
