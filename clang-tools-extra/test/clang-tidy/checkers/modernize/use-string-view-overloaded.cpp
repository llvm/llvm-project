// RUN: %check_clang_tidy -check-suffix=DONTCHECK \
// RUN: -std=c++20-or-later %s modernize-use-string-view %t -- \
// RUN: --config="{CheckOptions: {modernize-use-string-view.CheckOverloadedFunctions: false}}"

// RUN: %check_clang_tidy -check-suffix=CHECK \
// RUN: -std=c++20-or-later %s modernize-use-string-view %t -- \
// RUN: --config="{CheckOptions: {modernize-use-string-view.CheckOverloadedFunctions: true}}"
#include <string>
#include <utility>

namespace overload_funcs_redeclared {
std::basic_string<char> overload(int);
std::string overload(int);
std::string overload(int) { return "int"; }
// CHECK-MESSAGES-DONTCHECK:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-MESSAGES-CHECK:[[@LINE-2]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-DONTCHECK: std::string_view overload(int) { return "int"; }
// CHECK-FIXES-CHECK: std::string_view overload(int) { return "int"; }
}

namespace overload_non_func {
struct overload {};
std::string overload(int) { return "int"; }
// CHECK-MESSAGES-DONTCHECK:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-MESSAGES-CHECK:[[@LINE-2]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-DONTCHECK: std::string_view overload(int) { return "int"; }
// CHECK-FIXES-CHECK: std::string_view overload(int) { return "int"; }
}

namespace overload_with_inline {
  inline namespace inline_namespace {
    std::string overload1(int) { return "int"; }
// CHECK-MESSAGES-DONTCHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-MESSAGES-CHECK:[[@LINE-2]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-DONTCHECK: std::string_view overload1(int) { return "int"; }
// CHECK-FIXES-CHECK: std::string_view overload1(int) { return "int"; }
  }
  inline namespace regular_namespace {
    std::string overload1(int) { return "int"; }
// CHECK-MESSAGES-DONTCHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-MESSAGES-CHECK:[[@LINE-2]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-DONTCHECK: std::string_view overload1(int) { return "int"; }
// CHECK-FIXES-CHECK: std::string_view overload1(int) { return "int"; }
  }
}

////////////////////////////////////////////////////////////////////////////////

namespace overload_with_outer {
namespace overload_with_templates {
    template <typename T>
    std::string overload(T) { return "T"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
    std::string overload(std::string) { return "string"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(std::string) { return "string"; }
}
using overload_with_templates::overload;
std::string overload(char*) { return "char*"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(char*) { return "char*"; }
}

namespace overload_funcs {
std::string dbl2str(double f);
std::string overload(int) { return "int"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(int) { return "int"; }
std::string overload(double f) { return "f=" + dbl2str(f); }
std::string overload(std::string) { return "string"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(std::string) { return "string"; }
}

namespace overload_methods {
struct Foo {
  // Skip overloaded methods
  std::string overload(int) { return "int"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(int) { return "int"; }
  std::string overload(double f) { return "double"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(double f) { return "double"; }
  std::string overload(std::string) { return "string"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(std::string) { return "string"; }
};
}

namespace overload_methods_nested_classes {
struct Bar {
  std::string overload(int) { return "int"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(int) { return "int"; }
  std::string overload(std::string) { return "string"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(std::string) { return "string"; }
  struct FooBar {
    std::string overload(char*) { return "char*"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(char*) { return "char*"; }
    std::string overload(double f) { return "double"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(double f) { return "double"; }
  };
};
}

namespace overload_methods_nested_namespaces {
namespace foo {
  std::string overload(int) { return "int"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(int) { return "int"; }
  std::string overload(std::string) { return "string"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(std::string) { return "string"; }
}
using foo::overload;
std::string overload(char*) { return "char*"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(char*) { return "char*"; }
}

namespace overload_methods_templated {
    template <typename T>
    std::string overload(T value) { return "T";}
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(T value) { return "T";}
    std::string overload(int value) { return "int"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(int value) { return "int"; }
}

namespace two_overloads_with_inline {
  inline namespace inline_namespace {
    std::string overload(int) { return "int"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(int) { return "int"; }
    std::string overload(double) { return "double"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:5: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(double) { return "double"; }
  }
  std::string overload(int) { return "int"; }
// CHECK-MESSAGES-CHECK:[[@LINE-1]]:3: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES-CHECK: std::string_view overload(int) { return "int"; }
}