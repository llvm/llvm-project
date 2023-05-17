// RUN: %check_clang_tidy -expect-clang-tidy-error %s altera-struct-pack-align %t -- -header-filter=.*

struct A;
struct B {
  A a;
// CHECK-MESSAGES: :[[@LINE-1]]:5: error: field has incomplete type 'A' [clang-diagnostic-error]
};
