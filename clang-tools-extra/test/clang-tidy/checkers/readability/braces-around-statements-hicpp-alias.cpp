// RUN: %check_clang_tidy %s hicpp-braces-around-statements %t

void f();
bool b();

void test() {
if (b())
  f();
// CHECK-MESSAGES: warning: 'hicpp-braces-around-statements' check is deprecated and will be removed in a future release; consider using 'readability-braces-around-statements' instead [clang-tidy-config]
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: statement should be inside braces [hicpp-braces-around-statements]
// CHECK-FIXES: if (b()) {
// CHECK-FIXES: }
}
