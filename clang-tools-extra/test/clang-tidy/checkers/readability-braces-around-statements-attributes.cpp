// RUN: %check_clang_tidy  -std=c++20-or-later %s readability-braces-around-statements %t

void test(bool b) {
  if (b) {
    return;
  }
  if (b) [[likely]] {
    // CHECK-FIXES-NOT: if (b) { {{[[][[]}}likely{{[]][]]}} {
    return;
  }
  if (b) [[unlikely]] {
    // CHECK-FIXES-NOT: if (b) { {{[[][[]}}unlikely{{[]][]]}} {
    return;
  }

  if (b) [[likely]]
    // CHECK-FIXES: if (b) {{[[][[]}}likely{{[]][]]}} {
    return;
  // CHECK-FIXES: }
  if (b) [[unlikely]]
    // CHECK-FIXES: if (b) {{[[][[]}}unlikely{{[]][]]}} {
    return;
  // CHECK-FIXES: }
}
