// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-reinterpret-cast %t

int i = 0;
void *j;
void f() { j = reinterpret_cast<void *>(i); }
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use reinterpret_cast [cppcoreguidelines-pro-type-reinterpret-cast]
