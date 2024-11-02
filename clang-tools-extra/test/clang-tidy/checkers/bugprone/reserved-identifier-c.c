// RUN: %check_clang_tidy %s bugprone-reserved-identifier %t

// in C, double underscores are fine except at the beginning

void foo__(void);
void f__o__o(void);
void f_________oo(void);
void __foo(void);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier '__foo', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}void foo(void);{{$}}
