// RUN: %check_clang_tidy %s bugprone-custom-errno-declaration %t

extern int errno, preserved;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: errno declaration detected, include cerrno instead [bugprone-custom-errno-declaration]
// CHECK-FIXES: {{^}}{{$}}
