// RUN: %check_clang_tidy %s bugprone-custom-errno-declaration %t

extern int errno;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: errno declaration detected, include cerrno instead [bugprone-custom-errno-declaration]
// CHECK-FIXES: {{^}}{{$}}
