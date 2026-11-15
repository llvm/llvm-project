// RUN: %check_clang_tidy %s bugprone-custom-errno-declaration %t -- -- -I %S/Inputs/custom-errno-declaration

#include <stdint.h>

extern int16_t errno;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: errno declaration detected, include cerrno instead [bugprone-custom-errno-declaration]
// CHECK-FIXES: {{^}}{{$}}
