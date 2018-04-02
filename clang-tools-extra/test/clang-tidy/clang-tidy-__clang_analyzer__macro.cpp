// RUN: %check_clang_tidy %s * %t

#if defined(__clang_analyzer__)
#warning __clang_analyzer__ is defined
#endif
// CHECK-MESSAGES: :[[@LINE-2]]:2: warning: __clang_analyzer__ is defined [clang-diagnostic-#warnings]


