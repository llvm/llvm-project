// RUN: %check_clang_tidy -std=c17-or-earlier -check-suffixes=C17 %s readability-redundant-zero-initializer %t
// RUN: %check_clang_tidy -std=c23-or-later -check-suffixes=C23 %s readability-redundant-zero-initializer %t

char a[12] = {0};
// CHECK-MESSAGES-C23: :[[@LINE-1]]:14: warning: redundant zero initializer; replace with empty braces [readability-redundant-zero-initializer]
// CHECK-FIXES-C23: char a[12] = {};

int b[5] = {0};
// CHECK-MESSAGES-C23: :[[@LINE-1]]:12: warning: redundant zero initializer; replace with empty braces
// CHECK-FIXES-C23: int b[5] = {};

char deduced[] = {0};
int multiZero[3] = {0, 0};
int mixed[4] = {0, 5};
int oneByOne[1][1] = {0};
char nullChar[4] = {'\0'};
