// RUN: %check_clang_tidy %s readability-non-const-parameter %t

static int f();

int f(p)
  int *p;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: pointer parameter 'p' can be pointer to const [readability-non-const-parameter]
// CHECK-FIXES: {{^}}  const int *p;{{$}}
{
    return *p;
}
