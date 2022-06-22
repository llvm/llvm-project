// RUN: %check_clang_tidy %s readability-magic-numbers %t --
// XFAIL: *

int ProcessSomething(int input);

int DoWork()
{
  if (((int)4) > ProcessSomething(10))
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 4 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
    return 0;

   return 0;
}


