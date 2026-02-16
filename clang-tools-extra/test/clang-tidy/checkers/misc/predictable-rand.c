// RUN: %check_clang_tidy %s misc-predictable-rand %t

extern int rand(void);
int nonrand(void);

int cTest(void) {
  int i = rand();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: rand() has limited randomness [misc-predictable-rand]

  int k = nonrand();

  return 0;
}
