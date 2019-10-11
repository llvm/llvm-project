// RUN: %check_clang_tidy %s cert-msc30-c %t

extern int rand(void);
int nonrand();

int cTest() {
  int i = rand();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: rand() has limited randomness [cert-msc30-c]

  int k = nonrand();

  return 0;
}
