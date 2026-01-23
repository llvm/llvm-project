// RUN: %check_clang_tidy %s bugprone-random-generator-seed %t -- \
// RUN: -config="{CheckOptions: {bugprone-random-generator-seed.DisallowedSeedTypes: 'some_type,time_t'}}"

void srand(int seed);
typedef int time_t;
time_t time(time_t *t);

void f(void) {
  srand(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a constant value will generate a predictable sequence of values [bugprone-random-generator-seed]

  const int a = 1;
  srand(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a constant value will generate a predictable sequence of values [bugprone-random-generator-seed]

  time_t t;
  srand(time(&t)); // Disallowed seed type
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [bugprone-random-generator-seed]
}

void g(void) {
  typedef int user_t;
  user_t a = 1;
  srand(a);

  int b = 1;
  srand(b); // Can not evaluate as int
}
