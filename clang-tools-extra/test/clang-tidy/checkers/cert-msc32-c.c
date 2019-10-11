// RUN: %check_clang_tidy %s cert-msc32-c %t -- -config="{CheckOptions: [{key: cert-msc32-c.DisallowedSeedTypes, value: 'some_type,time_t'}]}" -- -std=c99

void srand(int seed);
typedef int time_t;
time_t time(time_t *t);

void f() {
  srand(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc32-c]

  const int a = 1;
  srand(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc32-c]

  time_t t;
  srand(time(&t)); // Disallowed seed type
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc32-c]
}

void g() {
  typedef int user_t;
  user_t a = 1;
  srand(a);

  int b = 1;
  srand(b); // Can not evaluate as int
}
