// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-member-init %t -- -config="{CheckOptions: [{key: "cppcoreguidelines-pro-type-member-init.UseAssignment", value: true}]}" -- -fsigned-char

struct T {
  int i;
};

struct S {
  bool b;
  // CHECK-FIXES: bool b = false;
  char c;
  // CHECK-FIXES: char c = 0;
  signed char sc;
  // CHECK-FIXES: signed char sc = 0;
  unsigned char uc;
  // CHECK-FIXES: unsigned char uc = 0U;
  int i;
  // CHECK-FIXES: int i = 0;
  unsigned u;
  // CHECK-FIXES: unsigned u = 0U;
  long l;
  // CHECK-FIXES: long l = 0L;
  unsigned long ul;
  // CHECK-FIXES: unsigned long ul = 0UL;
  long long ll;
  // CHECK-FIXES: long long ll = 0LL;
  unsigned long long ull;
  // CHECK-FIXES: unsigned long long ull = 0ULL;
  float f;
  // CHECK-FIXES: float f = 0.0F;
  double d;
  // CHECK-FIXES: double d = 0.0;
  long double ld;
  // CHECK-FIXES: double ld = 0.0L;
  int *ptr;
  // CHECK-FIXES: int *ptr = nullptr;
  T t;
  // CHECK-FIXES: T t{};
  S() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields:
};
