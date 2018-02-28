// RUN: %check_clang_tidy %s bugprone-macro-repeated-side-effects %t

#define badA(x,y)  ((x)+((x)+(y))+(y))
void bad(int ret, int a, int b) {
  ret = badA(a++, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: side effects in the 1st macro argument 'x' are repeated in macro expansion [bugprone-macro-repeated-side-effects]
  ret = badA(++a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: side effects in the 1st macro argument 'x'
  ret = badA(a--, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: side effects in the 1st macro argument 'x'
  ret = badA(--a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: side effects in the 1st macro argument 'x'
  ret = badA(a, b++);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: side effects in the 2nd macro argument 'y'
  ret = badA(a, ++b);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: side effects in the 2nd macro argument 'y'
  ret = badA(a, b--);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: side effects in the 2nd macro argument 'y'
  ret = badA(a, --b);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: side effects in the 2nd macro argument 'y'
}


#define MIN(A,B)     ((A) < (B) ? (A) : (B))                        // single ?:
#define LIMIT(X,A,B) ((X) < (A) ? (A) : ((X) > (B) ? (B) : (X)))    // two ?:
void question(int x) {
  MIN(x++, 12);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: side effects in the 1st macro argument 'A'
  MIN(34, x++);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: side effects in the 2nd macro argument 'B'
  LIMIT(x++, 0, 100);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: side effects in the 1st macro argument 'X'
  LIMIT(20, x++, 100);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: side effects in the 2nd macro argument 'A'
  LIMIT(20, 0, x++);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: side effects in the 3rd macro argument 'B'
}

// False positive: Repeated side effects is intentional.
// It is hard to know when it's done by intention so right now we warn.
#define UNROLL(A)    {A A}
void fp1(int i) {
  UNROLL({ i++; });
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: side effects in the 1st macro argument 'A'
}

// Do not produce a false positive on a strchr() macro. Explanation; Currently the '?'
// triggers the test to bail out, because it cannot evaluate __builtin_constant_p(c).
#  define strchrs(s, c) \
  (__extension__ (__builtin_constant_p (c) && !__builtin_constant_p (s)	      \
		  && (c) == '\0'					      \
		  ? (char *) __rawmemchr (s, c)				      \
		  : __builtin_strchr (s, c)))
char* __rawmemchr(char* a, char b) {
  return a;
}
void pass(char* pstr, char ch) {
  strchrs(pstr, ch++); // No error.
}

// Check large arguments (t=20, u=21).
#define largeA(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, x, y, z) \
  ((a) + (a) + (b) + (b) + (c) + (c) + (d) + (d) + (e) + (e) + (f) + (f) + (g) + (g) +    \
   (h) + (h) + (i) + (i) + (j) + (j) + (k) + (k) + (l) + (l) + (m) + (m) + (n) + (n) +    \
   (o) + (o) + (p) + (p) + (q) + (q) + (r) + (r) + (s) + (s) + (t) + (t) + (u) + (u) +    \
   (v) + (v) + (x) + (x) + (y) + (y) + (z) + (z))
void large(int a) {
  largeA(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a++, 0, 0, 0, 0, 0, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:64: warning: side effects in the 19th macro argument 's'
  largeA(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a++, 0, 0, 0, 0, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:67: warning: side effects in the 20th macro argument 't'
  largeA(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a++, 0, 0, 0, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:70: warning: side effects in the 21st macro argument 'u'
}

// Passing macro argument as argument to __builtin_constant_p and macros.
#define builtinbad(x)      (__builtin_constant_p(x) + (x) + (x))
#define builtingood1(x)    (__builtin_constant_p(x) + (x))
#define builtingood2(x)    ((__builtin_constant_p(x) && (x)) || (x))
#define macrobad(x)        (builtingood1(x) + (x) + (x))
#define macrogood(x)       (builtingood1(x) + (x))
void builtins(int ret, int a) {
  ret += builtinbad(a++);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: side effects in the 1st macro argument 'x'

  ret += builtingood1(a++);
  ret += builtingood2(a++);

  ret += macrobad(a++);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: side effects in the 1st macro argument 'x'

  ret += macrogood(a++);
}

// Bail out for conditionals.
#define condB(x,y)  if(x) {x=y;} else {x=y + 1;}
void conditionals(int a, int b)
{
  condB(a, b++);
}

void log(const char *s, int v);
#define LOG(val) log(#val, (val))
void test_log(int a) {
  LOG(a++);
}
