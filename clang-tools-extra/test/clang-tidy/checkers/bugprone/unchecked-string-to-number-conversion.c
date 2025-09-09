// RUN: %check_clang_tidy %s bugprone-unchecked-string-to-number-conversion %t

typedef __SIZE_TYPE__      size_t;
typedef signed             ptrdiff_t;
typedef long long          intmax_t;
typedef unsigned long long uintmax_t;
typedef void *             FILE;

extern FILE *stdin;

extern int fscanf(FILE *stream, const char *format, ...);
extern int scanf(const char *format, ...);
extern int sscanf(const char *s, const char *format, ...);

extern double atof(const char *nptr);
extern int atoi(const char *nptr);
extern long int atol(const char *nptr);
extern long long int atoll(const char *nptr);

void f1(const char *in) {
  int i;
  long long ll;
  unsigned int ui;
  unsigned long long ull;
  intmax_t im;
  uintmax_t uim;
  float f;
  double d;
  long double ld;

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'sscanf' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtol' instead [bugprone-unchecked-string-to-number-conversion]
  sscanf(in, "%d", &i);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'fscanf' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtoll' instead
  fscanf(stdin, "%lld", &ll);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'sscanf' used to convert a string to an unsigned integer value, but function will not report conversion errors; consider using 'strtoul' instead
  sscanf(in, "%u", &ui);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'fscanf' used to convert a string to an unsigned integer value, but function will not report conversion errors; consider using 'strtoull' instead
  fscanf(stdin, "%llu", &ull);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtoimax' instead
  scanf("%jd", &im);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'fscanf' used to convert a string to an unsigned integer value, but function will not report conversion errors; consider using 'strtoumax' instead
  fscanf(stdin, "%ju", &uim);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'sscanf' used to convert a string to a floating-point value, but function will not report conversion errors; consider using 'strtof' instead
  sscanf(in, "%f", &f); // to float
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'fscanf' used to convert a string to a floating-point value, but function will not report conversion errors; consider using 'strtod' instead
  fscanf(stdin, "%lg", &d);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'sscanf' used to convert a string to a floating-point value, but function will not report conversion errors; consider using 'strtold' instead
  sscanf(in, "%Le", &ld);

  // These are conversions with other modifiers
  short s;
  char c;
  size_t st;
  ptrdiff_t pt;

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%hhd", &c);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%hd", &s);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%zu", &st);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%td", &pt);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%o", ui);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%X", ui);
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%x", ui);
}

void f2(const char *in) {
  // CHECK-MESSAGES: :[[@LINE+1]]:11: warning: 'atoi' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtol' instead
  int i = atoi(in); // to int
  // CHECK-MESSAGES: :[[@LINE+1]]:12: warning: 'atol' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtol' instead
  long l = atol(in); // to long
  // CHECK-MESSAGES: :[[@LINE+1]]:18: warning: 'atoll' used to convert a string to an integer value, but function will not report conversion errors; consider using 'strtoll' instead
  long long ll = atoll(in); // to long long
  // CHECK-MESSAGES: :[[@LINE+1]]:14: warning: 'atof' used to convert a string to a floating-point value, but function will not report conversion errors; consider using 'strtod' instead
  double d = atof(in); // to double
}

void f3(void) {
  int i;
  unsigned int u;
  float f;
  char str[32];

  // Test that we don't report multiple infractions for a single call.
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%d%u%f", &i, &u, &f);

  // Test that we still catch infractions that are not the first specifier.
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: 'scanf' used to convert
  scanf("%s%d", str, &i);
}

void do_not_diagnose(void) {
  char str[32];

  scanf("%s", str); // Not a numerical conversion
  scanf("%*d"); // Assignment suppressed
}
