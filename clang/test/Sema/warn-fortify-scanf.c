// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify

typedef struct _FILE FILE;
extern int scanf(const char *format, ...);
extern int fscanf(FILE *f, const char *format, ...);
extern int sscanf(const char *input, const char *format, ...);

void call_scanf() {
  char buf10[10];
  char buf20[20];
  char buf30[30];
  scanf("%4s %5s %10s", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 4 has size 10, but the corresponding specifier may require size 11}}
  scanf("%4s %5s %11s", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 4 has size 10, but the corresponding specifier may require size 12}}
  scanf("%4s %5s %9s", buf20, buf30, buf10);
  scanf("%20s %5s %9s", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 2 has size 20, but the corresponding specifier may require size 21}}
  scanf("%21s %5s %9s", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 2 has size 20, but the corresponding specifier may require size 22}}
  scanf("%19s %5s %9s", buf20, buf30, buf10);
  scanf("%19s %29s %9s", buf20, buf30, buf10);

  scanf("%*21s %*30s %10s", buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 2 has size 10, but the corresponding specifier may require size 11}}
  scanf("%*21s %5s", buf10);
  scanf("%10s %*30s", buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 2 has size 10, but the corresponding specifier may require size 11}}
  scanf("%9s %*30s", buf10);

  scanf("%4[a] %5[a] %10[a]", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 4 has size 10, but the corresponding specifier may require size 11}}
  scanf("%4[a] %5[a] %11[a]", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 4 has size 10, but the corresponding specifier may require size 12}}
  scanf("%4[a] %5[a] %9[a]", buf20, buf30, buf10);
  scanf("%20[a] %5[a] %9[a]", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 2 has size 20, but the corresponding specifier may require size 21}}
  scanf("%21[a] %5[a] %9[a]", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 2 has size 20, but the corresponding specifier may require size 22}}
  scanf("%19[a] %5[a] %9[a]", buf20, buf30, buf10);
  scanf("%19[a] %29[a] %9[a]", buf20, buf30, buf10);

  scanf("%4c %5c %10c", buf20, buf30, buf10);
  scanf("%4c %5c %11c", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 4 has size 10, but the corresponding specifier may require size 11}}
  scanf("%4c %5c %9c", buf20, buf30, buf10);
  scanf("%20c %5c %9c", buf20, buf30, buf10);
  scanf("%21c %5c %9c", buf20, buf30, buf10); // expected-warning {{'scanf' may overflow; destination buffer in argument 2 has size 20, but the corresponding specifier may require size 21}}

  // Don't warn for other specifiers.
  int x;
  scanf("%12d", &x);
}

void call_sscanf() {
  char buf10[10];
  char buf20[20];
  char buf30[30];
  sscanf("a b c", "%4s %5s %10s", buf20, buf30, buf10); // expected-warning {{'sscanf' may overflow; destination buffer in argument 5 has size 10, but the corresponding specifier may require size 11}}
  sscanf("a b c", "%4s %5s %11s", buf20, buf30, buf10); // expected-warning {{'sscanf' may overflow; destination buffer in argument 5 has size 10, but the corresponding specifier may require size 12}}
  sscanf("a b c", "%4s %5s %9s", buf20, buf30, buf10);
  sscanf("a b c", "%20s %5s %9s", buf20, buf30, buf10); // expected-warning {{'sscanf' may overflow; destination buffer in argument 3 has size 20, but the corresponding specifier may require size 21}}
  sscanf("a b c", "%21s %5s %9s", buf20, buf30, buf10); // expected-warning {{'sscanf' may overflow; destination buffer in argument 3 has size 20, but the corresponding specifier may require size 22}}
  sscanf("a b c", "%19s %5s %9s", buf20, buf30, buf10);
  sscanf("a b c", "%19s %29s %9s", buf20, buf30, buf10);
}

void call_fscanf() {
  char buf10[10];
  char buf20[20];
  char buf30[30];
  fscanf(0, "%4s %5s %10s", buf20, buf30, buf10); // expected-warning {{'fscanf' may overflow; destination buffer in argument 5 has size 10, but the corresponding specifier may require size 11}}
  fscanf(0, "%4s %5s %11s", buf20, buf30, buf10); // expected-warning {{'fscanf' may overflow; destination buffer in argument 5 has size 10, but the corresponding specifier may require size 12}}
  fscanf(0, "%4s %5s %9s", buf20, buf30, buf10);
  fscanf(0, "%20s %5s %9s", buf20, buf30, buf10); // expected-warning {{'fscanf' may overflow; destination buffer in argument 3 has size 20, but the corresponding specifier may require size 21}}
  fscanf(0, "%21s %5s %9s", buf20, buf30, buf10); // expected-warning {{'fscanf' may overflow; destination buffer in argument 3 has size 20, but the corresponding specifier may require size 22}}
  fscanf(0, "%19s %5s %9s", buf20, buf30, buf10);
  fscanf(0, "%19s %29s %9s", buf20, buf30, buf10);
}
