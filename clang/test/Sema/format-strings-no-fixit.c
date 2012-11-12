// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -fixit %t
// RUN: %clang_cc1 -E -o - %t | FileCheck %s

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. Only
   warnings for format strings within the function call will be
   fixed by -fixit.  Other format strings will be left alone. */

int printf(char const *, ...);
int scanf(char const *, ...);

void pr9751() {
  const char kFormat1[] = "%s";
  printf(kFormat1, 5);
  printf("%s", 5);

  const char kFormat2[] = "%.3p";
  void *p;
  printf(kFormat2, p);
  printf("%.3p", p);

  const char kFormat3[] = "%0s";
  printf(kFormat3, "a");
  printf("%0s", "a");

  const char kFormat4[] = "%hhs";
  printf(kFormat4, "a");
  printf("%hhs", "a");

  const char kFormat5[] = "%-0d";
  printf(kFormat5, 5);
  printf("%-0d", 5);

  const char kFormat6[] = "%00d";
  int *i;
  scanf(kFormat6, i);
  scanf("%00d", i);
}

// CHECK:  const char kFormat1[] = "%s";
// CHECK:  printf(kFormat1, 5);
// CHECK:  printf("%d", 5);

// CHECK:  const char kFormat2[] = "%.3p";
// CHECK:  void *p;
// CHECK:  printf(kFormat2, p);
// CHECK:  printf("%p", p);

// CHECK:  const char kFormat3[] = "%0s";
// CHECK:  printf(kFormat3, "a");
// CHECK:  printf("%s", "a");

// CHECK:  const char kFormat4[] = "%hhs";
// CHECK:  printf(kFormat4, "a");
// CHECK:  printf("%s", "a");

// CHECK:  const char kFormat5[] = "%-0d";
// CHECK:  printf(kFormat5, 5);
// CHECK:  printf("%-d", 5);

// CHECK:  const char kFormat6[] = "%00d";
// CHECK:  int *i;
// CHECK:  scanf(kFormat6, i);
// CHECK:  scanf("%d", i);
