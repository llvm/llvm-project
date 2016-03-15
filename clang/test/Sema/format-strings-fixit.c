// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -Wall -fixit %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wall -Werror %t
// RUN: %clang_cc1 -E -o - %t | FileCheck %s

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

int printf(char const *, ...);

typedef __SIZE_TYPE__ size_t;
typedef __INTMAX_TYPE__ intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __WCHAR_TYPE__ wchar_t;

void test() {
  // Basic types
  printf("%s", (int) 123);
  printf("abc%0f", "testing testing 123");
  printf("%u", (long) -12);
  printf("%p", 123);
  printf("%c\n", "x");
  printf("%c\n", 1.23);

  // Larger types
  printf("%+.2d", (unsigned long long) 123456);
  printf("%1d", (long double) 1.23);

  // Flag handling
  printf("%0+s", (unsigned) 31337); // 0 flag should stay
  printf("%#p", (void *) 0);
  printf("% +f", 1.23); // + flag should stay
  printf("%0-f", 1.23); // - flag should stay

  // Positional arguments
#pragma clang diagnostic push // Don't warn about using positional arguments.
#pragma clang diagnostic ignored "-Wformat-non-iso"
  printf("%1$f:%2$.*3$f:%4$.*3$f\n", 1, 2, 3, 4);
#pragma clang diagnostic pop

  // Precision
  printf("%10.5d", 1l); // (bug 7394)
  printf("%.2c", 'a');

  // Ignored flags
  printf("%0-f", 1.23);

  // Bad length modifiers
  printf("%hhs", "foo");
#pragma clang diagnostic push // Don't warn about using positional arguments.
#pragma clang diagnostic ignored "-Wformat-non-iso"
  printf("%1$zp", (void *)0);
#pragma clang diagnostic pop

  // Preserve the original formatting for unsigned integers.
  unsigned long val = 42;
  printf("%X", val);

  // size_t, etc.
  printf("%f", (size_t) 42);
  printf("%f", (intmax_t) 42);
  printf("%f", (uintmax_t) 42);
  printf("%f", (ptrdiff_t) 42);

  // Look beyond the first typedef.
  typedef size_t my_size_type;
  typedef intmax_t my_intmax_type;
  typedef uintmax_t my_uintmax_type;
  typedef ptrdiff_t my_ptrdiff_type;
  typedef int my_int_type;
  printf("%f", (my_size_type) 42);
  printf("%f", (my_intmax_type) 42);
  printf("%f", (my_uintmax_type) 42);
  printf("%f", (my_ptrdiff_type) 42);
  printf("%f", (my_int_type) 42);

  // string
  printf("%ld", "foo");

  // Preserve the original choice of conversion specifier.
  printf("%o", (long) 42);
  printf("%u", (long) 42);
  printf("%x", (long) 42);
  printf("%X", (long) 42);
  printf("%i", (unsigned long) 42);
  printf("%d", (unsigned long) 42);
  printf("%F", (long double) 42);
  printf("%e", (long double) 42);
  printf("%E", (long double) 42);
  printf("%g", (long double) 42);
  printf("%G", (long double) 42);
  printf("%a", (long double) 42);
  printf("%A", (long double) 42);
}

int scanf(char const *, ...);

void test2(int intSAParm[static 2]) {
  char str[100];
  char *vstr = "abc";
  short shortVar;
  unsigned short uShortVar;
  int intVar;
  int intAVar[2];
  unsigned uIntVar;
  float floatVar;
  double doubleVar;
  long double longDoubleVar;
  long longVar;
  unsigned long uLongVar;
  long long longLongVar;
  unsigned long long uLongLongVar;
  size_t sizeVar;
  intmax_t intmaxVar;
  uintmax_t uIntmaxVar;
  ptrdiff_t ptrdiffVar;
  enum {A, B, C} enumVar;

  // Some string types.
  scanf("%lf", str);
  scanf("%lf", vstr);
  scanf("%ls", str);
  scanf("%ls", str);

  // Some integer types.
  scanf("%f", &shortVar);
  scanf("%f", &uShortVar);
  scanf("%p", &intVar);
  scanf("%f", intAVar);
  scanf("%f", intSAParm);
  scanf("%Lf", &uIntVar);
  scanf("%ld", &floatVar);
  scanf("%f", &doubleVar);
  scanf("%d", &longDoubleVar);
  scanf("%f", &longVar);
  scanf("%f", &uLongVar);
  scanf("%f", &longLongVar);
  scanf("%f", &uLongLongVar);
  scanf("%d", &enumVar); // FIXME: We ought to fix specifiers for enums.

  // Some named ints.
  scanf("%f", &sizeVar);
  scanf("%f", &intmaxVar);
  scanf("%f", &uIntmaxVar);
  scanf("%f", &ptrdiffVar);

  // Look beyond the first typedef for named integer types.
  typedef size_t my_size_type;
  typedef intmax_t my_intmax_type;
  typedef uintmax_t my_uintmax_type;
  typedef ptrdiff_t my_ptrdiff_type;
  typedef int my_int_type;
  scanf("%f", (my_size_type*)&sizeVar);
  scanf("%f", (my_intmax_type*)&intmaxVar);
  scanf("%f", (my_uintmax_type*)&uIntmaxVar);
  scanf("%f", (my_ptrdiff_type*)&ptrdiffVar);
  scanf("%f", (my_int_type*)&intVar);

  // Preserve the original formatting.
  scanf("%o", &longVar);
  scanf("%u", &longVar);
  scanf("%x", &longVar);
  scanf("%X", &longVar);
  scanf("%i", &uLongVar);
  scanf("%d", &uLongVar);
  scanf("%F", &longDoubleVar);
  scanf("%e", &longDoubleVar);
  scanf("%E", &longDoubleVar);
  scanf("%g", &longDoubleVar);
  scanf("%G", &longDoubleVar);
  scanf("%a", &longDoubleVar);
  scanf("%A", &longDoubleVar);
}

// Validate the fixes.
// CHECK: printf("%d", (int) 123);
// CHECK: printf("abc%s", "testing testing 123");
// CHECK: printf("%ld", (long) -12);
// CHECK: printf("%d", 123);
// CHECK: printf("%s\n", "x");
// CHECK: printf("%f\n", 1.23);
// CHECK: printf("%+.2lld", (unsigned long long) 123456);
// CHECK: printf("%1Lf", (long double) 1.23);
// CHECK: printf("%0u", (unsigned) 31337);
// CHECK: printf("%p", (void *) 0);
// CHECK: printf("%+f", 1.23);
// CHECK: printf("%-f", 1.23);
// CHECK: printf("%1$d:%2$.*3$d:%4$.*3$d\n", 1, 2, 3, 4);
// CHECK: printf("%10.5ld", 1l);
// CHECK: printf("%c", 'a');
// CHECK: printf("%-f", 1.23);
// CHECK: printf("%s", "foo");
// CHECK: printf("%1$p", (void *)0);
// CHECK: printf("%lX", val);
// CHECK: printf("%zu", (size_t) 42);
// CHECK: printf("%jd", (intmax_t) 42);
// CHECK: printf("%ju", (uintmax_t) 42);
// CHECK: printf("%td", (ptrdiff_t) 42);
// CHECK: printf("%zu", (my_size_type) 42);
// CHECK: printf("%jd", (my_intmax_type) 42);
// CHECK: printf("%ju", (my_uintmax_type) 42);
// CHECK: printf("%td", (my_ptrdiff_type) 42);
// CHECK: printf("%d", (my_int_type) 42);
// CHECK: printf("%s", "foo");
// CHECK: printf("%lo", (long) 42);
// CHECK: printf("%ld", (long) 42);
// CHECK: printf("%lx", (long) 42);
// CHECK: printf("%lX", (long) 42);
// CHECK: printf("%lu", (unsigned long) 42);
// CHECK: printf("%lu", (unsigned long) 42);
// CHECK: printf("%LF", (long double) 42);
// CHECK: printf("%Le", (long double) 42);
// CHECK: printf("%LE", (long double) 42);
// CHECK: printf("%Lg", (long double) 42);
// CHECK: printf("%LG", (long double) 42);
// CHECK: printf("%La", (long double) 42);
// CHECK: printf("%LA", (long double) 42);

// CHECK: scanf("%99s", str);
// CHECK: scanf("%s", vstr);
// CHECK: scanf("%99s", str);
// CHECK: scanf("%99s", str);
// CHECK: scanf("%hd", &shortVar);
// CHECK: scanf("%hu", &uShortVar);
// CHECK: scanf("%d", &intVar);
// CHECK: scanf("%d", intAVar);
// CHECK: scanf("%d", intSAParm);
// CHECK: scanf("%u", &uIntVar);
// CHECK: scanf("%f", &floatVar);
// CHECK: scanf("%lf", &doubleVar);
// CHECK: scanf("%Lf", &longDoubleVar);
// CHECK: scanf("%ld", &longVar);
// CHECK: scanf("%lu", &uLongVar);
// CHECK: scanf("%lld", &longLongVar);
// CHECK: scanf("%llu", &uLongLongVar);
// CHECK: scanf("%d", &enumVar);
// CHECK: scanf("%zu", &sizeVar);
// CHECK: scanf("%jd", &intmaxVar);
// CHECK: scanf("%ju", &uIntmaxVar);
// CHECK: scanf("%td", &ptrdiffVar);
// CHECK: scanf("%zu", (my_size_type*)&sizeVar);
// CHECK: scanf("%jd", (my_intmax_type*)&intmaxVar);
// CHECK: scanf("%ju", (my_uintmax_type*)&uIntmaxVar);
// CHECK: scanf("%td", (my_ptrdiff_type*)&ptrdiffVar);
// CHECK: scanf("%d", (my_int_type*)&intVar);
// CHECK: scanf("%lo", &longVar);
// CHECK: scanf("%lu", &longVar);
// CHECK: scanf("%lx", &longVar);
// CHECK: scanf("%lX", &longVar);
// CHECK: scanf("%li", &uLongVar);
// CHECK: scanf("%ld", &uLongVar);
// CHECK: scanf("%LF", &longDoubleVar);
// CHECK: scanf("%Le", &longDoubleVar);
// CHECK: scanf("%LE", &longDoubleVar);
// CHECK: scanf("%Lg", &longDoubleVar);
// CHECK: scanf("%LG", &longDoubleVar);
// CHECK: scanf("%La", &longDoubleVar);
// CHECK: scanf("%LA", &longDoubleVar);
