// Example from C99 6.10.3.4p9

// RUN: %clang_cc1 -E %s | FileCheck -strict-whitespace %s

#define debug(...) fprintf(stderr, __VA_ARGS__) 
#define showlist(...) puts(#__VA_ARGS__) 
#define report(test, ...) ((test)?puts(#test):\
                           printf(__VA_ARGS__)) 
debug("Flag");
// CHECK: fprintf(stderr, "Flag");

debug("X = %d\n", x);
// CHECK: fprintf(stderr, "X = %d\n", x);

showlist(The first, second, and third items.);
// CHECK: puts("The first, second, and third items.");

report(x>y, "x is %d but y is %d", x, y);
// CHECK: ((x>y)?puts("x>y"): printf("x is %d but y is %d", x, y));

