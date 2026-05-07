// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// Step 1: Build PCH and defmap.
// RUN: %clang_cc1 -x c -emit-pch -fvalidate-ast-input-files-content -o %t/other.c.ast %t/other.c
// RUN: %clang_extdef_map %t/other.c -- -c -x c > %t/externalDefMap.tmp.txt
// RUN: sed 's| .*other\.c| other.c.ast|' %t/externalDefMap.tmp.txt > %t/externalDefMap.txt

// Step 2: Run CTU using the PCH - the division by zero is found via inlining.
// RUN: %clang_cc1 -analyze \
// RUN:   -fvalidate-ast-input-files-content \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %t/main.c

// Step 3: Advance mtime of the source from which PCH was built.
// RUN: %python -c "import os, sys, time; os.utime(sys.argv[1], (time.time() + 120, time.time() + 120))" %t/other.c

// Step 4: Run CTU using the "stale" PCH
// RUN: %clang_cc1 -analyze \
// RUN:   -fvalidate-ast-input-files-content \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %t/main.c

//--- main.c
// Without CTU, always_zero() has an unknown return value so no bug is found.
// With CTU, always_zero() is inlined and its return value (0) is known,
// exposing the division by zero.

int always_zero(void);

void f(void) {
  int x = always_zero();
  (void)(1 / x); // expected-warning{{Division by zero}}
}

//--- other.c
int always_zero(void) { return 0; }
