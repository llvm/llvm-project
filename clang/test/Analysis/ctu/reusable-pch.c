// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// DEFINE: %{ctu_analysis} =  %clang_analyze_cc1 \
// DEFINE:                        -analyzer-checker=core \
// DEFINE:                        -analyzer-config experimental-enable-naive-ctu-analysis=true \
// DEFINE:                        -analyzer-config ctu-dir=%t \
// DEFINE:                        -verify

// Step 1: Build PCH and defmap.
// RUN: %clang_cc1 -x c -emit-pch -fvalidate-ast-input-files-content -o %t/other.c.ast %t/other.c
// RUN: %clang_extdef_map %t/other.c -- -c -x c > %t/externalDefMap.tmp.txt
// RUN: sed -e 's| .*other\.c| other.c.ast|' %t/externalDefMap.tmp.txt > %t/externalDefMap.txt

// Step 2a: Run CTU using the PCH - the division by zero is found via inlining.
// RUN: %{ctu_analysis} %t/main.c

// Step 2b: Run with content validation - no difference.
// RUN: %{ctu_analysis} %t/main.c -fvalidate-ast-input-files-content

// Step 3: Advance mtime of the source from which PCH was built.
// RUN: %python -c "import os, sys, time; os.utime(sys.argv[1], (time.time() + 120, time.time() + 120))" %t/other.c

// Step 4a: Run CTU using the "stale" PCH, and it should still load it and find the division by zero bug.
// RUN: %{ctu_analysis} -fvalidate-ast-input-files-content %t/main.c

// Step 4b: Run without content validation: CTU import failure
// RUN: not %{ctu_analysis} %t/main.c 2>&1 | FileCheck %s

//--- main.c
// Without CTU, always_zero() has an unknown return value so no bug is found.
// With CTU, always_zero() is inlined and its return value (0) is known,
// exposing the division by zero.

// CHECK: fatal error: file '{{.*}}other.c' has been modified since the precompiled file '{{.*}}other.c.ast' was built
// CHECK: note: mtime changed from expected
// CHECK: note: earlier input file validation has covered only user files
// CHECK: import of an external symbol for CTU failed: Failed to load external AST source.

int always_zero(void);

void f(void) {
  int x = always_zero();
  (void)(1 / x); // expected-warning{{Division by zero}}
}

//--- other.c
int always_zero(void) { return 0; }
