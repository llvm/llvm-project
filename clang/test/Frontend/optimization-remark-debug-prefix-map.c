/// Backend diagnostics are mapped back to the source through the debug info,
/// whose file names have -fdebug-prefix-map applied.

// RUN: rm -rf %t && split-file --no-leading-lines %s %t
// RUN: mkdir %t/decoy && cp %t/src/t.c %t/decoy/t.c

// RUN: %clang_cc1 %t/src/t.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-prefix-map=%t/src=%{fs-src-root}elsewhere
// RUN: %clang_cc1 %t/src/t.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-prefix-map=%t/src=.

/// A file outside the compilation maps to the same name.
// RUN: %clang_cc1 %t/src/t.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-prefix-map=%t/src=. -fdebug-prefix-map=%t/decoy=.

/// A file outside the compilation exists under the remapped name.
// RUN: %clang_cc1 %t/src/t.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-compilation-dir=%t -fdebug-prefix-map=%t/src=%t/decoy

/// Without checksums, two files in the compilation that map to the same name
/// share a DIFile, which then does not identify the file; fall back to the
/// location of the function.
// RUN: %clang_cc1 %t/src/both.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify -I%t \
// RUN:   -dwarf-version=4 \
// RUN:   -fdebug-prefix-map=%t/a=%{fs-src-root}M -fdebug-prefix-map=%t/b=%{fs-src-root}M

/// Line directives make debug locations use presumed lines, which do not map
/// back to the file, even if a DIFile for the file was created before the
/// directive was seen.
// RUN: %clang_cc1 %t/src/line.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-prefix-map=%t/src=%{fs-src-root}elsewhere

/// The same holds if the file with line directives shares a DIFile with
/// another file.
// RUN: %clang_cc1 %t/src/line-shared.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify -I%t \
// RUN:   -dwarf-version=4 \
// RUN:   -fdebug-prefix-map=%t/c=%{fs-src-root}M -fdebug-prefix-map=%t/b=%{fs-src-root}M

//--- src/t.c
int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }

int bar(int j) {
  // expected-remark@+1 {{'foo' inlined into 'bar'}}
  return foo(j, j - 2);
}

//--- src/both.c
static inline __attribute__((always_inline)) int foo(int x, int y) {
  return x + y;
}
#include "a/h.h"
#include "b/h.h"

//--- a/h.h
// expected-remark@+2 {{'foo' inlined into 'callA'}}
// expected-note@+1 {{could not determine the original source location}}
int callA(int j) {
  return foo(j, j - 2);
}

//--- b/h.h
// Maps to the same name in the debug info as a/h.h.

int callB(int j) {
  return j + (j - 2);
}

//--- src/line.c
int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }
int before_line_directive;
#line 100
// expected-remark@+2 {{'foo' inlined into 'bar'}}
// expected-note@+1 {{could not determine the original source location}}
int bar(int j) {
  return foo(j, j - 2);
}

//--- src/line-shared.c
static inline __attribute__((always_inline)) int foo(int x, int y) {
  return x + y;
}
#include "c/h.h"
#include "b/h.h"

//--- c/h.h
#line 1
// expected-remark@+2 {{'foo' inlined into 'callC'}}
// expected-note@+1 {{could not determine the original source location}}
int callC(int j) {
  return foo(j, j - 2);
}
