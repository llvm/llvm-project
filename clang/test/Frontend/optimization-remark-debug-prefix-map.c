/// Backend diagnostics are mapped back to the source through the file names in
/// the debug info, which have -fdebug-prefix-map applied.

// RUN: rm -rf %t && split-file --no-leading-lines %s %t
// RUN: mkdir %t/decoy && cp %t/src/t.c %t/decoy/t.c

// RUN: %clang_cc1 %t/src/t.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-prefix-map=%t/src=%{fs-src-root}elsewhere
// RUN: %clang_cc1 %t/src/t.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-prefix-map=%t/src=.

/// The inverse mapping is ambiguous; only the file being compiled is a match.
// RUN: %clang_cc1 %t/src/t.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-prefix-map=%t/src=. -fdebug-prefix-map=%t/decoy=.

/// A file outside the compilation exists under the remapped name.
// RUN: %clang_cc1 %t/src/t.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify \
// RUN:   -fdebug-compilation-dir=%t -fdebug-prefix-map=%t/src=%t/decoy

/// Two files in the compilation map to the same name, so the debug info does not
/// identify the file; fall back to the location of the function.
// RUN: %clang_cc1 %t/src/both.c -Rpass=inline -O1 -debug-info-kind=line-tables-only -emit-llvm-only -verify -I%t \
// RUN:   -fdebug-prefix-map=%t/a=%{fs-src-root}M -fdebug-prefix-map=%t/b=%{fs-src-root}M

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
