// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: cp %s %t/test.c
// RUN: echo "// dependency" > %t/header.h
// RUN: %python -c "import os; ts = os.stat(r'%t/test.c').st_mtime + 60; os.utime(r'%t/header.h', (ts, ts))"
// RUN: %clang_cc1 -fsyntax-only -verify %t/test.c

// The message contains tokens injected by a nested pragma. Skip the annotation
// while preserving surrounding ordinary tokens and the dependency warning.
#pragma GCC dependency "header.h" before _Pragma("weak foobar") after // expected-warning {{current file is older than dependency before foobar after}}

int after_pragmas;
int use_after_pragmas(void) { return after_pragmas; }
