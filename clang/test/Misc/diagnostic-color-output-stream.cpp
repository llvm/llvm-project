// REQUIRES: ansi-escape-sequences

// Default ('auto'): a redirected (non-terminal) stderr must not get ANSI codes.
// RUN: not %clang_cc1 -fsyntax-only -std=c++11 %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOCOLOR %s

// Forced off behaves the same.
// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -fno-color-diagnostics %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NOCOLOR %s

// Forced on: ANSI codes are emitted even to a non-terminal.
// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -fcolor-diagnostics %s 2>&1 \
// RUN:   | FileCheck --check-prefix=COLOR %s

// A -diagnostic-log-file pointing at a regular file must not contain ANSI.
// RUN: rm -f %t.log
// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -diagnostic-log-file %t.log %s 2>/dev/null
// RUN: FileCheck --check-prefix=NOCOLOR %s < %t.log

template <typename> struct foo {};
void func(foo<int>);
void g() { func(foo<double>()); }

// COLOR: {{.\[0;1;36m}}double{{.\[0m}}
// NOCOLOR-NOT: {{.\[0;1;36m}}
