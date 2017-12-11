// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -std=c++11 -ast-dump %s | FileCheck --check-prefix=DUMP %s
// RUN: not %clang_cc1 -std=c++11 -ast-print %s | FileCheck --check-prefix=PRINT %s

int f1() __attribute__((no_sanitize)); // expected-error{{'no_sanitize' attribute takes at least 1 argument}}

int f2() __attribute__((no_sanitize(1))); // expected-error{{'no_sanitize' attribute requires a string}}

// DUMP-LABEL: FunctionDecl {{.*}} f3
// DUMP: NoSanitizeAttr {{.*}} address
// PRINT: int f3() __attribute__((no_sanitize("address")))
int f3() __attribute__((no_sanitize("address")));

// DUMP-LABEL: FunctionDecl {{.*}} f4
// DUMP: NoSanitizeAttr {{.*}} thread
// PRINT: int f4() {{\[\[}}clang::no_sanitize("thread")]]
[[clang::no_sanitize("thread")]] int f4();

// DUMP-LABEL: FunctionDecl {{.*}} f4
// DUMP: NoSanitizeAttr {{.*}} hwaddress
// PRINT: int f4() {{\[\[}}clang::no_sanitize("hwaddress")]]
[[clang::no_sanitize("hwaddress")]] int f4();

// DUMP-LABEL: FunctionDecl {{.*}} f5
// DUMP: NoSanitizeAttr {{.*}} address thread hwaddress
// PRINT: int f5() __attribute__((no_sanitize("address", "thread", "hwaddress")))
int f5() __attribute__((no_sanitize("address", "thread", "hwaddress")));

// DUMP-LABEL: FunctionDecl {{.*}} f6
// DUMP: NoSanitizeAttr {{.*}} unknown
// PRINT: int f6() __attribute__((no_sanitize("unknown")))
int f6() __attribute__((no_sanitize("unknown"))); // expected-warning{{unknown sanitizer 'unknown' ignored}}
