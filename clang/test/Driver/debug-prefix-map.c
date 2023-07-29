// RUN: %clang -### -fdebug-prefix-map=old %s 2>&1 | FileCheck %s -check-prefix CHECK-DEBUG-INVALID
// RUN: %clang -### -fmacro-prefix-map=old %s 2>&1 | FileCheck %s -check-prefix CHECK-MACRO-INVALID
// RUN: %clang -### -fcoverage-prefix-map=old %s 2>&1 | FileCheck %s -check-prefix CHECK-COVERAGE-INVALID
// RUN: %clang -### -ffile-prefix-map=old %s 2>&1 | FileCheck %s -check-prefix CHECK-FILE-INVALID

// RUN: %clang -### -fdebug-prefix-map=old=new %s 2>&1 | FileCheck %s -check-prefix CHECK-DEBUG-SIMPLE
// RUN: %clang -### -fmacro-prefix-map=old=new %s 2>&1 | FileCheck %s -check-prefix CHECK-MACRO-SIMPLE
// RUN: %clang -### -fcoverage-prefix-map=old=new %s 2>&1 | FileCheck %s -check-prefix CHECK-COVERAGE-SIMPLE
// RUN: %clang -### -ffile-prefix-map=old=new %s 2>&1 | FileCheck %s -check-prefix CHECK-DEBUG-SIMPLE
// RUN: %clang -### -ffile-prefix-map=old=new %s 2>&1 | FileCheck %s -check-prefix CHECK-MACRO-SIMPLE
// RUN: %clang -### -ffile-prefix-map=old=new %s 2>&1 | FileCheck %s -check-prefix CHECK-COVERAGE-SIMPLE

// RUN: %clang -### -fdebug-prefix-map=old=n=ew %s 2>&1 | FileCheck %s -check-prefix CHECK-DEBUG-COMPLEX
// RUN: %clang -### -fmacro-prefix-map=old=n=ew %s 2>&1 | FileCheck %s -check-prefix CHECK-MACRO-COMPLEX
// RUN: %clang -### -fcoverage-prefix-map=old=n=ew %s 2>&1 | FileCheck %s -check-prefix CHECK-COVERAGE-COMPLEX
// RUN: %clang -### -ffile-prefix-map=old=n=ew %s 2>&1 | FileCheck %s -check-prefix CHECK-DEBUG-COMPLEX
// RUN: %clang -### -ffile-prefix-map=old=n=ew %s 2>&1 | FileCheck %s -check-prefix CHECK-MACRO-COMPLEX
// RUN: %clang -### -ffile-prefix-map=old=n=ew %s 2>&1 | FileCheck %s -check-prefix CHECK-COVERAGE-COMPLEX

// RUN: %clang -### -fdebug-prefix-map=old= %s 2>&1 | FileCheck %s -check-prefix CHECK-DEBUG-EMPTY
// RUN: %clang -### -fmacro-prefix-map=old= %s 2>&1 | FileCheck %s -check-prefix CHECK-MACRO-EMPTY
// RUN: %clang -### -fcoverage-prefix-map=old= %s 2>&1 | FileCheck %s -check-prefix CHECK-COVERAGE-EMPTY
// RUN: %clang -### -ffile-prefix-map=old= %s 2>&1 | FileCheck %s -check-prefix CHECK-DEBUG-EMPTY
// RUN: %clang -### -ffile-prefix-map=old= %s 2>&1 | FileCheck %s -check-prefix CHECK-MACRO-EMPTY
// RUN: %clang -### -ffile-prefix-map=old= %s 2>&1 | FileCheck %s -check-prefix CHECK-COVERAGE-EMPTY

// CHECK-DEBUG-INVALID: error: invalid argument 'old' to -fdebug-prefix-map
// CHECK-MACRO-INVALID: error: invalid argument 'old' to -fmacro-prefix-map
// CHECK-COVERAGE-INVALID: error: invalid argument 'old' to -fcoverage-prefix-map
// CHECK-FILE-INVALID: error: invalid argument 'old' to -ffile-prefix-map
// CHECK-DEBUG-SIMPLE: fdebug-prefix-map=old=new
// CHECK-MACRO-SIMPLE: fmacro-prefix-map=old=new
// CHECK-COVERAGE-SIMPLE: fcoverage-prefix-map=old=new
// CHECK-DEBUG-COMPLEX: fdebug-prefix-map=old=n=ew
// CHECK-MACRO-COMPLEX: fmacro-prefix-map=old=n=ew
// CHECK-COVERAGE-COMPLEX: fcoverage-prefix-map=old=n=ew
// CHECK-DEBUG-EMPTY: fdebug-prefix-map=old=
// CHECK-MACRO-EMPTY: fmacro-prefix-map=old=
// CHECK-COVERAGE-EMPTY: fcoverage-prefix-map=old=
