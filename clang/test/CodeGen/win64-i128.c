// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=GNU64
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=MSC64

typedef int int128_t __attribute__((mode(TI)));

int128_t foo() { return 0; }

// GNU64: define dso_local <2 x i64> @foo()
// MSC64: define dso_local <2 x i64> @foo()

int128_t bar(int128_t a, int128_t b) { return a * b; }

// GNU64: define dso_local <2 x i64> @bar(i128*, i128*)
// MSC64: define dso_local <2 x i64> @bar(i128*, i128*)
