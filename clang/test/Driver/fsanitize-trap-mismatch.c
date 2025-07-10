// RUN: %clang -S -emit-llvm -g -O0 %s -fsanitize-trap=undefined -o - 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: %clang -S -emit-llvm -g -O0 %s -fsanitize=undefined  -fsanitize-trap=undefined -o - 2>&1 | FileCheck %s --check-prefix=NOWARN
// RUN: %clang -S -emit-llvm -g -O0 %s -fsanitize-trap=undefined -Wno-sanitize-trap-mismatch -o - 2>&1 | FileCheck %s --check-prefix=SUPPRESS
//
// WARN: warning: -fsanitize-trap=undefined has no effect because the "undefined" sanitizer is disabled; consider passing "fsanitize=undefined" to enable the sanitizer
// NOWARN-NOT: warning: -fsanitize-trap=undefined has no effect because the "undefined" sanitizer is disabled; consider passing "fsanitize=undefined" to enable the sanitizer
// SUPPRESS-NOT: warning: -fsanitize-trap=undefined has no effect because the "undefined" sanitizer is disabled; consider passing "fsanitize=undefined" to enable the sanitizer

int main(void) { return 0; }
