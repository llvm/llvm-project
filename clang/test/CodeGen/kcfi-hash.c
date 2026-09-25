// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -o - %s | FileCheck --check-prefix=DEFAULT %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-kcfi-hash=xxHash64 -o - %s | FileCheck --check-prefix=XXHASH %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-kcfi-hash=FNV-1a -o - %s | FileCheck --check-prefix=FNV %s

// Invalid and empty values are rejected.
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-hash=bogus %s 2>&1 | FileCheck --check-prefix=BAD %s
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=kcfi -fsanitize-kcfi-hash= %s 2>&1 | FileCheck --check-prefix=EMPTY %s
// BAD: error: invalid value 'bogus' in '-fsanitize-kcfi-hash=bogus'
// EMPTY: error: invalid value '' in '-fsanitize-kcfi-hash='

void foo(void) {}

// DEFAULT: ![[#]] = !{i32 4, !"kcfi-hash", !"xxHash64"}
// XXHASH: ![[#]] = !{i32 4, !"kcfi-hash", !"xxHash64"}
// FNV: ![[#]] = !{i32 4, !"kcfi-hash", !"FNV-1a"}
