// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -o - %s | FileCheck --check-prefix=DEFAULT %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-kcfi-hash=xxHash64 -o - %s | FileCheck --check-prefix=XXHASH %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fsanitize-kcfi-hash=FNV-1a -o - %s | FileCheck --check-prefix=FNV %s

void foo(void) {}

// DEFAULT: ![[#]] = !{i32 4, !"kcfi-hash", !"xxHash64"}
// XXHASH: ![[#]] = !{i32 4, !"kcfi-hash", !"xxHash64"}
// FNV: ![[#]] = !{i32 4, !"kcfi-hash", !"FNV-1a"}
