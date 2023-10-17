// REQUIRES: shell

// Check that lld gets "-lto_library".
// (Separate test file since darwin-ld-lto requires system-darwin but this
// test doesn't require that.)

// Check that -object_lto_path is passed correctly to ld64
// RUN: %clang -fuse-ld=lld -B%S/Inputs/lld --target=x86_64-apple-darwin10 \
// RUN:     %s -flto=full -### 2>&1 \
// RUN:     | FileCheck -check-prefix=FULL_LTO_OBJECT_PATH %s
// FULL_LTO_OBJECT_PATH: {{ld(.exe)?"}}
// FULL_LTO_OBJECT_PATH-SAME: "-object_path_lto"
// FULL_LTO_OBJECT_PATH-SAME: {{cc\-[a-zA-Z0-9_]+.o}}"
// RUN: %clang -fuse-ld=lld -B%S/Inputs/lld --target=x86_64-apple-darwin10 \
// RUN:     %s -flto=thin -### 2>&1 \
// RUN:     | FileCheck -check-prefix=THIN_LTO_OBJECT_PATH %s
// THIN_LTO_OBJECT_PATH: {{ld(.exe)?"}}
// THIN_LTO_OBJECT_PATH-SAME: "-object_path_lto"
// THIN_LTO_OBJECT_PATH-SAME: {{thinlto\-[a-zA-Z0-9_]+}}
