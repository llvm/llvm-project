// RUN: rm -rf %t && mkdir %t
// RUN: touch %t/asan_ignorelist.txt
// RUN: touch %t/sys_asan_ignorelist.txt

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -fsanitize=address -Xclang -fsanitize-ignorelist=%t/asan_ignorelist.txt -Xclang -fsanitize-system-ignorelist=%t/sys_asan_ignorelist.txt \
// RUN:   -target x86_64-apple-macos11 -c %s -o %t/output.o -Rcompile-job-cache 2> %t/output-tree.txt

// RUN: cat %t/output-tree.txt | sed \
// RUN:   -e "s/^.*miss for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key-tree

// RUN: clang-cas-test -print-compile-job-cache-key -cas %t/cas/cas @%t/cache-key-tree > %t/key.txt
// RUN: FileCheck %s -DSRC_FILE=%s -DOUT_DIR=%t -input-file %t/key.txt
//
// CHECK: -fsanitize=address \
// CHECK: -fsanitize-ignorelist=[[OUT_DIR]]{{/|\\}}asan_ignorelist.txt \
// CHECK: -fsanitize-ignorelist=[[OUT_DIR]]{{/|\\}}sys_asan_ignorelist.txt \

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas LLVM_CACHE_PREFIX_MAPS="%S=/^src;%t=/^out" %clang-cache \
// RUN:   %clang -fsanitize=address -Xclang -fsanitize-ignorelist=%t/asan_ignorelist.txt -Xclang -fsanitize-system-ignorelist=%t/sys_asan_ignorelist.txt \
// RUN:   -target x86_64-apple-macos11 -c %s -o %t/output.o -Rcompile-job-cache 2> %t/output-tree2.txt

// RUN: cat %t/output-tree2.txt | sed \
// RUN:   -e "s/^.*miss for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key-tree2

// RUN: clang-cas-test -print-compile-job-cache-key -cas %t/cas/cas @%t/cache-key-tree2 > %t/key2.txt
// RUN: FileCheck %s -input-file %t/key2.txt -check-prefix=PREFIXED
//
// PREFIXED: -fsanitize=address \
// PREFIXED: -fsanitize-ignorelist=/^out{{/|\\}}asan_ignorelist.txt \
// PREFIXED: -fsanitize-ignorelist=/^out{{/|\\}}sys_asan_ignorelist.txt \

void test() {}
