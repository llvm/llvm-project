// XFAIL: linux

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.7 -fsyntax-only %s -o %t.o -index-store-path %t/idx1 -fmodules -fmodules-cache-path=%t/mcp -I %S/Inputs/module -Rindex-store 2> %t.err1
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.7 -fsyntax-only %s -o %t.o -index-store-path %t/idx2 -fmodules -fmodules-cache-path=%t/mcp -I %S/Inputs/module -Rindex-store 2> %t.err2
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.7 -fsyntax-only %s -o %t.o -index-store-path %t/idx2 -fmodules -fmodules-cache-path=%t/mcp -I %S/Inputs/module -Rindex-store 2> %t.err3
// RUN: FileCheck -input-file=%t.err1 -check-prefix=CREATING_MODULES %s -allow-empty
// RUN: FileCheck -input-file=%t.err2 -check-prefix=CREATING_INDEX_DATA_FROM_MODULE_FILES %s
// RUN: FileCheck -input-file=%t.err3 -check-prefix=EXISTING_INDEX_DATA_FROM_MODULE_FILES %s -allow-empty
// RUN: c-index-test core -print-unit %t/idx1 > %t/all-units1.txt
// RUN: c-index-test core -print-unit %t/idx2 > %t/all-units2.txt
// RUN: c-index-test core -print-record %t/idx1 > %t/all-records1.txt
// RUN: c-index-test core -print-record %t/idx2 > %t/all-records2.txt
// RUN: diff -u %t/all-units1.txt %t/all-units2.txt
// RUN: diff -u %t/all-records1.txt %t/all-records2.txt

@import ModDep;

// CREATING_MODULES-NOT: remark:

// CREATING_INDEX_DATA_FROM_MODULE_FILES: remark: producing index data for module file {{.*}}ModDep{{.*}}.pcm
// CREATING_INDEX_DATA_FROM_MODULE_FILES: remark: producing index data for module file {{.*}}ModTop{{.*}}.pcm

// EXISTING_INDEX_DATA_FROM_MODULE_FILES-NOT: remark:
