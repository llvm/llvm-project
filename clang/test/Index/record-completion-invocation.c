// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env CINDEXTEST_INVOCATION_EMISSION_PATH=%t not c-index-test -code-completion-at=%s:10:1 "-remap-file=%s,%S/Inputs/record-parsing-invocation-remap.c" %s
// RUN: cat %t/libclang-* | FileCheck %s

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env LIBCLANG_DISABLE_CRASH_RECOVERY=1 CINDEXTEST_INVOCATION_EMISSION_PATH=%t not --crash c-index-test -code-completion-at=%s:10:1 "-remap-file=%s,%S/Inputs/record-parsing-invocation-remap.c" %s
// RUN: cat %t/libclang-* | FileCheck %s

// CHECK: {"toolchain":"{{.*}}","libclang.operation":"complete","libclang.opts":1,"args":["clang","-fno-spell-checking","{{.*}}record-completion-invocation.c","-Xclang","-detailed-preprocessing-record","-fallow-editor-placeholders"],"invocation-args":["-code-completion-at={{.*}}record-completion-invocation.c:10:1"],"unsaved_file_hashes":[{"name":"{{.*}}record-completion-invocation.c","md5":"aee23773de90e665992b48209351d70e"}]}
