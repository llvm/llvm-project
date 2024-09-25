// RUN: rm -rf %/t
// RUN: mkdir %/t
// RUN: cp %/S/Inputs/file.script %/t/file.script
// RUN: cp %/S/Inputs/runtime_file.script %/t/runtime_file.script
// Need to embed the correct temp path in the actual JSON-RPC requests.
// RUN: sed -e "s|DIRECTORY|%/t|" %/t/file.script > %/t/file.script.temp

// RUN: clang-query -c 'file %/t/file.script.temp' %s -- | FileCheck %s

// CHECK: file-query.c:11:1: note: "f" binds here
void bar(void) {}

// CHECK: file-query.c:14:1: note: "v" binds here
int baz{1};
