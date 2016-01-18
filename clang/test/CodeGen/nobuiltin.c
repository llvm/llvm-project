// RUN: %clang_cc1 -fno-builtin -O1 -S -o - %s | FileCheck %s

void PR13497() {
  char content[2];
  // make sure we don't optimize this call to strcpy()
  // CHECK: __strcpy_chk
  __builtin___strcpy_chk(content, "", 1);
}
