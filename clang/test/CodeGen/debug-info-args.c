// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -g %s | FileCheck %s

int somefunc(char *x, int y, double z) {
  
  // CHECK: {{.*metadata !8, i32 0, i32 0}.*DW_TAG_subroutine_type}}
  // CHECK: {{!8 = .*metadata ![^,]*, metadata ![^,]*, metadata ![^,]*, metadata ![^,]*}}
  
  return y;
}
