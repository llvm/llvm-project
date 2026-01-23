// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=limited %s | FileCheck %s

int somefunc(char *x, int y, double z) {
  
  // CHECK: !DISubroutineType(types: ![[NUM:[0-9]+]])
  // CHECK: ![[NUM]] = {{!{![^,]*, ![^,]*, ![^,]*, ![^,]*}}}
  
  return y;
}
