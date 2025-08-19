// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cpp.cir
// RUN: FileCheck --check-prefix=CIR-CPP --input-file=%t.cpp.cir %s
// RUN: %clang_cc1 -x c -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.c.cir
// RUN: FileCheck --check-prefix=CIR-C --input-file=%t.c.cir %s

// CIR-CPP: module attributes {{{.*}}cir.lang = #cir.lang<cxx>{{.*}}}
// CIR-C: module attributes {{{.*}}cir.lang = #cir.lang<c>{{.*}}}

int main() {
  return 0;
}
