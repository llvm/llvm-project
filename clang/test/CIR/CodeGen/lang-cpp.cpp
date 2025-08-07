// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// CIR: module attributes {{{.*}}cir.lang = #cir.lang<cxx>{{.*}}}

int main() {
  return 0;
}
