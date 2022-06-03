// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --print-forest | FileCheck %s
void s() {
  __func__;
  // CHECK: expression~__FUNC__ := tok[5]
}
