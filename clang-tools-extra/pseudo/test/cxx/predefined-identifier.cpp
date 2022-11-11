// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
void s() {
  __func__;
  // CHECK: expression~__FUNC__ := tok[5]
}
