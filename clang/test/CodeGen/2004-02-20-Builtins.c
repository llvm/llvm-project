// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s
double sqrt(double x);

// CHECK-LABEL: @zsqrtxxx
// CHECK-NOT: builtin
// Don't search into metadata definitions.  !llvm.ident can contain the
// substring "builtin" if it's in the source tree path.
// CHECK-LABEL: !llvm.ident
void zsqrtxxx(float num) {
   num = sqrt(num);
}
