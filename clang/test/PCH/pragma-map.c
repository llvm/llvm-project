// REQUIRES: systemz-registered-target

/// Test this without pch.
// RUN: %clang_cc1 -triple s390x-ibm-zos -include %S/pragma-map.h %s -emit-llvm -o - | FileCheck %s

/// Test with pch.
// RUN: %clang_cc1 -source-date-epoch 1613676318 -triple s390x-ibm-zos -x c-header -emit-pch -o %t %S/pragma-map.h
// RUN: %clang_cc1 -triple s390x-ibm-zos -include-pch %t %s -emit-llvm -o - | FileCheck %s

/// Compile it a few times to check that the PCH file is deterministic.
// RUN: %clang_cc1 -source-date-epoch 1613676318 -triple s390x-ibm-zos -x c-header -emit-pch -o %t.cmp %S/pragma-map.h
// RUN: diff %t %t.cmp >/dev/null
// RUN: %clang_cc1 -source-date-epoch 1613676318 -triple s390x-ibm-zos -x c-header -emit-pch -o %t.cmp %S/pragma-map.h
// RUN: diff %t %t.cmp >/dev/null
// RUN: %clang_cc1 -source-date-epoch 1613676318 -triple s390x-ibm-zos -x c-header -emit-pch -o %t.cmp %S/pragma-map.h
// RUN: diff %t %t.cmp >/dev/null

/// Check that #pragma map is exported into PCHs even if the header contains no
/// declaration of the symbol.
void undecfunc1(void);
void undecfunc2(void) {}
void undecfunc3(void);
void undecfunc4(void);
void undecfunc5(void);
void undecfunc6(void);
extern int undecvar;
void undecesc(void);

void use(void) {
  undecfunc1();
  undecfunc2();
  undecfunc3();
  undecfunc4();
  undecfunc5();
  undecfunc6();
  undecvar = 1;
  undecesc();
  declfunc();
}

// CHECK-DAG: @UNDECVAR = external global i32
// CHECK-DAG: define void @UNDEC2()
// CHECK-DAG: call void @UNDEC1()
// CHECK-DAG: call void @UNDEC2()
// CHECK-DAG: call void @UNDEC3()
// CHECK-DAG: call void @UNDEC4()
// CHECK-DAG: call void @UNDEC5()
// CHECK-DAG: call void @UNDEC6()
// CHECK-DAG: call void @"@ESC"()
// CHECK-DAG: call void @DECL()
