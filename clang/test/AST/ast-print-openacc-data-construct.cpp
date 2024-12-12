// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -Wno-source-uses-openacc -ast-print %s -o - | FileCheck %s

void foo() {
  int Var;
  // TODO OpenACC: These are only legal if they have one of a list of clauses on
  // them, so the 'check' lines should start to include those once we implement
  // them.  For now, they don't emit those because they are 'not implemented'.

// CHECK: #pragma acc data
// CHECK-NOT: default(none)
#pragma acc data default(none)
  ;

// CHECK: #pragma acc data device_type(int)
#pragma acc data device_type(int)
  ;

// CHECK: #pragma acc enter data
// CHECK-NOT: copyin(Var)
#pragma acc enter data copyin(Var)
  ;
// CHECK: #pragma acc exit data
// CHECK-NOT: copyout(Var)
#pragma acc exit data copyout(Var)
  ;
// CHECK: #pragma acc host_data
// CHECK-NOT: use_device(Var)
#pragma acc host_data use_device(Var)
  ;
}
