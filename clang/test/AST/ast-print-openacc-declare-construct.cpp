// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

int *Global, *Global2;
int GlobalArray[5];
int GlobalArray2[5];
// CHECK: #pragma acc declare deviceptr(Global) copyin(GlobalArray)
#pragma acc declare deviceptr(Global), copyin(GlobalArray)
// CHECK: #pragma acc declare create(Global2, GlobalArray2)
#pragma acc declare create(Global2, GlobalArray2)

namespace NS {
int NSVar;
int NSArray[5];
// CHECK: #pragma acc declare create(NSVar, NSArray)
#pragma acc declare create(NSVar, NSArray)
}

struct Struct {
  static const int StaticMem = 5;
  static const int StaticMemArray[5];
// CHECK: #pragma acc declare copyin(StaticMem, StaticMemArray)
#pragma acc declare copyin(StaticMem, StaticMemArray)

  void MemFunc1(int Arg) {
    int Local;
    int LocalArray[5];
// CHECK: #pragma acc declare present(Arg, Local, LocalArray)
#pragma acc declare present(Arg, Local, LocalArray)
  }
  void MemFunc2(int Arg);
};
void Struct::MemFunc2(int Arg) {
  int Local;
  int LocalArray[5];
// CHECK: #pragma acc declare present(Arg, Local, LocalArray)
#pragma acc declare present(Arg, Local, LocalArray)
}

void NormalFunc(int Arg) {
  int Local;
  int LocalArray[5];
// CHECK: #pragma acc declare present(Arg, Local, LocalArray)
#pragma acc declare present(Arg, Local, LocalArray)
}

void NormalFunc2(int *Arg) {
  int Local;
  int LocalArray[5];
  extern int ExternLocal;
// CHECK: #pragma acc declare deviceptr(Arg) device_resident(Local) link(ExternLocal)
#pragma acc declare deviceptr(Arg) device_resident(Local) link(ExternLocal)
}
