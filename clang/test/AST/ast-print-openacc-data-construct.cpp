// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -Wno-source-uses-openacc -ast-print %s -o - | FileCheck %s

void foo() {
  int Var;
  // TODO OpenACC: These are only legal if they have one of a list of clauses on
  // them, so the 'check' lines should start to include those once we implement
  // them.  For now, they don't emit those because they are 'not implemented'.

// CHECK: #pragma acc data default(none)
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

  int i;
  int *iPtr;
  int array[5];

// CHECK: #pragma acc data default(none) if(i == array[1])
#pragma acc data default(none) if(i == array[1])
  ;
// CHECK: #pragma acc enter data if(i == array[1])
#pragma acc enter data copyin(Var) if(i == array[1])
  ;
// CHECK: #pragma acc exit data if(i == array[1])
#pragma acc exit data copyout(Var) if(i == array[1])
  ;
// CHECK: #pragma acc host_data if(i == array[1])
#pragma acc host_data use_device(Var) if(i == array[1])
  ;

// CHECK: #pragma acc data default(none) async(i)
#pragma acc data default(none) async(i)
  ;
// CHECK: #pragma acc enter data async(i)
#pragma acc enter data copyin(i) async(i)
// CHECK: #pragma acc exit data async
#pragma acc exit data copyout(i) async

// CHECK: #pragma acc data default(none) wait
#pragma acc data default(none) wait()
  ;

// CHECK: #pragma acc enter data wait()
#pragma acc enter data copyin(Var) wait()

// CHECK: #pragma acc exit data wait(*iPtr, i)
#pragma acc exit data copyout(Var) wait(*iPtr, i)

// CHECK: #pragma acc data default(none) wait(queues: *iPtr, i)
#pragma acc data default(none) wait(queues:*iPtr, i)
  ;

// CHECK: #pragma acc enter data wait(devnum: i : *iPtr, i)
#pragma acc enter data copyin(Var) wait(devnum:i:*iPtr, i)

// CHECK: #pragma acc exit data wait(devnum: i : queues: *iPtr, i)
#pragma acc exit data copyout(Var) wait(devnum:i:queues:*iPtr, i)

// CHECK: #pragma acc data default(none)
#pragma acc data default(none)
  ;

// CHECK: #pragma acc data default(present)
#pragma acc data default(present)
  ;
}
