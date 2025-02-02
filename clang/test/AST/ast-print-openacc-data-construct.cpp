// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -Wno-source-uses-openacc -ast-print %s -o - | FileCheck %s

void foo() {
  int Var;
  // TODO OpenACC: These are only legal if they have one of a list of clauses on
  // them, so the 'check' lines should start to include those once we implement
  // them.  For now, they don't emit those because they are 'not implemented'.

// CHECK: #pragma acc data default(none)
#pragma acc data default(none)
  ;

// CHECK: #pragma acc data default(none) device_type(int)
#pragma acc data default(none) device_type(int)
  ;

// CHECK: #pragma acc enter data copyin(Var)
#pragma acc enter data copyin(Var)
  ;
// CHECK: #pragma acc exit data copyout(Var)
#pragma acc exit data copyout(Var)
  ;
// CHECK: #pragma acc host_data use_device(Var)
#pragma acc host_data use_device(Var)
  ;

  int i;
  int *iPtr;
  int array[5];

// CHECK: #pragma acc data default(none) if(i == array[1])
#pragma acc data default(none) if(i == array[1])
  ;
// CHECK: #pragma acc enter data copyin(Var) if(i == array[1])
#pragma acc enter data copyin(Var) if(i == array[1])
  ;
// CHECK: #pragma acc exit data copyout(Var) if(i == array[1])
#pragma acc exit data copyout(Var) if(i == array[1])
  ;
// CHECK: #pragma acc host_data use_device(Var) if(i == array[1])
#pragma acc host_data use_device(Var) if(i == array[1])
  ;

// CHECK: #pragma acc data default(none) async(i)
#pragma acc data default(none) async(i)
  ;
// CHECK: #pragma acc enter data copyin(i) async(i)
#pragma acc enter data copyin(i) async(i)
// CHECK: #pragma acc exit data copyout(i) async
#pragma acc exit data copyout(i) async

// CHECK: #pragma acc data default(none) wait
#pragma acc data default(none) wait()
  ;

// CHECK: #pragma acc enter data copyin(Var) wait()
#pragma acc enter data copyin(Var) wait()

// CHECK: #pragma acc exit data copyout(Var) wait(*iPtr, i)
#pragma acc exit data copyout(Var) wait(*iPtr, i)

// CHECK: #pragma acc data default(none) wait(queues: *iPtr, i)
#pragma acc data default(none) wait(queues:*iPtr, i)
  ;

// CHECK: #pragma acc enter data copyin(Var) wait(devnum: i : *iPtr, i)
#pragma acc enter data copyin(Var) wait(devnum:i:*iPtr, i)

// CHECK: #pragma acc exit data copyout(Var) wait(devnum: i : queues: *iPtr, i)
#pragma acc exit data copyout(Var) wait(devnum:i:queues:*iPtr, i)

// CHECK: #pragma acc data default(none)
#pragma acc data default(none)
  ;

// CHECK: #pragma acc data default(present)
#pragma acc data default(present)
  ;

// CHECK: #pragma acc data default(none) no_create(i, array[1], array, array[1:2])
#pragma acc data default(none) no_create(i, array[1], array, array[1:2])
  ;

// CHECK: #pragma acc data default(none) no_create(i, array[1], array, array[1:2]) present(i, array[1], array, array[1:2])
#pragma acc data default(none) no_create(i, array[1], array, array[1:2]) present(i, array[1], array, array[1:2])
  ;
// CHECK: #pragma acc data present(i, array[1], array, array[1:2])
#pragma acc data present(i, array[1], array, array[1:2])
  ;

// CHECK: #pragma acc data default(none) copy(i, array[1], array, array[1:2]) pcopy(i, array[1], array, array[1:2]) present_or_copy(i, array[1], array, array[1:2])
#pragma acc data default(none) copy(i, array[1], array, array[1:2]) pcopy(i, array[1], array, array[1:2]) present_or_copy(i, array[1], array, array[1:2])
  ;

// CHECK: #pragma acc enter data copyin(i, array[1], array, array[1:2]) pcopyin(readonly: i, array[1], array, array[1:2]) present_or_copyin(i, array[1], array, array[1:2])
#pragma acc enter data copyin(i, array[1], array, array[1:2]) pcopyin(readonly:i, array[1], array, array[1:2]) present_or_copyin(i, array[1], array, array[1:2])

// CHECK: #pragma acc exit data copyout(i, array[1], array, array[1:2]) pcopyout(zero: i, array[1], array, array[1:2]) present_or_copyout(i, array[1], array, array[1:2])
#pragma acc exit data copyout(i, array[1], array, array[1:2]) pcopyout(zero: i, array[1], array, array[1:2]) present_or_copyout(i, array[1], array, array[1:2])

// CHECK: #pragma acc enter data create(i, array[1], array, array[1:2]) pcreate(zero: i, array[1], array, array[1:2]) present_or_create(i, array[1], array, array[1:2])
#pragma acc enter data create(i, array[1], array, array[1:2]) pcreate(zero: i, array[1], array, array[1:2]) present_or_create(i, array[1], array, array[1:2])

  float *arrayPtr[5];

// CHECK: #pragma acc data default(none) deviceptr(iPtr, arrayPtr[0])
#pragma acc data default(none) deviceptr(iPtr, arrayPtr[0])

// CHECK: #pragma acc data default(none) attach(iPtr, arrayPtr[0])
#pragma acc data default(none) attach(iPtr, arrayPtr[0])
  ;

// CHECK: #pragma acc exit data copyout(i) finalize
#pragma acc exit data copyout(i) finalize

// CHECK: #pragma acc host_data use_device(i) if_present
#pragma acc host_data use_device(i) if_present
  ;
// CHECK: #pragma acc exit data copyout(i) detach(iPtr, arrayPtr[0])
#pragma acc exit data copyout(i) detach(iPtr, arrayPtr[0])

// CHECK: #pragma acc exit data copyout(i) delete(i, array[1], array, array[1:2])
#pragma acc exit data copyout(i) delete(i, array[1], array, array[1:2])
  ;

// CHECK: #pragma acc exit data copyout(i) delete(i, array[1], array, array[1:2])
#pragma acc exit data copyout(i) delete(i, array[1], array, array[1:2])

// CHECK: #pragma acc host_data use_device(i)
#pragma acc host_data use_device(i)
  ;
}
