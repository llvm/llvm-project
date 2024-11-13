// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -ast-print %s -o - | FileCheck %s

void foo() {
// CHECK: #pragma acc parallel loop
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc serial loop
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc kernels loop
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc parallel loop auto
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop auto
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc serial loop seq
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop seq
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc kernels loop independent
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop independent
  for(int i = 0;i<5;++i);

  bool SomeB;
  struct SomeStruct{} SomeStructImpl;

//CHECK: #pragma acc parallel loop dtype(SomeB)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop dtype(SomeB)
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc serial loop device_type(SomeStruct)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop device_type(SomeStruct)
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc kernels loop device_type(int)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop device_type(int)
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc parallel loop dtype(bool)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop dtype(bool)
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc serial loop device_type(SomeStructImpl)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop device_type (SomeStructImpl)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels loop dtype(AnotherIdent)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop dtype(AnotherIdent)
  for(int i = 0;i<5;++i);


}
