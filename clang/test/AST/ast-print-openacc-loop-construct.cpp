// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -ast-print %s -o - | FileCheck %s

struct SomeStruct{};

void foo() {
// CHECK: #pragma acc loop
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop
  for(;;);

// CHECK: #pragma acc loop device_type(SomeStruct)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop device_type(SomeStruct)
  for(;;);

// CHECK: #pragma acc loop device_type(int)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop device_type(int)
  for(;;);

// CHECK: #pragma acc loop dtype(bool)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop dtype(bool)
  for(;;);

// CHECK: #pragma acc loop dtype(AnotherIdent)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop dtype(AnotherIdent)
  for(;;);
}
