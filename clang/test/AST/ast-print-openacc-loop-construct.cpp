// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -ast-print %s -o - | FileCheck %s

struct SomeStruct{};

constexpr int get_value() { return 1; }
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

// CHECK: #pragma acc loop independent
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop independent
  for(;;);
// CHECK: #pragma acc loop seq
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop seq
  for(;;);
// CHECK: #pragma acc loop auto
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop auto
  for(;;);

  int i;
  float array[5];

// CHECK: #pragma acc loop private(i, array[1], array, array[1:2])
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop private(i, array[1], array, array[1:2])
  for(;;);

// CHECK: #pragma acc loop collapse(1)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop collapse(1)
  for(;;);
// CHECK: #pragma acc loop collapse(force:1)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop collapse(force:1)
  for(;;);
// CHECK: #pragma acc loop collapse(2)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop collapse(2)
  for(;;)
    for(;;);
// CHECK: #pragma acc loop collapse(force:2)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop collapse(force:2)
  for(;;)
    for(;;);

// CHECK: #pragma acc loop tile(1, 3, *, get_value())
// CHECK-NEXT: for (;;)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop tile(1, 3, *, get_value())
  for(;;)
    for(;;)
      for(;;)
        for(;;);

// CHECK: #pragma acc loop gang(dim: 2)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop gang(dim:2)
  for(;;);

// CHECK: #pragma acc loop gang(static: i)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop gang(static:i)
  for(;;);

// CHECK: #pragma acc loop gang(static: i) gang(dim: 2)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop gang(static:i) gang(dim:2)
  for(;;);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop gang(dim: 2)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop gang(dim:2)
  for(;;);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop gang(static: i)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop gang(static:i)
  for(;;);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop gang(static: i) gang(dim: 2)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop gang(static:i) gang(dim:2)
  for(;;);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop gang(num: i) gang(static: i)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop gang(i) gang(static:i)
  for(;;);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop gang(num: i) gang(static: i)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop gang(num:i) gang(static:i)
  for(;;);

// CHECK: #pragma acc serial
// CHECK-NEXT: #pragma acc loop gang(static: i)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc serial
#pragma acc loop gang(static:i)
  for(;;);

// CHECK: #pragma acc serial
// CHECK-NEXT: #pragma acc loop gang(static: *)
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc serial
#pragma acc loop gang(static:*)
  for(;;);

// CHECK: #pragma acc serial
// CHECK-NEXT: #pragma acc loop
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc serial
#pragma acc loop gang
  for(;;);
}
