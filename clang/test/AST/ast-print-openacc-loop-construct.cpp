// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -ast-print %s -o - | FileCheck %s

struct SomeStruct{};

constexpr int get_value() { return 1; }
void foo() {
// CHECK: #pragma acc loop
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop device_type(default)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop device_type(default)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop device_type(nvidia)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop device_type(nvidia)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop dtype(radeon)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop dtype(radeon)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop dtype(host)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop dtype(host)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop independent
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop independent
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc loop seq
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop seq
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc loop auto
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop auto
  for(int i = 0;i<5;++i);

  int i;
  float array[5];

// CHECK: #pragma acc loop private(i, array[1], array, array[1:2])
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop private(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop collapse(1)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop collapse(1)
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc loop collapse(force:1)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop collapse(force:1)
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc loop collapse(2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop collapse(2)
  for(int i = 0;i<5;++i)
    for(int i = 0;i<5;++i);
// CHECK: #pragma acc loop collapse(force:2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop collapse(force:2)
  for(int i = 0;i<5;++i)
    for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop tile(1, 3, *, get_value())
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop tile(1, 3, *, get_value())
  for(int i = 0;i<5;++i)
    for(int i = 0;i<5;++i)
      for(int i = 0;i<5;++i)
        for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop gang(dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop gang(dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop gang(static: i) gang(dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop gang(static:i) gang(dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop gang(dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop gang(dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop gang(static: i) gang(dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop gang(static:i) gang(dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop gang(num: i) gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop gang(i) gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop gang(num: i) gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop gang(num:i) gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial
// CHECK-NEXT: #pragma acc loop gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial
#pragma acc loop gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial
// CHECK-NEXT: #pragma acc loop gang(static: *)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial
#pragma acc loop gang(static:*)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial
// CHECK-NEXT: #pragma acc loop
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial
#pragma acc loop gang
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop worker
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop worker
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop worker
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop worker
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial
// CHECK-NEXT: #pragma acc loop worker
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial
#pragma acc loop worker
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop worker(num: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop worker(5)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop worker(num: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop worker(num:5)
  for(int i = 0;i<5;++i);

  // CHECK: #pragma acc loop vector
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop vector
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop vector(5)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc loop vector(length:5)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop vector
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop vector
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop vector(5)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel
// CHECK-NEXT: #pragma acc loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel
#pragma acc loop vector(length:5)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop vector
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop vector
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop vector(5)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels
// CHECK-NEXT: #pragma acc loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels
#pragma acc loop vector(length:5)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial
// CHECK-NEXT: #pragma acc loop vector
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial
#pragma acc loop vector
  for(int i = 0;i<5;++i);

  bool SomeB;

//CHECK: #pragma acc loop reduction(*: i)
#pragma acc loop reduction(*: i)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc loop reduction(max: SomeB)
#pragma acc loop reduction(max: SomeB)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc loop reduction(&: i)
#pragma acc loop reduction(&: i)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc loop reduction(|: SomeB)
#pragma acc loop reduction(|: SomeB)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc loop reduction(&&: i)
#pragma acc loop reduction(&&: i)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc loop reduction(||: SomeB)
#pragma acc loop reduction(||: SomeB)
  for(int i = 0;i<5;++i);
}
