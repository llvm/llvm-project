// RUN: tco -emit-fir %s | tco | FileCheck %s

// TODO: Is test correct?
// CHECK-LABEL: define void @f(double* %0)
func @f(%arg : !fir.ref<!fir.array<?xf64>>) {
  %c1000 = constant 1000 : index
  %aDim = fir.shape %c1000 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg(%aDim) : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  return
}
