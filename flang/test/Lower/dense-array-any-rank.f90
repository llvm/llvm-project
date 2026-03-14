! RUN: bbc -emit-fir -o - %s | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck --check-prefixes="CHECK-LLVMIR" %s

! CHECK-LABEL: test
subroutine test()
  integer, dimension(10) :: a1
  integer, dimension(3,4) :: a2
  integer, dimension(2,3,4) :: a3

  a1 = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
  a2 = reshape((/11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43/), shape(a2))
  a3 = reshape((/111, 112, 121, 122, 131, 132, 211, 212, 221, 222, 231, 232, 311, 312, 321, 322, 331, 332, 411, 412, 421, 422, 431, 432/), shape(a3))
end subroutine

! a1 array constructor
! CHECK-FIR: fir.global internal @_QQro.10xi4.{{.*}}(dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi32>) constant : !fir.array<10xi32>
! CHECK-LLVMIR: @_QQroX10xi4X0 = internal constant [10 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10]

! a2 array constructor
! CHECK-FIR: fir.global internal @_QQro.3x4xi4.{{.*}}(dense<{{\[\[11, 12, 13], \[21, 22, 23], \[31, 32, 33], \[41, 42, 43]]}}> : tensor<4x3xi32>) constant : !fir.array<3x4xi32>
! CHECK-LLVMIR: @_QQroX3x4xi4X1 = internal constant [4 x [3 x i32]] {{\[\[3 x i32] \[i32 11, i32 12, i32 13], \[3 x i32] \[i32 21, i32 22, i32 23], \[3 x i32] \[i32 31, i32 32, i32 33], \[3 x i32] \[i32 41, i32 42, i32 43]]}}

! a3 array constructor
! CHECK-FIR: fir.global internal @_QQro.2x3x4xi4.{{.*}}(dense<{{\[\[\[111, 112], \[121, 122], \[131, 132]], \[\[211, 212], \[221, 222], \[231, 232]], \[\[311, 312], \[321, 322], \[331, 332]], \[\[411, 412], \[421, 422], \[431, 432]]]}}> : tensor<4x3x2xi32>) constant : !fir.array<2x3x4xi32>
! CHECK-LLVMIR: @_QQroX2x3x4xi4X2 = internal constant [4 x [3 x [2 x i32]]] {{\[\[3 x \[2 x i32]] \[\[2 x i32] \[i32 111, i32 112], \[2 x i32] \[i32 121, i32 122], \[2 x i32] \[i32 131, i32 132]], \[3 x \[2 x i32]] \[\[2 x i32] \[i32 211, i32 212], \[2 x i32] \[i32 221, i32 222], \[2 x i32] \[i32 231, i32 232]], \[3 x \[2 x i32]] \[\[2 x i32] \[i32 311, i32 312], \[2 x i32] \[i32 321, i32 322], \[2 x i32] \[i32 331, i32 332]], \[3 x \[2 x i32]] \[\[2 x i32] \[i32 411, i32 412], \[2 x i32] \[i32 421, i32 422], \[2 x i32] \[i32 431, i32 432]]]}}
