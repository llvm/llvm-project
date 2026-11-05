! RUN: bbc -emit-fir -o - %s | FileCheck %s

! Constant array literals are hoisted into globals that are shared between
! equivalent literals.  Every constant expression hashes to the same bucket,
! so the equality predicate in Fortran::lower::isEqual() is what keeps them
! apart: literals that differ only in the kind of their elements must not be
! given the same global, or the global would be emitted with the element type
! of whichever literal was lowered first.

subroutine integer_kinds
  interface
    subroutine i1(x)
      integer(1) :: x(3)
    end subroutine
    subroutine i2(x)
      integer(2) :: x(3)
    end subroutine
    subroutine i4(x)
      integer(4) :: x(3)
    end subroutine
    subroutine i8(x)
      integer(8) :: x(3)
    end subroutine
  end interface
  call i1([1_1, 2_1, 3_1])
  call i2([1_2, 2_2, 3_2])
  call i4([1_4, 2_4, 3_4])
  call i8([1_8, 2_8, 3_8])
end subroutine
! CHECK-DAG: fir.global internal @_QQro.3xi1.{{[0-9]+}}(dense<[1, 2, 3]> : tensor<3xi8>) {{.*}} : !fir.array<3xi8>
! CHECK-DAG: fir.global internal @_QQro.3xi2.{{[0-9]+}}(dense<[1, 2, 3]> : tensor<3xi16>) {{.*}} : !fir.array<3xi16>
! CHECK-DAG: fir.global internal @_QQro.3xi4.{{[0-9]+}}(dense<[1, 2, 3]> : tensor<3xi32>) {{.*}} : !fir.array<3xi32>
! CHECK-DAG: fir.global internal @_QQro.3xi8.{{[0-9]+}}(dense<[1, 2, 3]> : tensor<3xi64>) {{.*}} : !fir.array<3xi64>

subroutine real_kinds
  interface
    subroutine r4(x)
      real(4) :: x(3)
    end subroutine
    subroutine r8(x)
      real(8) :: x(3)
    end subroutine
  end interface
  call r4([1.0_4, 2.0_4, 3.0_4])
  call r8([1.0_8, 2.0_8, 3.0_8])
end subroutine
! CHECK-DAG: fir.global internal @_QQro.3xr4.{{[0-9]+}}({{.*}} : tensor<3xf32>) {{.*}} : !fir.array<3xf32>
! CHECK-DAG: fir.global internal @_QQro.3xr8.{{[0-9]+}}({{.*}} : tensor<3xf64>) {{.*}} : !fir.array<3xf64>

subroutine logical_kinds
  interface
    subroutine l1(x)
      logical(1) :: x(2)
    end subroutine
    subroutine l4(x)
      logical(4) :: x(2)
    end subroutine
  end interface
  call l1([.true._1, .false._1])
  call l4([.true._4, .false._4])
end subroutine
! CHECK-DAG: fir.global internal @_QQro.2xl1.{{[0-9]+}}({{.*}} : tensor<2xi8>) {{.*}} : !fir.array<2x!fir.logical<1>>
! CHECK-DAG: fir.global internal @_QQro.2xl4.{{[0-9]+}}({{.*}} : tensor<2xi32>) {{.*}} : !fir.array<2x!fir.logical<4>>
