! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}

! This test was extracted from product.f90, since most tests in product.f90 didn't
! need types specific to x86 platform.

! Note: originally, the test used 'REQUIRES' directive for
! x86-registered-target' to try to avoid running on platforms that don't
! support kind 10. Unfortunately, this checks if the x86 backend was compiled
! into LLVM, not whether the host is x86. On ARM CI machines with multi-target
! LLVM builds, this feature is set to true, so the test runs. With
! COMPLEX(KIND=10) (f80) not unsupported on ARM, this caused a semantic error.
! For this reason, change this test to support CHECK-KIND10 mechanism. (The
! same mechanism is used by nearest.f90 and sum.f90 tests.)

! CHECK-LABEL: func @_QPproduct_test4(
! CHECK-KIND10-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xcomplex<f80>>>{{.*}}) -> complex<f80>
function product_test4(x)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  complex(kind10) :: product_test4
  complex(kind10):: x(:)
  product_test4 = product(x)
! CHECK-KIND10: hlfir.product {{.*}} : (!fir.box<!fir.array<?xcomplex<f80>>>) -> complex<f80>
end function
