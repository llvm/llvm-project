! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! This test was extracted from product.f90, since most tests in product.f90 didn't
! need types specific to x86 platform.

! CHECK-LABEL: func @_QPproduct_test4(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xcomplex<f80>>>{{.*}}) -> complex<f80>
complex(10) function product_test4(x)
complex(10):: x(:)
product_test4 = product(x)
! CHECK: hlfir.product {{.*}} : (!fir.box<!fir.array<?xcomplex<f80>>>) -> complex<f80>
end
