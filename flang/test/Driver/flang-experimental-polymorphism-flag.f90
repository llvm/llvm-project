! Test -flang-experimental-hlfir flag
! RUN: %flang_fc1 -flang-experimental-polymorphism -emit-fir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -o - %s 2>&1 | FileCheck %s --check-prefix NO-POLYMORPHISM

! CHECK: func.func @_QPtest(%{{.*}}: !fir.class<none> {fir.bindc_name = "poly"})
subroutine test(poly)
  class(*) :: poly
end subroutine test

! NO-POLYMORPHISM: func.func @_QPtest
