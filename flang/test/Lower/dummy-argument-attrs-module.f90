! The defining compilation records dummy intent on the func.func arguments.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK: func.func @_QMmPtwice(%{{.*}}: !fir.ref<f64> {fir.bindc_name = "x", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}, %{{.*}}: !fir.ref<f64> {fir.bindc_name = "y", fir.fortran_attrs = #fir.var_attrs<intent_out>})
module m
contains
  subroutine twice(x, y)
    double precision, intent(in) :: x
    double precision, intent(out) :: y
    y = 2 * x
  end subroutine twice
end module m
