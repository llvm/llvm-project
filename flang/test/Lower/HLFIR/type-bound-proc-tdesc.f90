! Test lowering of type bound procedure in the derived type descriptors (that
! are compiler generated constant structure constructors).
! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s

module type_bound_proc_tdesc
  type :: t
  contains
    procedure, nopass :: simple => simple_impl
    procedure, nopass :: return_char => return_char_impl
  end type

interface
  function return_char_impl()
    character(10) :: return_char_impl
  end function
  subroutine simple_impl()
  end subroutine
end interface
end
  use type_bound_proc_tdesc
  type(t) :: a
end

! CHECK-LABEL: fir.global {{.*}} @_QMtype_bound_proc_tdescE.v.t
! CHECK:  fir.address_of(@_QPreturn_char_impl) : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_8:.*]] = fir.extract_value %{{.*}}, [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  fir.box_addr %[[VAL_8]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! ...
! CHECK:  %[[VAL_25:.*]] = fir.address_of(@_QPsimple_impl) : () -> ()
! CHECK:  %[[VAL_26:.*]] = fir.emboxproc %[[VAL_25]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  fir.box_addr %[[VAL_26]] : (!fir.boxproc<() -> ()>) -> (() -> ())
