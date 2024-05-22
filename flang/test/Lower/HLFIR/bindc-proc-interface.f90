! Test mangling with BIND(C) inherited from procedure interface.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test()
  interface
    subroutine iface_notbindc()
    end subroutine
    subroutine iface_bindc() bind(c)
    end subroutine
    subroutine iface_explicit_name() bind(c, name="explicit_name")
    end subroutine
    subroutine iface_nobinding() bind(c, name="")
    end subroutine
  end interface

  procedure(iface_bindc) :: foo_iface_bindc
  procedure(iface_explicit_name) :: foo_iface_explicit_name
  procedure(iface_nobinding) :: foo_iface_nobinding

  procedure(iface_bindc), bind(c) :: extra_bindc_iface_bindc
  procedure(iface_explicit_name), bind(c) :: extra_bindc_iface_explicit_name
  procedure(iface_nobinding), bind(c) :: extra_bindc_iface_nobinding

  procedure(iface_bindc),  bind(c, name="bar_iface_bindc_2") :: bar_iface_bindc
  procedure(iface_explicit_name),  bind(c,name="bar_iface_explicit_name_2") ::  bar_iface_explicit_name
  procedure(iface_nobinding), bind(c, name="bar_iface_nobinding_2") :: bar_iface_nobinding

  procedure(iface_bindc),  bind(c, name="") :: nobinding_iface_bindc
  procedure(iface_explicit_name),  bind(c, name="") :: nobinding_iface_explicit_name
  procedure(iface_nobinding), bind(c, name="") :: nobinding_iface_nobinding

  call iface_notbindc()
  call iface_bindc()
  call iface_explicit_name()
  call iface_nobinding()

  call foo_iface_bindc()
  call foo_iface_explicit_name()
  call foo_iface_nobinding()

  call extra_bindc_iface_bindc()
  call extra_bindc_iface_explicit_name()
  call extra_bindc_iface_nobinding()

  call bar_iface_bindc()
  call bar_iface_explicit_name()
  call bar_iface_nobinding()

  call nobinding_iface_bindc()
  call nobinding_iface_explicit_name()
  call nobinding_iface_nobinding()

! CHECK:  fir.call @_QPiface_notbindc()
! CHECK:  fir.call @iface_bindc()
! CHECK:  fir.call @explicit_name()
! CHECK:  fir.call @_QPiface_nobinding()
! CHECK:  fir.call @foo_iface_bindc()
! CHECK:  fir.call @foo_iface_explicit_name()
! CHECK:  fir.call @foo_iface_nobinding()
! CHECK:  fir.call @extra_bindc_iface_bindc()
! CHECK:  fir.call @extra_bindc_iface_explicit_name()
! CHECK:  fir.call @extra_bindc_iface_nobinding()
! CHECK:  fir.call @bar_iface_bindc_2()
! CHECK:  fir.call @bar_iface_explicit_name_2()
! CHECK:  fir.call @bar_iface_nobinding_2()
! CHECK:  fir.call @_QPnobinding_iface_bindc()
! CHECK:  fir.call @_QPnobinding_iface_explicit_name()
! CHECK:  fir.call @_QPnobinding_iface_nobinding()
end subroutine
