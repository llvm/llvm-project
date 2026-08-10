! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-DAG: func.func private @proc1() attributes {fir.bindc_name = "proc1", fir.proc_attrs = #fir.proc_attrs<bind_c>}
module decl1
  interface
     subroutine proc_iface() bind(C)
     end subroutine proc_iface
  end interface
  procedure (proc_iface) PrOc1
end module decl1
subroutine test1(x)
  use decl1
  call PrOc1
end subroutine test1

! CHECK-DAG: func.func private @proc2() attributes {fir.bindc_name = "proc2", fir.proc_attrs = #fir.proc_attrs<bind_c>}
module decl2
  interface
     subroutine proc_iface() bind(C)
     end subroutine proc_iface
  end interface
end module decl2
subroutine test2(x)
  use decl2
  procedure (proc_iface) PrOc2
  call PrOc2
end subroutine test2

! CHECK-DAG: func.func private @func3() -> f32 attributes {fir.bindc_name = "func3", fir.proc_attrs = #fir.proc_attrs<bind_c>}
module decl3
  interface
     real function func_iface() bind(C)
     end function func_iface
  end interface
  procedure (func_iface) FuNc3
end module decl3
subroutine test3(x)
  use decl3
  real :: x
  x = FuNc3()
end subroutine test3

! CHECK-DAG: func.func private @func4() -> f32 attributes {fir.bindc_name = "func4", fir.proc_attrs = #fir.proc_attrs<bind_c>}
module decl4
  interface
     real function func_iface() bind(C)
     end function func_iface
  end interface
end module decl4
subroutine test4(x)
  use decl4
  procedure (func_iface) FuNc4
  real :: x
  x = FuNc4()
end subroutine test4

