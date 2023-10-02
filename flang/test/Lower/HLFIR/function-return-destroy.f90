! RUN: bbc -emit-hlfir -polymorphic-type %s -o - -I nowhere | FileCheck %s

module types
  type t1
     real :: x
  end type t1
  type t2
     real, allocatable :: x
  end type t2
  type t3
     real, pointer :: p
  end type t3
  type t4
     type(t1) :: c
  end type t4
  type t5
     type(t2) :: c
  end type t5
  type t6
   contains
     final :: finalize_t6
  end type t6
  type, extends(t1) :: t7
  end type t7
  type, extends(t2) :: t8
  end type t8
  type, extends(t6) :: t9
  end type t9
contains
  subroutine finalize_t6(x)
    type(t6), intent(inout) :: x
  end subroutine finalize_t6
end module types

subroutine test1
  use types
  interface
     function ret_type_t1
       use types
       type(t1) :: ret_type_t1
     end function ret_type_t1
  end interface
  type(t1) :: x
  x = ret_type_t1()
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1() {
! CHECK-NOT: fir.call{{.*}}Destroy

subroutine test1a
  use types
  interface
     function ret_type_t1a
       use types
       type(t1), allocatable :: ret_type_t1a
     end function ret_type_t1a
  end interface
  type(t1), allocatable :: x
  x = ret_type_t1a()
end subroutine test1a
! CHECK-LABEL:   func.func @_QPtest1a() {
! CHECK-NOT: fir.call{{.*}}Destroy
! CHECK:           fir.if %{{.*}} {
! CHECK-NEXT:        fir.freemem %{{.*}} : !fir.heap<!fir.type<_QMtypesTt1{x:f32}>>
! CHECK-NOT: fir.call{{.*}}Destroy
! CHECK:           fir.if %{{.*}} {
! CHECK:             fir.call @_FortranAAllocatableDeallocate({{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-NOT: fir.call{{.*}}Destroy

subroutine test1c
  use types
  interface
     function ret_class_t1
       use types
       class(t1), allocatable :: ret_class_t1
     end function ret_class_t1
  end interface
  type(t1) :: x
  x = ret_class_t1()
end subroutine test1c
! CHECK-LABEL:   func.func @_QPtest1c() {
! CHECK: fir.call @_FortranADestroy
! CHECK:           fir.if %{{.*}} {
! CHECK-NEXT:        fir.freemem %{{.*}} : !fir.heap<!fir.type<_QMtypesTt1{x:f32}>>

subroutine test2
  use types
  interface
     function ret_type_t2
       use types
       type(t2) :: ret_type_t2
     end function ret_type_t2
  end interface
  type(t2) :: x
  x = ret_type_t2()
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2() {
! CHECK: fir.call @_FortranADestroy

subroutine test3
  use types
  interface
     function ret_type_t3
       use types
       type(t3) :: ret_type_t3
     end function ret_type_t3
  end interface
  type(t3) :: x
  x = ret_type_t3()
end subroutine test3
! CHECK-LABEL:   func.func @_QPtest3() {
! CHECK-NOT: fir.call{{.*}}Destroy

subroutine test4
  use types
  interface
     function ret_type_t4
       use types
       type(t4) :: ret_type_t4
     end function ret_type_t4
  end interface
  type(t4) :: x
  x = ret_type_t4()
end subroutine test4
! CHECK-LABEL:   func.func @_QPtest4() {
! CHECK-NOT: fir.call{{.*}}Destroy

subroutine test5
  use types
  interface
     function ret_type_t5
       use types
       type(t5) :: ret_type_t5
     end function ret_type_t5
  end interface
  type(t5) :: x
  x = ret_type_t5()
end subroutine test5
! CHECK-LABEL:   func.func @_QPtest5() {
! CHECK: fir.call @_FortranADestroy

subroutine test6
  use types
  interface
     function ret_type_t6
       use types
       type(t6) :: ret_type_t6
     end function ret_type_t6
  end interface
  type(t6) :: x
  x = ret_type_t6()
end subroutine test6
! CHECK-LABEL:   func.func @_QPtest6() {
! CHECK: fir.call @_FortranADestroy
! CHECK: fir.call @_FortranADestroy

subroutine test7
  use types
  interface
     function ret_type_t7
       use types
       type(t7) :: ret_type_t7
     end function ret_type_t7
  end interface
  type(t7) :: x
  x = ret_type_t7()
end subroutine test7
! CHECK-LABEL:   func.func @_QPtest7() {
! CHECK-NOT: fir.call{{.*}}Destroy

subroutine test8
  use types
  interface
     function ret_type_t8
       use types
       type(t8) :: ret_type_t8
     end function ret_type_t8
  end interface
  type(t8) :: x
  x = ret_type_t8()
end subroutine test8
! CHECK-LABEL:   func.func @_QPtest8() {
! CHECK: fir.call @_FortranADestroy

subroutine test9
  use types
  interface
     function ret_type_t9
       use types
       type(t9) :: ret_type_t9
     end function ret_type_t9
  end interface
  type(t9) :: x
  x = ret_type_t9()
end subroutine test9
! CHECK-LABEL:   func.func @_QPtest9() {
! CHECK: fir.call @_FortranADestroy
! CHECK: fir.call @_FortranADestroy
