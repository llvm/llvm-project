! Test lowering of internal procedures returning arrays or characters.
! This test allocation on the caller side of the results that may depend on
! host associated symbols.
! RUN: bbc %s -o - | FileCheck %s

module some_module
 integer :: n_module
end module

! Test host calling array internal procedure.
! Result depends on host variable.
! CHECK-LABEL: func @_QPhost1
subroutine host1()
  implicit none
  integer :: n
! CHECK:  %[[VAL_1:.*]] = fir.alloca i32
  call takes_array(return_array())
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_5]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_5]], %{{.*}} : index
! CHECK:  %[[VAL_6:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
contains
  function return_array()
    real :: return_array(n)
  end function
end subroutine

! Test host calling array internal procedure.
! Result depends on module variable with the use statement inside the host.
! CHECK-LABEL: func @_QPhost2
subroutine host2()
  use :: some_module
  call takes_array(return_array())
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QMsome_moduleEn_module) : !fir.ref<i32>
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_2]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_2]], %{{.*}} : index
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
contains
  function return_array()
    real :: return_array(n_module)
  end function
end subroutine

! Test host calling array internal procedure.
! Result depends on module variable with the use statement inside the internal procedure.
! CHECK-LABEL: func @_QPhost3
subroutine host3()
  call takes_array(return_array())
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QMsome_moduleEn_module) : !fir.ref<i32>
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_2]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_2]], %{{.*}} : index
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
contains
  function return_array()
    use :: some_module
    real :: return_array(n_module)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on host variable not directly used in A.
subroutine host4()
  implicit none
  integer :: n
  call internal_proc_a()
contains
! CHECK-LABEL: func @_QFhost4Pinternal_proc_a
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) attributes {fir.internal_proc} {
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_5]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_5]], %{{.*}} : index
! CHECK:  %[[VAL_6:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
  end subroutine
  function return_array()
    real :: return_array(n)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on module variable with use statement in the host.
subroutine host5()
  use :: some_module
  implicit none
  call internal_proc_a()
contains
! CHECK-LABEL: func @_QFhost5Pinternal_proc_a() {
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QMsome_moduleEn_module) : !fir.ref<i32>
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_2]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_2]], %{{.*}} : index
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
  end subroutine
  function return_array()
    real :: return_array(n_module)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on module variable with use statement in B.
subroutine host6()
  implicit none
  call internal_proc_a()
contains
! CHECK-LABEL: func @_QFhost6Pinternal_proc_a
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QMsome_moduleEn_module) : !fir.ref<i32>
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_2]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_2]], %{{.*}} : index
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
  end subroutine
  function return_array()
    use :: some_module
    real :: return_array(n_module)
  end function
end subroutine

! Test host calling array internal procedure.
! Result depends on a common block variable declared in the host.
! CHECK-LABEL: func @_QPhost7
subroutine host7()
  implicit none
  integer :: n_common
  common /mycom/ n_common
  call takes_array(return_array())
! CHECK:  %[[VAL_0:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QCmycom) : !fir.ref<!fir.array<4xi8>>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i8>) -> !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_9]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_9]], %{{.*}} : index
! CHECK:  %[[VAL_10:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
contains
  function return_array()
    real :: return_array(n_common)
  end function
end subroutine

! Test host calling array internal procedure.
! Result depends on a common block variable declared in the internal procedure.
! CHECK-LABEL: func @_QPhost8
subroutine host8()
  implicit none
  call takes_array(return_array())
! CHECK:  %[[VAL_0:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QCmycom) : !fir.ref<!fir.array<4xi8>>
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_6]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_6]], %{{.*}} : index
! CHECK:  %[[VAL_7:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
contains
  function return_array()
    integer :: n_common
    common /mycom/ n_common
    real :: return_array(n_common)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on a common block variable declared in the host.
subroutine host9()
  implicit none
  integer :: n_common
  common /mycom/ n_common
  call internal_proc_a()
contains
! CHECK-LABEL: func @_QFhost9Pinternal_proc_a
  subroutine internal_proc_a()
! CHECK:  %[[VAL_0:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QCmycom) : !fir.ref<!fir.array<4xi8>>
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:  %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_0]] : index
! CHECK:  %[[VAL_8:.*]] = arith.select %[[VAL_7]], %[[VAL_6]], %[[VAL_0]] : index
! CHECK:  %[[VAL_10:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_8]] {bindc_name = ".result"}
    call takes_array(return_array())
  end subroutine
  function return_array()
    use :: some_module
    real :: return_array(n_common)
  end function
end subroutine

! Test internal procedure A calling array internal procedure B.
! Result depends on a common block variable declared in B.
subroutine host10()
  implicit none
  call internal_proc_a()
contains
! CHECK-LABEL: func @_QFhost10Pinternal_proc_a
  subroutine internal_proc_a()
    call takes_array(return_array())
! CHECK:  %[[VAL_0:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QCmycom) : !fir.ref<!fir.array<4xi8>>
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:  %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_6]], %{{.*}} : index
! CHECK:  %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_6]], %{{.*}} : index
! CHECK:  %[[VAL_7:.*]] = fir.alloca !fir.array<?xf32>, %[[SELECT]] {bindc_name = ".result"}
  end subroutine
  function return_array()
    integer :: n_common
    common /mycom/ n_common
    real :: return_array(n_common)
  end function
end subroutine


! Test call to a function returning an array where the interface is use
! associated from a module.
module define_interface
contains
function foo()
  real :: foo(100)
  foo = 42
end function
end module
! CHECK-LABEL: func @_QPtest_call_to_used_interface(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
subroutine test_call_to_used_interface(dummy_proc)
  use define_interface
  procedure(foo) :: dummy_proc
  call takes_array(dummy_proc())
! CHECK:  %[[VAL_1:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = ".result"}
! CHECK:  %[[VAL_3:.*]] = fir.call @llvm.stacksave() {{.*}}: () -> !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> !fir.array<100xf32>)
! CHECK:  %[[VAL_6:.*]] = fir.call %[[VAL_5]]() {{.*}}: () -> !fir.array<100xf32>
! CHECK:  fir.save_result %[[VAL_6]] to %[[VAL_2]](%[[VAL_4]]) : !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>, !fir.shape<1>
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  fir.call @_QPtakes_array(%[[VAL_7]]) {{.*}}: (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:  fir.call @llvm.stackrestore(%[[VAL_3]]) {{.*}}: (!fir.ref<i8>) -> ()
end subroutine
