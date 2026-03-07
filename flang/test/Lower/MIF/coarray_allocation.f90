! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program alloc_test
  type :: my_type2
    integer, allocatable :: co[:]
  end type
  
  type :: my_type
    integer :: x
    integer, allocatable :: y(:)
    type(my_type2) :: z
  end type
  
  ! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFEa) : !fir.ref<i32>
  ! CHECK: mif.alloc_coarray %[[VAL_1]] {lcobounds = array<i64: 1, 1>, ucobounds = array<i64: 2, -1>, uniq_name = "_QFEa"} : (!fir.ref<i32>) -> ()
  
  ! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[ADDR_1:.*]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa2"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
  
  integer :: a[2, *]
  ! CHECK: %[[VAL_2:.*]] = fir.address_of(@_QFEb) : !fir.ref<f32>
  ! CHECK: mif.alloc_coarray %[[VAL_2]] {lcobounds = array<i64: 3, 1, 1>, ucobounds = array<i64: 4, 5, -1>, uniq_name = "_QFEb"} : (!fir.ref<f32>) -> ()
  
  ! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[ADDR_2:.*]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEb2"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
  
  real :: b[3:4, 5, *]
  ! CHECK: %[[VAL_3:.*]] = fir.address_of(@_QFEc) : !fir.ref<!fir.char<1,10>>
  ! CHECK: mif.alloc_coarray %[[VAL_3]] {lcobounds = array<i64: 1>, ucobounds = array<i64: -1>, uniq_name = "_QFEc"} : (!fir.ref<!fir.char<1,10>>) -> ()
  
  ! CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[ADDR_3:.*]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEc2"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>)
  character(len=10) :: c[*]
  type(my_type) :: d

  real, allocatable :: b2[:,:,:]
  character(len=:), allocatable :: c2(:)[:]
  integer, allocatable :: a2[:,:]
  
  ! CHECK: %[[VAL_7:.*]] = fir.absent !fir.box<none>
  ! CHECK: mif.alloc_coarray %[[VAL_4]]#0 errmsg %[[VAL_7]] {lcobounds = array<i64: 1, 1>, ucobounds = array<i64: 2, -1>, uniq_name = "_QFEa2"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<none>) -> ()
  allocate(a2[2,*])
  
  ! CHECK: %[[VAL_8:.*]] = fir.absent !fir.box<none>
  ! CHECK: mif.alloc_coarray %[[VAL_5]]#0 errmsg %[[VAL_8]] {lcobounds = array<i64: 3, 1, 1>, ucobounds = array<i64: 4, 5, -1>, uniq_name = "_QFEb2"} : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.box<none>) -> ()
  allocate(b2[3:4, 5, *])
  
  ! CHECK: %[[VAL_9:.*]] = fir.absent !fir.box<none>
  ! CHECK: mif.alloc_coarray %[[VAL_6]]#0 errmsg %[[VAL_9]] {lcobounds = array<i64: 1>, ucobounds = array<i64: -1>, uniq_name = "_QFEc2"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.box<none>) -> ()
  allocate(character(100) :: c2(5)[*])
  
  ! CHECK: %[[VAL_10:.*]] = fir.absent !fir.box<none>
  ! CHECK: %[[VAL_12:.*]] = hlfir.designate %[[VAL_11:.*]]{"co"}
  ! CHECK:  mif.alloc_coarray %[[VAL_12]] errmsg %[[VAL_10]] {lcobounds = array<i64: 1>, ucobounds = array<i64: -1>, uniq_name = "_QFEd.z.co"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<none>) -> ()
  allocate(d%z%co[*])

  ! CHECK: mif.dealloc_coarray %[[VAL_4]]#0 stat %[[STAT:.*]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<i32>, !fir.box<none>) -> ()
  ! CHECK: mif.dealloc_coarray %[[VAL_5]]#0 stat %[[STAT:.*]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<i32>, !fir.box<none>) -> ()
  ! CHECK: mif.dealloc_coarray %[[VAL_6]]#0 stat %[[STAT:.*]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<i32>, !fir.box<none>) -> ()
  deallocate(a2, b2, c2)
  
  ! CHECK: %[[VAL_14:.*]] = hlfir.designate %[[VAL_13:.*]]{"co"} {{.*}}
  ! CHECK: mif.dealloc_coarray %[[VAL_14]] stat %[[STAT:.*]] errmsg %[[ERRMSG:.*]] : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<i32>, !fir.box<none>) -> ()
  deallocate(d%z%co)
  
end program 
