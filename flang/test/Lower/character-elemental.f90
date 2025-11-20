! RUN: bbc -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: substring_main
subroutine substring_main
  character*7 :: string(2) = ['12     ', '12     ']
  integer :: result(2)
  integer :: ival
interface
  elemental function inner(arg)
    character(len=*), intent(in) :: arg
    integer :: inner
  end function inner
end interface

  ival = 1
  ! CHECK: %[[a0:.*]] = fir.alloca i32 {bindc_name = "ival", uniq_name = "_QFsubstring_mainEival"}
  ! CHECK: %[[a2:.*]] = fir.address_of(@_QFsubstring_mainEstring) : !fir.ref<!fir.array<2x!fir.char<1,7>>>
  ! CHECK: fir.store {{.*}} to %[[a0]] : !fir.ref<i32>
  ! CHECK: %[[a3:.*]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[a4:.*]] = fir.slice {{.*}}, {{.*}}, {{.*}} : (index, index, index) -> !fir.slice<1>
  ! CHECK: br ^bb1({{.*}}, {{.*}} : index, index)
  ! CHECK: ^bb1(%[[a5:.*]]: index, %[[a6:.*]]: index):  // 2 preds: ^bb0, ^bb2
  ! CHECK: %[[a7:.*]] = arith.cmpi sgt, %[[a6]], {{.*}} : index
  ! CHECK: cond_br %[[a7]], ^bb2, ^bb3
  ! CHECK: ^bb2:  // pred: ^bb1
  ! CHECK: %[[a8:.*]] = arith.addi %[[a5]], {{.*}} : index
  ! CHECK: %[[a9:.*]] = fir.array_coor %[[a2]](%[[a3]]) [%[[a4]]] %[[a8]] : (!fir.ref<!fir.array<2x!fir.char<1,7>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,7>>
  ! CHECK: %[[a10:.*]] = fir.load %[[a0]] : !fir.ref<i32>
  ! CHECK: %[[a11:.*]] = fir.convert %[[a10]] : (i32) -> index
  ! CHECK: %[[a12:.*]] = arith.subi %[[a11]], {{.*}} : index
  ! CHECK: %[[a13:.*]] = fir.convert %[[a9]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<!fir.array<7x!fir.char<1>>>
  ! CHECK: %[[a14:.*]] = fir.coordinate_of %[[a13]], %[[a12]] : (!fir.ref<!fir.array<7x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[a15:.*]] = fir.convert %[[a14]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[a16:.*]] = fir.emboxchar %[[a15]], {{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[a17:.*]] = fir.call @_QPinner(%[[a16]]) {{.*}}: (!fir.boxchar<1>) -> i32
  result = inner(string(1:2)(ival:ival))
  print *, result
end subroutine substring_main
