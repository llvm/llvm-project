! Test Lowering into [hl]fir.declare with the storage specification.

! Create a temporary directory for the module output, so that
! the module files do not compete with other LIT tests.
! RUN: rm -fr %t && mkdir -p %t && cd %t
! RUN: bbc -emit-fir %s --module=%t -o - | FileCheck %s --check-prefixes=ALL,FIR
! RUN: bbc -emit-hlfir %s --module=%t -o - | FileCheck %s --check-prefixes=ALL,HLFIR

module data1
  real :: m1(5)
  character*5 :: m2(3)
  common /common1/ m1, m2
end module data1
module data2
  integer :: m3 = 1
  real :: m4(7)
  common /common1/ m3, m4
end module data2
module data3
  real :: x(10)
  character*5 :: y(5)
  common /common2/ x
  equivalence (x(9), y(2))
end module data3

! Test different common1 layouts coming from data1 and data2 modules.
subroutine test1
  use data1
  use data2
end subroutine test1
! ALL-LABEL:     func.func @_QPtest1() {
! HLFIR:           %[[VAL_1:.*]] = fir.address_of(@common1_) : !fir.ref<tuple<i32, !fir.array<31xi8>>>
! HLFIR:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<tuple<i32, !fir.array<31xi8>>>) -> !fir.ref<!fir.array<35xi8>>
! HLFIR:           %[[VAL_3:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<!fir.array<35xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
! HLFIR:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]](%{{.*}}) storage(%[[VAL_2]][0]) {uniq_name = "_QMdata1Em1"} : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>, !fir.ref<!fir.array<35xi8>>) -> (!fir.ref<!fir.array<5xf32>>, !fir.ref<!fir.array<5xf32>>)
! HLFIR:           %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<tuple<i32, !fir.array<31xi8>>>) -> !fir.ref<!fir.array<35xi8>>
! HLFIR:           %[[VAL_10:.*]] = arith.constant 20 : index
! HLFIR:           %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_10]] : (!fir.ref<!fir.array<35xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<3x!fir.char<1,5>>>
! HLFIR:           %[[VAL_13:.*]] = arith.constant 5 : index
! HLFIR:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_12]](%{{.*}}) typeparams %[[VAL_13]] storage(%[[VAL_9]][20]) {uniq_name = "_QMdata1Em2"} : (!fir.ref<!fir.array<3x!fir.char<1,5>>>, !fir.shape<1>, index, !fir.ref<!fir.array<35xi8>>) -> (!fir.ref<!fir.array<3x!fir.char<1,5>>>, !fir.ref<!fir.array<3x!fir.char<1,5>>>)
! HLFIR:           %[[VAL_17:.*]] = fir.address_of(@common1_) : !fir.ref<tuple<i32, !fir.array<31xi8>>>
! HLFIR:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<tuple<i32, !fir.array<31xi8>>>) -> !fir.ref<!fir.array<32xi8>>
! HLFIR:           %[[VAL_19:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_18]], %[[VAL_19]] : (!fir.ref<!fir.array<32xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<i8>) -> !fir.ref<i32>
! HLFIR:           %[[VAL_22:.*]]:2 = hlfir.declare %[[VAL_21]] storage(%[[VAL_18]][0]) {uniq_name = "_QMdata2Em3"} : (!fir.ref<i32>, !fir.ref<!fir.array<32xi8>>) -> (!fir.ref<i32>, !fir.ref<i32>)
! HLFIR:           %[[VAL_23:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<tuple<i32, !fir.array<31xi8>>>) -> !fir.ref<!fir.array<32xi8>>
! HLFIR:           %[[VAL_24:.*]] = arith.constant 4 : index
! HLFIR:           %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_23]], %[[VAL_24]] : (!fir.ref<!fir.array<32xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<7xf32>>
! HLFIR:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_26]](%{{.*}}) storage(%[[VAL_23]][4]) {uniq_name = "_QMdata2Em4"} : (!fir.ref<!fir.array<7xf32>>, !fir.shape<1>, !fir.ref<!fir.array<32xi8>>) -> (!fir.ref<!fir.array<7xf32>>, !fir.ref<!fir.array<7xf32>>)

! FIR:           %[[VAL_1:.*]] = arith.constant 4 : index
! FIR:           %[[VAL_3:.*]] = arith.constant 20 : index
! FIR:           %[[VAL_4:.*]] = arith.constant 5 : index
! FIR:           %[[VAL_5:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_7:.*]] = fir.address_of(@common1_) : !fir.ref<tuple<i32, !fir.array<31xi8>>>
! FIR:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<tuple<i32, !fir.array<31xi8>>>) -> !fir.ref<!fir.array<35xi8>>
! FIR:           %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_5]] : (!fir.ref<!fir.array<35xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5xf32>>
! FIR:           %[[VAL_12:.*]] = fir.declare %[[VAL_10]](%{{.*}}) storage(%[[VAL_8]][0]) {uniq_name = "_QMdata1Em1"} : (!fir.ref<!fir.array<5xf32>>, !fir.shape<1>, !fir.ref<!fir.array<35xi8>>) -> !fir.ref<!fir.array<5xf32>>
! FIR:           %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_3]] : (!fir.ref<!fir.array<35xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<3x!fir.char<1,5>>>
! FIR:           %[[VAL_16:.*]] = fir.declare %[[VAL_14]](%{{.*}}) typeparams %[[VAL_4]] storage(%[[VAL_8]][20]) {uniq_name = "_QMdata1Em2"} : (!fir.ref<!fir.array<3x!fir.char<1,5>>>, !fir.shape<1>, index, !fir.ref<!fir.array<35xi8>>) -> !fir.ref<!fir.array<3x!fir.char<1,5>>>
! FIR:           %[[VAL_17:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<tuple<i32, !fir.array<31xi8>>>) -> !fir.ref<!fir.array<32xi8>>
! FIR:           %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_5]] : (!fir.ref<!fir.array<32xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<i8>) -> !fir.ref<i32>
! FIR:           %[[VAL_20:.*]] = fir.declare %[[VAL_19]] storage(%[[VAL_17]][0]) {uniq_name = "_QMdata2Em3"} : (!fir.ref<i32>, !fir.ref<!fir.array<32xi8>>) -> !fir.ref<i32>
! FIR:           %[[VAL_21:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_1]] : (!fir.ref<!fir.array<32xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<7xf32>>
! FIR:           %[[VAL_24:.*]] = fir.declare %[[VAL_22]](%{{.*}}) storage(%[[VAL_17]][4]) {uniq_name = "_QMdata2Em4"} : (!fir.ref<!fir.array<7xf32>>, !fir.shape<1>, !fir.ref<!fir.array<32xi8>>) -> !fir.ref<!fir.array<7xf32>>

! Test the local common1 (different from the global definition).
subroutine test2
  real :: x
  common /common1/ x
end subroutine test2
! ALL-LABEL:     func.func @_QPtest2() {
! HLFIR:           %[[VAL_1:.*]] = fir.address_of(@common1_) : !fir.ref<tuple<i32, !fir.array<31xi8>>>
! HLFIR:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<tuple<i32, !fir.array<31xi8>>>) -> !fir.ref<!fir.array<4xi8>>
! HLFIR:           %[[VAL_3:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i8>) -> !fir.ref<f32>
! HLFIR:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] storage(%[[VAL_2]][0]) {uniq_name = "_QFtest2Ex"} : (!fir.ref<f32>, !fir.ref<!fir.array<4xi8>>) -> (!fir.ref<f32>, !fir.ref<f32>)

! FIR:           %[[VAL_0:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_2:.*]] = fir.address_of(@common1_) : !fir.ref<tuple<i32, !fir.array<31xi8>>>
! FIR:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<tuple<i32, !fir.array<31xi8>>>) -> !fir.ref<!fir.array<4xi8>>
! FIR:           %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_0]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i8>) -> !fir.ref<f32>
! FIR:           %[[VAL_6:.*]] = fir.declare %[[VAL_5]] storage(%[[VAL_3]][0]) {uniq_name = "_QFtest2Ex"} : (!fir.ref<f32>, !fir.ref<!fir.array<4xi8>>) -> !fir.ref<f32>

! Test common2 with equivalence.
subroutine test3
  use data3
end subroutine test3
! ALL-LABEL:     func.func @_QPtest3() {
! HLFIR:           %[[VAL_1:.*]] = fir.address_of(@common2_) : !fir.ref<!fir.array<52xi8>>
! HLFIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<52xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
! HLFIR:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]](%{{.*}}) storage(%[[VAL_1]][0]) {uniq_name = "_QMdata3Ex"} : (!fir.ptr<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<52xi8>>) -> (!fir.ptr<!fir.array<10xf32>>, !fir.ptr<!fir.array<10xf32>>)
! HLFIR:           %[[VAL_8:.*]] = arith.constant 27 : index
! HLFIR:           %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_8]] : (!fir.ref<!fir.array<52xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<5x!fir.char<1,5>>>
! HLFIR:           %[[VAL_11:.*]] = arith.constant 5 : index
! HLFIR:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_10]](%{{.*}}) typeparams %[[VAL_11]] storage(%[[VAL_1]][27]) {uniq_name = "_QMdata3Ey"} : (!fir.ptr<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index, !fir.ref<!fir.array<52xi8>>) -> (!fir.ptr<!fir.array<5x!fir.char<1,5>>>, !fir.ptr<!fir.array<5x!fir.char<1,5>>>)

! FIR:           %[[VAL_0:.*]] = arith.constant 5 : index
! FIR:           %[[VAL_1:.*]] = arith.constant 27 : index
! FIR:           %[[VAL_3:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_5:.*]] = fir.address_of(@common2_) : !fir.ref<!fir.array<52xi8>>
! FIR:           %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_3]] : (!fir.ref<!fir.array<52xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
! FIR:           %[[VAL_9:.*]] = fir.declare %[[VAL_7]](%{{.*}}) storage(%[[VAL_5]][0]) {uniq_name = "_QMdata3Ex"} : (!fir.ptr<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<52xi8>>) -> !fir.ptr<!fir.array<10xf32>>
! FIR:           %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_1]] : (!fir.ref<!fir.array<52xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<5x!fir.char<1,5>>>
! FIR:           %[[VAL_13:.*]] = fir.declare %[[VAL_11]](%{{.*}}) typeparams %[[VAL_0]] storage(%[[VAL_5]][27]) {uniq_name = "_QMdata3Ey"} : (!fir.ptr<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index, !fir.ref<!fir.array<52xi8>>) -> !fir.ptr<!fir.array<5x!fir.char<1,5>>>

! Test host-assocaited common2 usage.
subroutine test4
  use data3
  call inner
contains
  subroutine inner
    x(9) = 7
    y(5) = '12345'
  end subroutine inner
end subroutine test4
! ALL-LABEL:     func.func private @_QFtest4Pinner() attributes {fir.host_symbol = @_QPtest4, llvm.linkage = #llvm.linkage<internal>} {
! HLFIR:           %[[VAL_1:.*]] = fir.address_of(@common2_) : !fir.ref<!fir.array<52xi8>>
! HLFIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<52xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
! HLFIR:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]](%{{.*}}) storage(%[[VAL_1]][0]) {uniq_name = "_QMdata3Ex"} : (!fir.ptr<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<52xi8>>) -> (!fir.ptr<!fir.array<10xf32>>, !fir.ptr<!fir.array<10xf32>>)
! HLFIR:           %[[VAL_8:.*]] = arith.constant 27 : index
! HLFIR:           %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_8]] : (!fir.ref<!fir.array<52xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<5x!fir.char<1,5>>>
! HLFIR:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_10]](%{{.*}}) typeparams %{{.*}} storage(%[[VAL_1]][27]) {uniq_name = "_QMdata3Ey"} : (!fir.ptr<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index, !fir.ref<!fir.array<52xi8>>) -> (!fir.ptr<!fir.array<5x!fir.char<1,5>>>, !fir.ptr<!fir.array<5x!fir.char<1,5>>>)

! FIR:           %[[VAL_2:.*]] = arith.constant 5 : index
! FIR:           %[[VAL_3:.*]] = arith.constant 27 : index
! FIR:           %[[VAL_5:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_7:.*]] = fir.address_of(@common2_) : !fir.ref<!fir.array<52xi8>>
! FIR:           %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_5]] : (!fir.ref<!fir.array<52xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
! FIR:           %[[VAL_11:.*]] = fir.declare %[[VAL_9]](%{{.*}}) storage(%[[VAL_7]][0]) {uniq_name = "_QMdata3Ex"} : (!fir.ptr<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<52xi8>>) -> !fir.ptr<!fir.array<10xf32>>
! FIR:           %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_3]] : (!fir.ref<!fir.array<52xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<5x!fir.char<1,5>>>
! FIR:           %[[VAL_14:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! FIR:           %[[VAL_15:.*]] = fir.declare %[[VAL_13]](%{{.*}}) typeparams %[[VAL_2]] storage(%[[VAL_7]][27]) {uniq_name = "_QMdata3Ey"} : (!fir.ptr<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index, !fir.ref<!fir.array<52xi8>>) -> !fir.ptr<!fir.array<5x!fir.char<1,5>>>

! Test local equivalence.
subroutine test5
  real :: x(10), y(10)
  equivalence (x(5), y(7))
end subroutine test5
! ALL-LABEL:     func.func @_QPtest5() {
! HLFIR:           %[[VAL_1:.*]] = fir.alloca !fir.array<48xi8> {uniq_name = "_QFtest5Ex"}
! HLFIR:           %[[VAL_2:.*]] = arith.constant 8 : index
! HLFIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<48xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
! HLFIR:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]](%{{.*}}) storage(%[[VAL_1]][8]) {uniq_name = "_QFtest5Ex"} : (!fir.ptr<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<48xi8>>) -> (!fir.ptr<!fir.array<10xf32>>, !fir.ptr<!fir.array<10xf32>>)
! HLFIR:           %[[VAL_8:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_8]] : (!fir.ref<!fir.array<48xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
! HLFIR:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_10]](%{{.*}}) storage(%[[VAL_1]][0]) {uniq_name = "_QFtest5Ey"} : (!fir.ptr<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<48xi8>>) -> (!fir.ptr<!fir.array<10xf32>>, !fir.ptr<!fir.array<10xf32>>)

! FIR:           %[[VAL_0:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_2:.*]] = arith.constant 8 : index
! FIR:           %[[VAL_4:.*]] = fir.alloca !fir.array<48xi8> {uniq_name = "_QFtest5Ex"}
! FIR:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_2]] : (!fir.ref<!fir.array<48xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
! FIR:           %[[VAL_8:.*]] = fir.declare %[[VAL_6]](%{{.*}}) storage(%[[VAL_4]][8]) {uniq_name = "_QFtest5Ex"} : (!fir.ptr<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<48xi8>>) -> !fir.ptr<!fir.array<10xf32>>
! FIR:           %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_0]] : (!fir.ref<!fir.array<48xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
! FIR:           %[[VAL_11:.*]] = fir.declare %[[VAL_10]](%{{.*}}) storage(%[[VAL_4]][0]) {uniq_name = "_QFtest5Ey"} : (!fir.ptr<!fir.array<10xf32>>, !fir.shape<1>, !fir.ref<!fir.array<48xi8>>) -> !fir.ptr<!fir.array<10xf32>>

! Test equivalence with saved symbol.
subroutine test6
  real(2), save :: x = 1.0_2
  integer :: y(2)
  equivalence (x, y(2))
end subroutine test6
! ALL-LABEL:     func.func @_QPtest6() {
! HLFIR:           %[[VAL_1:.*]] = fir.address_of(@_QFtest6Ex) : !fir.ref<!fir.array<4xi16>>
! HLFIR:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.array<4xi16>>) -> !fir.ref<!fir.array<8xi8>>
! HLFIR:           %[[VAL_3:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<4xi16>>
! HLFIR:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_5]](%{{.*}}) storage(%[[VAL_2]][0]) {uniq_name = "_QFtest6E.f18.0"} : (!fir.ptr<!fir.array<4xi16>>, !fir.shape<1>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<!fir.array<4xi16>>, !fir.ptr<!fir.array<4xi16>>)
! HLFIR:           %[[VAL_9:.*]] = arith.constant 4 : index
! HLFIR:           %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_9]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<i8>) -> !fir.ptr<f16>
! HLFIR:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] storage(%[[VAL_2]][4]) {uniq_name = "_QFtest6Ex"} : (!fir.ptr<f16>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<f16>, !fir.ptr<f16>)
! HLFIR:           %[[VAL_13:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_13]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xi32>>
! HLFIR:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_15]](%{{.*}}) storage(%[[VAL_2]][0]) {uniq_name = "_QFtest6Ey"} : (!fir.ptr<!fir.array<2xi32>>, !fir.shape<1>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ptr<!fir.array<2xi32>>, !fir.ptr<!fir.array<2xi32>>)

! FIR:           %[[VAL_1:.*]] = arith.constant 4 : index
! FIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_4:.*]] = fir.address_of(@_QFtest6Ex) : !fir.ref<!fir.array<4xi16>>
! FIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.array<4xi16>>) -> !fir.ref<!fir.array<8xi8>>
! FIR:           %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_2]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<4xi16>>
! FIR:           %[[VAL_9:.*]] = fir.declare %[[VAL_7]](%{{.*}}) storage(%[[VAL_5]][0]) {uniq_name = "_QFtest6E.f18.0"} : (!fir.ptr<!fir.array<4xi16>>, !fir.shape<1>, !fir.ref<!fir.array<8xi8>>) -> !fir.ptr<!fir.array<4xi16>>
! FIR:           %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_1]] : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<i8>) -> !fir.ptr<f16>
! FIR:           %[[VAL_12:.*]] = fir.declare %[[VAL_11]] storage(%[[VAL_5]][4]) {uniq_name = "_QFtest6Ex"} : (!fir.ptr<f16>, !fir.ref<!fir.array<8xi8>>) -> !fir.ptr<f16>
! FIR:           %[[VAL_13:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2xi32>>
! FIR:           %[[VAL_15:.*]] = fir.declare %[[VAL_13]](%{{.*}}) storage(%[[VAL_5]][0]) {uniq_name = "_QFtest6Ey"} : (!fir.ptr<!fir.array<2xi32>>, !fir.shape<1>, !fir.ref<!fir.array<8xi8>>) -> !fir.ptr<!fir.array<2xi32>>

! Test host-associated equivalence.
! TODO: it makes more sense to me to pass only the storage address
! via the host-associated tuple, and then declare x and y inside
! inner via the storage. This gives more information about
! the overlapping of x and y inside inner, which might be useful
! at some point.
subroutine test7
  integer :: x(10), y(7)
  equivalence (x(1), y(7))
  call inner
contains
  subroutine inner
    x(1) = 1
    y(7) = 1
  end subroutine inner
end subroutine test7
! ALL-LABEL:     func.func private @_QFtest7Pinner(
! ALL-SAME:        %[[ARG0:.*]]: !fir.ref<tuple<!fir.box<!fir.array<10xi32>>, !fir.box<!fir.array<7xi32>>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QPtest7, llvm.linkage = #llvm.linkage<internal>} {
! HLFIR:           %[[VAL_1:.*]] = arith.constant 0 : i32
! HLFIR:           %[[VAL_2:.*]] = fir.coordinate_of %[[ARG0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>, !fir.box<!fir.array<7xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
! HLFIR:           %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<10xi32>>>
! HLFIR:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! HLFIR:           %[[VAL_5:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_4]](%{{[^)]*}}) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest7Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! HLFIR:           %[[VAL_9:.*]] = arith.constant 1 : i32
! HLFIR:           %[[VAL_10:.*]] = fir.coordinate_of %[[ARG0]], %[[VAL_9]] : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>, !fir.box<!fir.array<7xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<7xi32>>>
! HLFIR:           %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<!fir.box<!fir.array<7xi32>>>
! HLFIR:           %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.array<7xi32>>) -> !fir.ref<!fir.array<7xi32>>
! HLFIR:           %[[VAL_13:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_12]](%{{[^)]*}}) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest7Ey"} : (!fir.ref<!fir.array<7xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<7xi32>>, !fir.ref<!fir.array<7xi32>>)

! FIR:           %[[VAL_2:.*]] = arith.constant 1 : i32
! FIR:           %[[VAL_3:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_4:.*]] = arith.constant 0 : i32
! FIR:           %[[VAL_6:.*]] = fir.coordinate_of %[[ARG0]], %[[VAL_4]] : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>, !fir.box<!fir.array<7xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<10xi32>>>
! FIR:           %[[VAL_7:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.array<10xi32>>>
! FIR:           %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! FIR:           %[[VAL_11:.*]] = fir.declare %[[VAL_8]](%{{.*}}) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest7Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
! FIR:           %[[VAL_12:.*]] = fir.coordinate_of %[[ARG0]], %[[VAL_2]] : (!fir.ref<tuple<!fir.box<!fir.array<10xi32>>, !fir.box<!fir.array<7xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<7xi32>>>
! FIR:           %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<!fir.box<!fir.array<7xi32>>>
! FIR:           %[[VAL_14:.*]] = fir.box_addr %[[VAL_13]] : (!fir.box<!fir.array<7xi32>>) -> !fir.ref<!fir.array<7xi32>>
! FIR:           %[[VAL_17:.*]] = fir.declare %[[VAL_14]](%{{.*}}) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest7Ey"} : (!fir.ref<!fir.array<7xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<7xi32>>
