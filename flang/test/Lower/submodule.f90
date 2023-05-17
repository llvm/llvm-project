! RUN: bbc -emit-fir %s -o - | FileCheck %s

module mm
  integer :: vv = 20
  interface
    module function ff1(nn)
      integer ff1(nn+1)
    end function ff1
    module function ff2(nn)
      integer ff2(nn+2)
    end function ff2
    module function ff3(nn)
      integer ff3(nn+3)
    end function ff3
  end interface
end module mm

submodule(mm) ss1
  integer :: ww = 20
  interface
    module function fff(nn)
      integer fff
    end function fff
  end interface
contains
  ! CHECK-LABEL: func @_QMmmPff2
  ! CHECK:     %[[V_0:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK:     %[[V_1:[0-9]+]] = arith.addi %[[V_0]], %c2{{.*}} : i32
  ! CHECK:     %[[V_2:[0-9]+]] = fir.convert %[[V_1]] : (i32) -> i64
  ! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[V_2]] : (i64) -> index
  ! CHECK:     %[[V_4:[0-9]+]] = arith.cmpi sgt, %[[V_3]], %c0{{.*}} : index
  ! CHECK:     %[[V_5:[0-9]+]] = arith.select %[[V_4]], %[[V_3]], %c0{{.*}} : index
  ! CHECK:     %[[V_6:[0-9]+]] = fir.alloca !fir.array<?xi32>, %[[V_5]] {bindc_name = "ff2", uniq_name = "_QMmmSss1Fff2Eff2"}
  ! CHECK:     %[[V_7:[0-9]+]] = fir.shape %[[V_5]] : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_8:[0-9]+]] = fir.array_load %[[V_6]](%[[V_7]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
  ! CHECK:     %[[V_9:[0-9]+]] = fir.call @_QMmmSss1Pfff(%arg0) {{.*}} : (!fir.ref<i32>) -> i32
  ! CHECK:     %[[V_10:[0-9]+]] = arith.subi %[[V_5]], %c1{{.*}} : index
  ! CHECK:     %[[V_11:[0-9]+]] = fir.do_loop %arg1 = %c0{{.*}} to %[[V_10]] step %c1{{.*}} unordered iter_args(%arg2 = %[[V_8]]) -> (!fir.array<?xi32>) {
  ! CHECK:       %[[V_13:[0-9]+]] = fir.array_update %arg2, %[[V_9]], %arg1 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
  ! CHECK:       fir.result %[[V_13]] : !fir.array<?xi32>
  ! CHECK:     }
  ! CHECK:     fir.array_merge_store %[[V_8]], %[[V_11]] to %[[V_6]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>
  ! CHECK:     %[[V_12:[0-9]+]] = fir.load %[[V_6]] : !fir.ref<!fir.array<?xi32>>
  ! CHECK:     return %[[V_12]] : !fir.array<?xi32>
  ! CHECK:   }
  module procedure ff2
    ff2 = fff(nn)
  end procedure ff2
end submodule ss1

submodule(mm:ss1) ss2
contains
  ! CHECK-LABEL: func @_QMmmPff1
  ! CHECK-DAG: %[[V_0:[0-9]+]] = fir.address_of(@_QMmmEvv) : !fir.ref<i32>
  ! CHECK-DAG: %[[V_1:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK:     %[[V_2:[0-9]+]] = arith.addi %[[V_1]], %c1{{.*}} : i32
  ! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[V_2]] : (i32) -> i64
  ! CHECK:     %[[V_4:[0-9]+]] = fir.convert %[[V_3]] : (i64) -> index
  ! CHECK:     %[[V_5:[0-9]+]] = arith.cmpi sgt, %[[V_4]], %c0{{.*}} : index
  ! CHECK:     %[[V_6:[0-9]+]] = arith.select %[[V_5]], %[[V_4]], %c0{{.*}} : index
  ! CHECK:     %[[V_7:[0-9]+]] = fir.alloca !fir.array<?xi32>, %[[V_6]] {bindc_name = "ff1", uniq_name = "_QMmmSss1Sss2Fff1Eff1"}
  ! CHECK:     %[[V_8:[0-9]+]] = fir.shape %[[V_6]] : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_9:[0-9]+]] = fir.array_load %[[V_7]](%[[V_8]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
  ! CHECK:     %[[V_10:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
  ! CHECK:     %[[V_11:[0-9]+]] = arith.addi %[[V_10]], %c2{{.*}} : i32
  ! CHECK:     %[[V_12:[0-9]+]] = arith.subi %[[V_6]], %c1{{.*}} : index
  ! CHECK:     %[[V_13:[0-9]+]] = fir.do_loop %arg1 = %c0{{.*}} to %[[V_12]] step %c1{{.*}} unordered iter_args(%arg2 = %[[V_9]]) -> (!fir.array<?xi32>) {
  ! CHECK:       %[[V_15:[0-9]+]] = fir.array_update %arg2, %[[V_11]], %arg1 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
  ! CHECK:       fir.result %[[V_15]] : !fir.array<?xi32>
  ! CHECK:     }
  ! CHECK:     fir.array_merge_store %[[V_9]], %[[V_13]] to %[[V_7]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>
  ! CHECK:     %[[V_14:[0-9]+]] = fir.load %[[V_7]] : !fir.ref<!fir.array<?xi32>>
  ! CHECK:     return %[[V_14]] : !fir.array<?xi32>
  ! CHECK:   }
  module function ff1(nn)
    integer ff1(nn+1)
    ff1 = vv + 2
  end function ff1

  ! CHECK-LABEL: func @_QMmmSss1Pfff
  ! CHECK-DAG: %[[V_0:[0-9]+]] = fir.address_of(@_QMmmSss1Eww) : !fir.ref<i32>
  ! CHECK-DAG: %[[V_1:[0-9]+]] = fir.address_of(@_QMmmEvv) : !fir.ref<i32>
  ! CHECK-DAG: %[[V_2:[0-9]+]] = fir.alloca i32 {bindc_name = "fff", uniq_name = "_QMmmSss1Sss2FfffEfff"}
  ! CHECK-DAG: %[[V_3:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
  ! CHECK-DAG: %[[V_4:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
  ! CHECK:     %[[V_5:[0-9]+]] = arith.addi %[[V_3]], %[[V_4]] : i32
  ! CHECK:     %[[V_6:[0-9]+]] = arith.addi %[[V_5]], %c4{{.*}} : i32
  ! CHECK:     fir.store %[[V_6]] to %[[V_2]] : !fir.ref<i32>
  ! CHECK:     %[[V_7:[0-9]+]] = fir.load %[[V_2]] : !fir.ref<i32>
  ! CHECK:     return %[[V_7]] : i32
  ! CHECK:   }
  module procedure fff
    fff = vv + ww + 4
  end procedure fff
end submodule ss2

submodule(mm) sss
contains
  ! CHECK-LABEL: func @_QMmmPff3
  ! CHECK-DAG: %[[V_0:[0-9]+]] = fir.address_of(@_QMmmEvv) : !fir.ref<i32>
  ! CHECK-DAG: %[[V_1:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK:     %[[V_2:[0-9]+]] = arith.addi %[[V_1]], %c3{{.*}} : i32
  ! CHECK:     %[[V_3:[0-9]+]] = fir.convert %[[V_2]] : (i32) -> i64
  ! CHECK:     %[[V_4:[0-9]+]] = fir.convert %[[V_3]] : (i64) -> index
  ! CHECK:     %[[V_5:[0-9]+]] = arith.cmpi sgt, %[[V_4]], %c0{{.*}} : index
  ! CHECK:     %[[V_6:[0-9]+]] = arith.select %[[V_5]], %[[V_4]], %c0{{.*}} : index
  ! CHECK:     %[[V_7:[0-9]+]] = fir.alloca !fir.array<?xi32>, %[[V_6]] {bindc_name = "ff3", uniq_name = "_QMmmSsssFff3Eff3"}
  ! CHECK:     %[[V_8:[0-9]+]] = fir.shape %[[V_6]] : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_9:[0-9]+]] = fir.array_load %[[V_7]](%[[V_8]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
  ! CHECK-DAG: %[[V_10:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[V_11:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
  ! CHECK:     %[[V_12:[0-9]+]] = arith.muli %[[V_10]], %[[V_11]] : i32
  ! CHECK:     %[[V_13:[0-9]+]] = arith.addi %[[V_12]], %c6{{.*}} : i32
  ! CHECK:     %[[V_14:[0-9]+]] = arith.subi %[[V_6]], %c1{{.*}} : index
  ! CHECK:     %[[V_15:[0-9]+]] = fir.do_loop %arg1 = %c0{{.*}} to %[[V_14]] step %c1{{.*}} unordered iter_args(%arg2 = %[[V_9]]) -> (!fir.array<?xi32>) {
  ! CHECK:       %[[V_17:[0-9]+]] = fir.array_update %arg2, %[[V_13]], %arg1 : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
  ! CHECK:       fir.result %[[V_17]] : !fir.array<?xi32>
  ! CHECK:     }
  ! CHECK:     fir.array_merge_store %[[V_9]], %[[V_15]] to %[[V_7]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>
  ! CHECK:     %[[V_16:[0-9]+]] = fir.load %[[V_7]] : !fir.ref<!fir.array<?xi32>>
  ! CHECK:     return %[[V_16]] : !fir.array<?xi32>
  ! CHECK:   }
  module function ff3(nn)
    integer ff3(nn+3)
    ff3 = nn*vv + 6
  end function ff3
end submodule sss

! CHECK-LABEL: func @_QQmain
program pp
  use mm
  ! CHECK:     fir.call @_QMmmPff1(%{{[0-9]+}}) {{.*}} : (!fir.ref<i32>) -> !fir.array<?xi32>
  print*, ff1(1) ! expect: 22 22
  ! CHECK:     fir.call @_QMmmPff2(%{{[0-9]+}}) {{.*}} : (!fir.ref<i32>) -> !fir.array<?xi32>
  print*, ff2(2) ! expect: 44 44 44 44
  ! CHECK:     fir.call @_QMmmPff3(%{{[0-9]+}}) {{.*}} : (!fir.ref<i32>) -> !fir.array<?xi32>
  print*, ff3(3) ! expect: 66 66 66 66 66 66
end program pp
