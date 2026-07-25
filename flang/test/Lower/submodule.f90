! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

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
  ! CHECK:     %[[V_0:.*]]:2 = hlfir.declare %arg0 {{.*}} {uniq_name = "_QMmmSss1Fff2Enn"}
  ! CHECK:     %[[V_1:.*]] = fir.load %[[V_0]]#0 : !fir.ref<i32>
  ! CHECK:     %[[V_2:.*]] = arith.addi %[[V_1]], %c2{{.*}} : i32
  ! CHECK:     %[[V_3:.*]] = fir.convert %[[V_2]] : (i32) -> i64
  ! CHECK:     %[[V_4:.*]] = fir.convert %[[V_3]] : (i64) -> index
  ! CHECK:     %[[V_5:.*]] = arith.cmpi sgt, %[[V_4]], %c0{{.*}} : index
  ! CHECK:     %[[V_6:.*]] = arith.select %[[V_5]], %[[V_4]], %c0{{.*}} : index
  ! CHECK:     %[[V_7:.*]] = fir.alloca !fir.array<?xi32>, %[[V_6]] {bindc_name = "ff2", uniq_name = "_QMmmSss1Fff2Eff2"}
  ! CHECK:     %[[V_8:.*]] = fir.shape %[[V_6]] : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_9:.*]]:2 = hlfir.declare %[[V_7]](%[[V_8]]) {uniq_name = "_QMmmSss1Fff2Eff2"}
  ! CHECK:     %[[V_10:.*]] = fir.call @_QMmmSss1Pfff(%[[V_0]]#0) {{.*}} : (!fir.ref<i32>) -> i32
  ! CHECK:     hlfir.assign %[[V_10]] to %[[V_9]]#0 : i32, !fir.box<!fir.array<?xi32>>
  ! CHECK:     %[[V_11:.*]] = fir.load %[[V_9]]#1 : !fir.ref<!fir.array<?xi32>>
  ! CHECK:     return %[[V_11]] : !fir.array<?xi32>
  ! CHECK:   }
  module procedure ff2
    ff2 = fff(nn)
  end procedure ff2
end submodule ss1

submodule(mm:ss1) ss2
contains
  ! CHECK-LABEL: func @_QMmmPff1
  ! CHECK-DAG: %[[V_0:.*]] = fir.address_of(@_QMmmEvv) : !fir.ref<i32>
  ! CHECK-DAG: %[[V_1:.*]]:2 = hlfir.declare %[[V_0]] {uniq_name = "_QMmmEvv"}
  ! CHECK-DAG: %[[V_2:.*]]:2 = hlfir.declare %arg0 {{.*}} {uniq_name = "_QMmmSss1Sss2Fff1Enn"}
  ! CHECK:     %[[V_3:.*]] = fir.load %[[V_2]]#0 : !fir.ref<i32>
  ! CHECK:     %[[V_4:.*]] = arith.addi %[[V_3]], %c1{{.*}} : i32
  ! CHECK:     %[[V_5:.*]] = fir.convert %[[V_4]] : (i32) -> i64
  ! CHECK:     %[[V_6:.*]] = fir.convert %[[V_5]] : (i64) -> index
  ! CHECK:     %[[V_7:.*]] = arith.cmpi sgt, %[[V_6]], %c0{{.*}} : index
  ! CHECK:     %[[V_8:.*]] = arith.select %[[V_7]], %[[V_6]], %c0{{.*}} : index
  ! CHECK:     %[[V_9:.*]] = fir.alloca !fir.array<?xi32>, %[[V_8]] {bindc_name = "ff1", uniq_name = "_QMmmSss1Sss2Fff1Eff1"}
  ! CHECK:     %[[V_10:.*]] = fir.shape %[[V_8]] : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_11:.*]]:2 = hlfir.declare %[[V_9]](%[[V_10]]) {uniq_name = "_QMmmSss1Sss2Fff1Eff1"}
  ! CHECK:     %[[V_12:.*]] = fir.load %[[V_1]]#0 : !fir.ref<i32>
  ! CHECK:     %[[V_13:.*]] = arith.addi %[[V_12]], %c2{{.*}} : i32
  ! CHECK:     hlfir.assign %[[V_13]] to %[[V_11]]#0 : i32, !fir.box<!fir.array<?xi32>>
  ! CHECK:     %[[V_14:.*]] = fir.load %[[V_11]]#1 : !fir.ref<!fir.array<?xi32>>
  ! CHECK:     return %[[V_14]] : !fir.array<?xi32>
  ! CHECK:   }
  module function ff1(nn)
    integer ff1(nn+1)
    ff1 = vv + 2
  end function ff1

  ! CHECK-LABEL: func @_QMmmSss1Pfff
  ! CHECK-DAG: %[[V_0:.*]] = fir.address_of(@_QMmmEvv) : !fir.ref<i32>
  ! CHECK-DAG: %[[V_1:.*]]:2 = hlfir.declare %[[V_0]] {uniq_name = "_QMmmEvv"}
  ! CHECK-DAG: %[[V_2:.*]] = fir.address_of(@_QMmmSss1Eww) : !fir.ref<i32>
  ! CHECK-DAG: %[[V_3:.*]]:2 = hlfir.declare %[[V_2]] {uniq_name = "_QMmmSss1Eww"}
  ! CHECK-DAG: %[[V_4:.*]] = fir.alloca i32 {bindc_name = "fff", uniq_name = "_QMmmSss1Sss2FfffEfff"}
  ! CHECK-DAG: %[[V_5:.*]]:2 = hlfir.declare %[[V_4]] {uniq_name = "_QMmmSss1Sss2FfffEfff"}
  ! CHECK-DAG: %[[V_6:.*]] = fir.load %[[V_1]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[V_7:.*]] = fir.load %[[V_3]]#0 : !fir.ref<i32>
  ! CHECK:     %[[V_8:.*]] = arith.addi %[[V_6]], %[[V_7]] : i32
  ! CHECK:     %[[V_9:.*]] = arith.addi %[[V_8]], %c4{{.*}} : i32
  ! CHECK:     hlfir.assign %[[V_9]] to %[[V_5]]#0 : i32, !fir.ref<i32>
  ! CHECK:     %[[V_10:.*]] = fir.load %[[V_5]]#0 : !fir.ref<i32>
  ! CHECK:     return %[[V_10]] : i32
  ! CHECK:   }
  module procedure fff
    fff = vv + ww + 4
  end procedure fff
end submodule ss2

submodule(mm) sss
contains
  ! CHECK-LABEL: func @_QMmmPff3
  ! CHECK-DAG: %[[V_0:.*]] = fir.address_of(@_QMmmEvv) : !fir.ref<i32>
  ! CHECK-DAG: %[[V_1:.*]]:2 = hlfir.declare %[[V_0]] {uniq_name = "_QMmmEvv"}
  ! CHECK-DAG: %[[V_2:.*]]:2 = hlfir.declare %arg0 {{.*}} {uniq_name = "_QMmmSsssFff3Enn"}
  ! CHECK:     %[[V_3:.*]] = fir.load %[[V_2]]#0 : !fir.ref<i32>
  ! CHECK:     %[[V_4:.*]] = arith.addi %[[V_3]], %c3{{.*}} : i32
  ! CHECK:     %[[V_5:.*]] = fir.convert %[[V_4]] : (i32) -> i64
  ! CHECK:     %[[V_6:.*]] = fir.convert %[[V_5]] : (i64) -> index
  ! CHECK:     %[[V_7:.*]] = arith.cmpi sgt, %[[V_6]], %c0{{.*}} : index
  ! CHECK:     %[[V_8:.*]] = arith.select %[[V_7]], %[[V_6]], %c0{{.*}} : index
  ! CHECK:     %[[V_9:.*]] = fir.alloca !fir.array<?xi32>, %[[V_8]] {bindc_name = "ff3", uniq_name = "_QMmmSsssFff3Eff3"}
  ! CHECK:     %[[V_10:.*]] = fir.shape %[[V_8]] : (index) -> !fir.shape<1>
  ! CHECK:     %[[V_11:.*]]:2 = hlfir.declare %[[V_9]](%[[V_10]]) {uniq_name = "_QMmmSsssFff3Eff3"}
  ! CHECK-DAG: %[[V_12:.*]] = fir.load %[[V_2]]#0 : !fir.ref<i32>
  ! CHECK-DAG: %[[V_13:.*]] = fir.load %[[V_1]]#0 : !fir.ref<i32>
  ! CHECK:     %[[V_14:.*]] = arith.muli %[[V_12]], %[[V_13]] : i32
  ! CHECK:     %[[V_15:.*]] = arith.addi %[[V_14]], %c6{{.*}} : i32
  ! CHECK:     hlfir.assign %[[V_15]] to %[[V_11]]#0 : i32, !fir.box<!fir.array<?xi32>>
  ! CHECK:     %[[V_16:.*]] = fir.load %[[V_11]]#1 : !fir.ref<!fir.array<?xi32>>
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
  ! CHECK:     fir.call @_QMmmPff1(%{{.*}}) {{.*}} : (!fir.ref<i32>) -> !fir.array<?xi32>
  print*, ff1(1) ! expect: 22 22
  ! CHECK:     fir.call @_QMmmPff2(%{{.*}}) {{.*}} : (!fir.ref<i32>) -> !fir.array<?xi32>
  print*, ff2(2) ! expect: 44 44 44 44
  ! CHECK:     fir.call @_QMmmPff3(%{{.*}}) {{.*}} : (!fir.ref<i32>) -> !fir.array<?xi32>
  print*, ff3(3) ! expect: 66 66 66 66 66 66
end program pp
