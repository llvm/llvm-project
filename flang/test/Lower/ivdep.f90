! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK: #access_group = #llvm.access_group<id = distinct[0]<>>
! CHECK: #access_group1 = #llvm.access_group<id = distinct[1]<>>
! CHECK: #access_group2 = #llvm.access_group<id = distinct[2]<>>
! CHECK: #loop_vectorize = #llvm.loop_vectorize<disable = false>
! CHECK: #loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize, parallelAccesses = #access_group>
! CHECK: #loop_annotation1 = #llvm.loop_annotation<vectorize = #loop_vectorize, parallelAccesses = #access_group1>
! CHECK: #loop_annotation2 = #llvm.loop_annotation<vectorize = #loop_vectorize, parallelAccesses = #access_group2>

! CHECK-LABEL: ivdep_test1
subroutine ivdep_test1 
  integer :: a(10)
  !dir$ ivdep 
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
     !CHECK: fir.store %[[ARG1:.*]] to %[[VAL_4:.*]]#0 {accessGroups = [#access_group]}
     !CHECK: %[[VAL_9:.*]] = fir.load %[[VAL_4]]#0 {accessGroups = [#access_group]} 
     !CHECK: %[[VAL_10:.*]] = fir.load %[[VAL_4]]#0 {accessGroups = [#access_group]}
     !CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
     !CHECK: %[[VAL_12:.*]] = hlfir.designate %[[VAL_2:.*]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<10xi32>>, i64)
     !CHECK: hlfir.assign %[[VAL_9]] to %[[VAL_12]] {access_groups = [#access_group]} : i32, !fir.ref<i32> 
     !CHECK: %[[VAL_14:.*]] = fir.convert %[[C1:.*]] : (index) -> i32
     !CHECK: %[[VAL_15:.*]] = fir.load %[[VAL_4]]#0 {accessGroups = [#access_group]}
     !CHECK: %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_14]] overflow<nsw> : i32
     !CHECK: fir.result %[[VAL_16]] : i32 
  end do     
end subroutine ivdep_test1


! CHECK-LABEL: ivdep_test2
subroutine ivdep_test2
  integer :: a(10), b(10), c(10)
  !dir$ ivdep 
  !dir$ unknown
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation1}
  do i=1,10
     a(i)=b(i)+c(i)
     !CHECK: fir.store %[[ARG1:.*]] to %[[VAL_10:.*]]#0 {accessGroups = [#access_group1]}
     !CHECK: %[[VAL_15:.*]] = fir.load %[[VAL_10]]#0 {accessGroups = [#access_group1]}
     !CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
     !CHECK: %[[VAL_17:.*]] = hlfir.designate %[[VAL_5:.*]]#0 (%[[VAL_16]])  : (!fir.ref<!fir.array<10xi32>>, i64)
     !CHECK: %[[VAL_18:.*]] = fir.load %[[VAL_17]] {accessGroups = [#access_group1]}
     !CHECK: %[[VAL_19:.*]] = fir.load %[[VAL_10]]#0 {accessGroups = [#access_group1]}
     !CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
     !CHECK: %[[VAL_21:.*]] = hlfir.designate %[[VAL_8:.*]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<10xi32>>, i64)
     !CHECK: %[[VAL_22:.*]] = fir.load %[[VAL_21]] {accessGroups = [#access_group1]}
     !CHECK: %[[VAL_23:.*]] = arith.addi %[[VAL_18]], %[[VAL_22]] : i32 
     !CHECK: %[[VAL_24:.*]] = fir.load %[[VAL_10]]#0 {accessGroups = [#access_group1]}
     !CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
     !CHECK: %[[VAL_26:.*]] = hlfir.designate %[[VAL_2:.*]]#0 (%[[VAL_25]])  : (!fir.ref<!fir.array<10xi32>>, i64)
     !CHECK: hlfir.assign %[[VAL_23]] to %[[VAL_26]] {access_groups = [#access_group1]} : i32, !fir.ref<i32> 
     !CHECK: %[[VAL_28:.*]] = fir.convert %[[C1:.*]] : (index) -> i32
     !CHECK: %[[VAL_29:.*]] = fir.load %[[VAL_10]]#0 {accessGroups = [#access_group1]}
     !CHECK: %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_28]] overflow<nsw> : i32
     !CHECK: fir.result %[[VAL_30]] : i32
  end do
end subroutine ivdep_test2


! CHECK-LABEL: ivdep_test3
subroutine ivdep_test3
  integer :: a(10), b(10), c(10)
  !dir$ ivdep 
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation2}
  do i=1,10
     a(i)=b(i)+c(i)
     call foo() 
     !CHECK: fir.store %[[ARG1:.*]] to %[[VAL_10:.*]]#0 {accessGroups = [#access_group2]}
     !CHECK: %[[VAL_15:.*]] = fir.load %[[VAL_10]]#0 {accessGroups = [#access_group2]}
     !CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
     !CHECK: %[[VAL_17:.*]] = hlfir.designate %[[VAL_5:.*]]#0 (%[[VAL_16]])  : (!fir.ref<!fir.array<10xi32>>, i64)
     !CHECK: %[[VAL_18:.*]] = fir.load %[[VAL_17]] {accessGroups = [#access_group2]}
     !CHECK: %[[VAL_19:.*]] = fir.load %[[VAL_10]]#0 {accessGroups = [#access_group2]}
     !CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
     !CHECK: %[[VAL_21:.*]] = hlfir.designate %[[VAL_8:.*]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<10xi32>>, i64)
     !CHECK: %[[VAL_22:.*]] = fir.load %[[VAL_21]] {accessGroups = [#access_group2]}
     !CHECK: %[[VAL_23:.*]] = arith.addi %[[VAL_18]], %[[VAL_22]] : i32 
     !CHECK: %[[VAL_24:.*]] = fir.load %[[VAL_10]]#0 {accessGroups = [#access_group2]}
     !CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
     !CHECK: %[[VAL_26:.*]] = hlfir.designate %[[VAL_2:.*]]#0 (%[[VAL_25]])  : (!fir.ref<!fir.array<10xi32>>, i64)
     !CHECK: hlfir.assign %[[VAL_23]] to %[[VAL_26]] {access_groups = [#access_group2]} : i32, !fir.ref<i32>
     !CHECK: fir.call @_QFivdep_test3Pfoo() fastmath<contract> {accessGroups = [#access_group2]}
     !CHECK: %[[VAL_28:.*]] = fir.convert %[[C1:.*]] : (index) -> i32
     !CHECK: %[[VAL_29:.*]] = fir.load %[[VAL_10]]#0 {accessGroups = [#access_group2]}
     !CHECK: %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_28]] overflow<nsw> : i32
     !CHECK: fir.result %[[VAL_30]] : i32 
  end do
  contains
    subroutine foo()
    end subroutine
end subroutine ivdep_test3

