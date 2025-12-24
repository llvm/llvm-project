! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

! CHECK-LABEL: ivdep_test1
subroutine ivdep_test1 
  integer :: a(10)
  !dir$ ivdep 
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  do i=1,10
     a(i)=i
     !CHECK: store i32 {{.*}}, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT:.*]]
     !CHECK: %[[VAL_8:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT]]
     !CHECK: %[[VAL_9:.*]] = sext i32 %[[VAL_8]] to i64
     !CHECK: %[[VAL_10:.*]] = sub nsw i64 %[[VAL_9]], 1
     !CHECK: %[[VAL_11:.*]] = mul nsw i64 %[[VAL_10]], 1
     !CHECK: %[[VAL_12:.*]] = mul nsw i64 %[[VAL_11]], 1
     !CHECK: %[[VAL_13:.*]] = add nsw i64 %[[VAL_12]], 0
     !CHECK: %[[VAL_14:.*]] = getelementptr i32, ptr {{.*}}, i64 %[[VAL_13]]
     !CHECK: store i32 %[[VAL_8]], ptr %[[VAL_14]], align 4, !llvm.access.group [[DISTRINCT]]
     !CHECK: %[[VAL_15:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT]]
     !CHECK: %[[VAL_16:.*]] = add nsw i32 %[[VAL_15]], 1
     !CHECK: %[[VAL_17:.*]] = sub i64 {{.*}}, 1
     !CHECK: br label {{.*}}, !llvm.loop ![[ANNOTATION:.*]]
  end do
end subroutine ivdep_test1


! CHECK-LABEL: ivdep_test2
subroutine ivdep_test2
  integer :: a(10), b(10), c(10)
  !dir$ ivdep 
  !dir$ unknown
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  do i=1,10
     a(i)=b(i)+c(i)
     !CHECK: store i32 {{.*}}, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT1:.*]] 
     !CHECK: %[[VAL_10:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT1]]
     !CHECK: %[[VAL_11:.*]] = sext i32 %[[VAL_10]] to i64
     !CHECK: %[[VAL_12:.*]] = sub nsw i64 %[[VAL_11]], 1
     !CHECK: %[[VAL_13:.*]] = mul nsw i64 %[[VAL_12]], 1
     !CHECK: %[[VAL_14:.*]] = mul nsw i64 %[[VAL_13]], 1
     !CHECK: %[[VAL_15:.*]] = add nsw i64 %[[VAL_14]], 0
     !CHECK: %[[VAL_16:.*]] = getelementptr i32, ptr {{.*}}, i64 %[[VAL_15]]
     !CHECK: %[[VAL_17:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT1]] 
     !CHECK: %[[VAL_18:.*]] = sub nsw i64 %[[VAL_11]], 1
     !CHECK: %[[VAL_19:.*]] = mul nsw i64 %[[VAL_18]], 1
     !CHECK: %[[VAL_20:.*]] = mul nsw i64 %[[VAL_19]], 1
     !CHECK: %[[VAL_21:.*]] = add nsw i64 %[[VAL_20]], 0
     !CHECK: %[[VAL_22:.*]] = getelementptr i32, ptr {{.*}}, i64 %[[VAL_21]]
     !CHECK: %[[VAL_23:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT1]] 
     !CHECK: %[[VAL_24:.*]] = add i32 %[[VAL_17]], %[[VAL_23]]
     !CHECK: %[[VAL_25:.*]] = sub nsw i64 %[[VAL_11]], 1
     !CHECK: %[[VAL_26:.*]] = mul nsw i64 %[[VAL_25]], 1
     !CHECK: %[[VAL_27:.*]] = mul nsw i64 %[[VAL_26]], 1
     !CHECK: %[[VAL_28:.*]] = add nsw i64 %[[VAL_27]], 0
     !CHECK: %[[VAL_29:.*]] = getelementptr i32, ptr {{.*}}, i64 %[[VAL_28]]
     !CHECK: store i32 %[[VAL_24]], ptr %[[VAL_29]], align 4, !llvm.access.group [[DISTRINCT1]]
     !CHECK: %[[VAL_30:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT1]] 
     !CHECK: %[[VAL_31:.*]] = add nsw i32 %[[VAL_30]], 1
     !CHECK: %[[VAL_32:.*]] = sub i64 {{.*}}, 1
     !CHECK: br label {{.*}}, !llvm.loop ![[ANNOTATION1:.*]]
  end do
end subroutine ivdep_test2


! CHECK-LABEL: ivdep_test3
subroutine ivdep_test3
  integer :: a(10), b(10), c(10)
  !dir$ ivdep 
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  do i=1,10
     a(i)=b(i)+c(i)
     call foo() 
     !CHECK: store i32 {{.*}}, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT2:.*]] 
     !CHECK: %[[VAL_10:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT2]]
     !CHECK: %[[VAL_11:.*]] = sext i32 %[[VAL_10]] to i64
     !CHECK: %[[VAL_12:.*]] = sub nsw i64 %[[VAL_11]], 1
     !CHECK: %[[VAL_13:.*]] = mul nsw i64 %[[VAL_12]], 1
     !CHECK: %[[VAL_14:.*]] = mul nsw i64 %[[VAL_13]], 1
     !CHECK: %[[VAL_15:.*]] = add nsw i64 %[[VAL_14]], 0
     !CHECK: %[[VAL_16:.*]] = getelementptr i32, ptr {{.*}}, i64 %[[VAL_15]]
     !CHECK: %[[VAL_17:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT2]] 
     !CHECK: %[[VAL_18:.*]] = sub nsw i64 %[[VAL_11]], 1
     !CHECK: %[[VAL_19:.*]] = mul nsw i64 %[[VAL_18]], 1
     !CHECK: %[[VAL_20:.*]] = mul nsw i64 %[[VAL_19]], 1
     !CHECK: %[[VAL_21:.*]] = add nsw i64 %[[VAL_20]], 0
     !CHECK: %[[VAL_22:.*]] = getelementptr i32, ptr {{.*}}, i64 %[[VAL_21]]
     !CHECK: %[[VAL_23:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT2]] 
     !CHECK: %[[VAL_24:.*]] = add i32 %[[VAL_17]], %[[VAL_23]]
     !CHECK: %[[VAL_25:.*]] = sub nsw i64 %[[VAL_11]], 1
     !CHECK: %[[VAL_26:.*]] = mul nsw i64 %[[VAL_25]], 1
     !CHECK: %[[VAL_27:.*]] = mul nsw i64 %[[VAL_26]], 1
     !CHECK: %[[VAL_28:.*]] = add nsw i64 %[[VAL_27]], 0
     !CHECK: %[[VAL_29:.*]] = getelementptr i32, ptr {{.*}}, i64 %[[VAL_28]]
     !CHECK: store i32 %[[VAL_24]], ptr %[[VAL_29]], align 4, !llvm.access.group [[DISTRINCT2]]
     !CHECK: call void @_QFivdep_test3Pfoo(), !llvm.access.group [[DISTRINCT2]]
     !CHECK: %[[VAL_30:.*]] = load i32, ptr {{.*}}, align 4, !llvm.access.group [[DISTRINCT2]] 
     !CHECK: %[[VAL_31:.*]] = add nsw i32 %[[VAL_30]], 1
     !CHECK: %[[VAL_32:.*]] = sub i64 {{.*}}, 1
     !CHECK: br label {{.*}}, !llvm.loop ![[ANNOTATION2:.*]]
  end do
  contains
    subroutine foo()
    end subroutine
end subroutine ivdep_test3

! CHECK: [[DISTRINCT]] = distinct !{}
! CHECK: ![[ANNOTATION]] = distinct !{![[ANNOTATION]], ![[VECTORIZE:.*]], ![[PARALLEL_ACCESSES:.*]]}
! CHECK: ![[VECTORIZE]] = !{!"llvm.loop.vectorize.enable", i1 true}
! CHECK: ![[PARALLEL_ACCESSES]] = !{!"llvm.loop.parallel_accesses", [[DISTRINCT]]}
! CHECK: [[DISTRINCT1]] = distinct !{}
! CHECK: ![[ANNOTATION1]] = distinct !{![[ANNOTATION1]], ![[VECTORIZE:.*]], ![[PARALLEL_ACCESSES1:.*]]}
! CHECK: ![[PARALLEL_ACCESSES1]] = !{!"llvm.loop.parallel_accesses", [[DISTRINCT1]]}
! CHECK: [[DISTRINCT2]] = distinct !{}
! CHECK: ![[ANNOTATION2]] = distinct !{![[ANNOTATION2]], ![[VECTORIZE:.*]], ![[PARALLEL_ACCESSES2:.*]]}
! CHECK: ![[PARALLEL_ACCESSES2]] = !{!"llvm.loop.parallel_accesses", [[DISTRINCT2]]}

