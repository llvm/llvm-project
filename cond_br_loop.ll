; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @loop_cond_br(i32 %0) {
  br label %2

2:                                                ; preds = %6, %1
  %3 = phi i32 [ %8, %6 ], [ 0, %1 ]
  %4 = phi i32 [ %7, %6 ], [ 0, %1 ]
  %5 = icmp sge i32 %3, %0
  br i1 %5, label %9, label %6

6:                                                ; preds = %2
  %7 = add i32 %4, %3
  %8 = add i32 %3, 1
  br label %2

9:                                                ; preds = %2
  ret i32 %4
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
