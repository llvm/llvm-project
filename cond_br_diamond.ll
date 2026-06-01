; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @diamond_cond_br(i32 %0, i32 %1, i32 %2) {
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %3
  br label %7

6:                                                ; preds = %3
  br label %7

7:                                                ; preds = %5, %6
  %8 = phi i32 [ %2, %6 ], [ %1, %5 ]
  ret i32 %8
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
