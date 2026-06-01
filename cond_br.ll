; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @cond_br_demo(i32 %0, i32 %1, i1 %2) {
  br i1 %2, label %4, label %5

4:                                                ; preds = %3
  ret i32 %0

5:                                                ; preds = %3
  ret i32 %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
