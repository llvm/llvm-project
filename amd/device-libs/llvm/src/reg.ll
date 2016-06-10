target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

declare i32 @llvm.read_register.i32(metadata) #0
declare i64 @llvm.read_register.i64(metadata) #0

define i64 @__llvm_amdgcn_read_exec() #1 {
    %1 = call i64 @llvm.read_register.i64(metadata !0) #2
    ret i64 %1
}

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { nounwind convergent }

!0 = !{!"exec"}
