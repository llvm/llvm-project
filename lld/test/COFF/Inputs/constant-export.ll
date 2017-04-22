target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-unknown-windows-msvc18.0.0"

@__CFConstantStringClassReference = common global [32 x i32] zeroinitializer, align 4

!llvm.module.flags = !{!0}

!0 = !{i32 6, !"Linker Options", !1}
!1 = !{!2}
!2 = !{!" -export:___CFConstantStringClassReference,CONSTANT"}
