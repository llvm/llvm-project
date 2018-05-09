; RUN: not %clang_cc1 -triple r600-unknown-unknown -S -o - %s 2>&1 | FileCheck %s
; REQUIRES: amdgpu-registered-target

; This is to check that backend errors for unsupported features are formatted correctly

; CHECK: error: test.c:2:20: in function bar i32 (): unsupported call to function foo.2

target triple = "r600-unknown-unknown"

; Function Attrs: nounwind uwtable
define i32 @bar() #0 !dbg !4 {
entry:
  %call = call i32 @foo(), !dbg !12
  ret i32 %call, !dbg !13
}

; Function Attrs: nounwind uwtable
define i32 @foo() #0 !dbg !8 {
entry:
  %call = call i32 @bar(), !dbg !14
  ret i32 %call, !dbg !15
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.9.0"}
!12 = !DILocation(line: 2, column: 20, scope: !4)
!13 = !DILocation(line: 2, column: 13, scope: !4)
!14 = !DILocation(line: 3, column: 20, scope: !8)
!15 = !DILocation(line: 3, column: 13, scope: !8)
