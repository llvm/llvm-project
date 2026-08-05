; REQUIRES: amdgpu-registered-target
; RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -S -o - %s 2>&1 | FileCheck %s

; Check that a DiagnosticUnsupported reported as a warning works
; correctly, and is not emitted as an error.

; CHECK: warning: test.c:2:20: in function use_lds_global_in_func i32 (): local memory global used by non-kernel function

target triple = "amdgcn-amd-amdhsa"

@lds = external addrspace(3) global i32, align 4

define i32 @use_lds_global_in_func() !dbg !5 {
  %load = load i32, ptr addrspace(3) @lds, !dbg !9
  ret i32 %load, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DILocation(line: 2, column: 20, scope: !5)
!10 = !DILocation(line: 2, column: 13, scope: !5)
