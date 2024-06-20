target triple = "wasm32-unknown-emscripten"

define i32 @main() !dbg !6 {
entry:
  call void @test0(), !dbg !10
  call void @test1(), !dbg !11
  ret i32 0, !dbg !12
}

declare void @test0()

declare void @test1()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 19.0.0git"}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !7, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 5, column: 3, scope: !6)
!11 = !DILocation(line: 6, column: 3, scope: !6)
!12 = !DILocation(line: 7, column: 3, scope: !6)
