target triple = "wasm32-unknown-emscripten"

define weak void @foo(i32 %a, i32 %b) !dbg !6 {
  ret void
}

define void @test1() !dbg !10 {
  call void @foo(i32 4, i32 5)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "b.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 19.0.0git"}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !7, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 7, type: !11, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
