; ModuleID = 'hi_foo.c'
source_filename = "hi_foo.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; // hi_foo.c:
; int y[2] = {23, 41};
;  
; void foo(int p) {
;   y[p & 1]++;
; }

@y = hidden local_unnamed_addr global [2 x i32] [i32 23, i32 41], align 4, !dbg !0

; Function Attrs: nounwind
define hidden void @foo(i32 %p) local_unnamed_addr #0 !dbg !14 {
entry:
  call void @llvm.dbg.value(metadata i32 %p, metadata !18, metadata !DIExpression()), !dbg !19
  %and = and i32 %p, 1, !dbg !20
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* @y, i32 0, i32 %and, !dbg !21
  %0 = load i32, i32* %arrayidx, align 4, !dbg !22, !tbaa !23
  %inc = add nsw i32 %0, 1, !dbg !22
  store i32 %inc, i32* %arrayidx, align 4, !dbg !22, !tbaa !23
  ret void, !dbg !27
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 331321)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "hi_foo.c", directory: "/Users/yury/llvmwasm")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 64, elements: !8)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 2)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 7.0.0 (trunk 331321)"}
!14 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 3, type: !15, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !7}
!17 = !{!18}
!18 = !DILocalVariable(name: "p", arg: 1, scope: !14, file: !3, line: 3, type: !7)
!19 = !DILocation(line: 3, column: 14, scope: !14)
!20 = !DILocation(line: 4, column: 7, scope: !14)
!21 = !DILocation(line: 4, column: 3, scope: !14)
!22 = !DILocation(line: 4, column: 11, scope: !14)
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 5, column: 1, scope: !14)
