; ModuleID = 'helper.cpp'
source_filename = "helper.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@z = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0
@d = dso_local local_unnamed_addr global i32 0, align 4, !dbg !5

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local noundef i32 @_Z6helperii(i32 noundef %z_, i32 noundef %d_) local_unnamed_addr #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata i32 %z_, metadata !17, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %d_, metadata !18, metadata !DIExpression()), !dbg !19
  %0 = load i32, i32* @z, align 4, !dbg !20, !tbaa !21
  %add = add nsw i32 %0, %z_, !dbg !20
  store i32 %add, i32* @z, align 4, !dbg !20, !tbaa !21
  %1 = load i32, i32* @d, align 4, !dbg !25, !tbaa !21
  %add1 = add nsw i32 %1, %d_, !dbg !25
  store i32 %add1, i32* @d, align 4, !dbg !25, !tbaa !21
  %mul = mul nsw i32 %add1, %add, !dbg !26
  ret i32 %mul, !dbg !27
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "z", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 15.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "helper.cpp", directory: ".")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "d", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{!"clang version 15.0.0"}
!13 = distinct !DISubprogram(name: "helper", linkageName: "_Z6helperii", scope: !3, file: !3, line: 4, type: !14, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!7, !7, !7}
!16 = !{!17, !18}
!17 = !DILocalVariable(name: "z_", arg: 1, scope: !13, file: !3, line: 4, type: !7)
!18 = !DILocalVariable(name: "d_", arg: 2, scope: !13, file: !3, line: 4, type: !7)
!19 = !DILocation(line: 0, scope: !13)
!20 = !DILocation(line: 5, column: 4, scope: !13)
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !DILocation(line: 6, column: 4, scope: !13)
!26 = !DILocation(line: 7, column: 11, scope: !13)
!27 = !DILocation(line: 7, column: 2, scope: !13)
