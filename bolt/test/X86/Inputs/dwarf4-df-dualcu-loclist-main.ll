; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0
@y = dso_local local_unnamed_addr global i32 1, align 4, !dbg !5

; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local void @_Z3usePiS_(i32* nocapture noundef %x, i32* nocapture noundef %y) local_unnamed_addr #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata i32* %x, metadata !18, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32* %y, metadata !19, metadata !DIExpression()), !dbg !20
  %0 = load i32, i32* %x, align 4, !dbg !21, !tbaa !22
  %add = add nsw i32 %0, 4, !dbg !21
  store i32 %add, i32* %x, align 4, !dbg !21, !tbaa !22
  %1 = load i32, i32* %y, align 4, !dbg !26, !tbaa !22
  %sub = add nsw i32 %1, -2, !dbg !26
  store i32 %sub, i32* %y, align 4, !dbg !26, !tbaa !22
  ret void, !dbg !27
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %argc, i8** nocapture noundef readnone %argv) local_unnamed_addr #1 !dbg !28 {
entry:
  call void @llvm.dbg.value(metadata i32 %argc, metadata !35, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i8** %argv, metadata !36, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.value(metadata i32* @x, metadata !18, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.value(metadata i32* @y, metadata !19, metadata !DIExpression()), !dbg !38
  %add.i = add nsw i32 %argc, 4, !dbg !40
  store i32 %add.i, i32* @x, align 4, !dbg !40, !tbaa !22
  %sub.i = add nsw i32 %argc, 1, !dbg !41
  store i32 %sub.i, i32* @y, align 4, !dbg !41, !tbaa !22
  %call = tail call noundef i32 @_Z6helperii(i32 noundef %add.i, i32 noundef %sub.i), !dbg !42
  ret i32 %call, !dbg !43
}

declare !dbg !44 dso_local noundef i32 @_Z6helperii(i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress norecurse uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 7, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 15.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "main.cpp", directory: ".")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 8, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{!"clang version 15.0.0"}
!13 = distinct !DISubprogram(name: "use", linkageName: "_Z3usePiS_", scope: !3, file: !3, line: 1, type: !14, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!17 = !{!18, !19}
!18 = !DILocalVariable(name: "x", arg: 1, scope: !13, file: !3, line: 1, type: !16)
!19 = !DILocalVariable(name: "y", arg: 2, scope: !13, file: !3, line: 1, type: !16)
!20 = !DILocation(line: 0, scope: !13)
!21 = !DILocation(line: 2, column: 4, scope: !13)
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !DILocation(line: 3, column: 4, scope: !13)
!27 = !DILocation(line: 4, column: 1, scope: !13)
!28 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 9, type: !29, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !34)
!29 = !DISubroutineType(types: !30)
!30 = !{!7, !7, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !32, size: 64)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64)
!33 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!34 = !{!35, !36}
!35 = !DILocalVariable(name: "argc", arg: 1, scope: !28, file: !3, line: 9, type: !7)
!36 = !DILocalVariable(name: "argv", arg: 2, scope: !28, file: !3, line: 9, type: !31)
!37 = !DILocation(line: 0, scope: !28)
!38 = !DILocation(line: 0, scope: !13, inlinedAt: !39)
!39 = distinct !DILocation(line: 12, column: 4, scope: !28)
!40 = !DILocation(line: 2, column: 4, scope: !13, inlinedAt: !39)
!41 = !DILocation(line: 3, column: 4, scope: !13, inlinedAt: !39)
!42 = !DILocation(line: 13, column: 11, scope: !28)
!43 = !DILocation(line: 13, column: 4, scope: !28)
!44 = !DISubprogram(name: "helper", linkageName: "_Z6helperii", scope: !3, file: !3, line: 6, type: !45, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !47)
!45 = !DISubroutineType(types: !46)
!46 = !{!7, !7, !7}
!47 = !{}
