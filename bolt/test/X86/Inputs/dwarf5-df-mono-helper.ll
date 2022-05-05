; clang++ -g -gdwarf-5 -emit-llvm -S helper.cpp
; int z = 0;
; int d = 0;
;
; int helper(int z_, int d_) {
;  z += z_;
;  d += d_;
;  return z * d;
; }

; ModuleID = 'helper.cpp'
source_filename = "helper.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@z = dso_local global i32 0, align 4, !dbg !0
@d = dso_local global i32 0, align 4, !dbg !5

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z6helperii(i32 noundef %z_, i32 noundef %d_) #0 !dbg !14 {
entry:
  %z_.addr = alloca i32, align 4
  %d_.addr = alloca i32, align 4
  store i32 %z_, i32* %z_.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %z_.addr, metadata !18, metadata !DIExpression()), !dbg !19
  store i32 %d_, i32* %d_.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %d_.addr, metadata !20, metadata !DIExpression()), !dbg !21
  %0 = load i32, i32* %z_.addr, align 4, !dbg !22
  %1 = load i32, i32* @z, align 4, !dbg !23
  %add = add nsw i32 %1, %0, !dbg !23
  store i32 %add, i32* @z, align 4, !dbg !23
  %2 = load i32, i32* %d_.addr, align 4, !dbg !24
  %3 = load i32, i32* @d, align 4, !dbg !25
  %add1 = add nsw i32 %3, %2, !dbg !25
  store i32 %add1, i32* @d, align 4, !dbg !25
  %4 = load i32, i32* @z, align 4, !dbg !26
  %5 = load i32, i32* @d, align 4, !dbg !27
  %mul = mul nsw i32 %4, %5, !dbg !28
  ret i32 %mul, !dbg !29
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "z", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 15.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "helper.cpp", directory: ".", checksumkind: CSK_MD5, checksum: "e635924a35b65444173d0c76a54b866f")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "d", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang version 15.0.0"}
!14 = distinct !DISubprogram(name: "helper", linkageName: "_Z6helperii", scope: !3, file: !3, line: 4, type: !15, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{!7, !7, !7}
!17 = !{}
!18 = !DILocalVariable(name: "z_", arg: 1, scope: !14, file: !3, line: 4, type: !7)
!19 = !DILocation(line: 4, column: 16, scope: !14)
!20 = !DILocalVariable(name: "d_", arg: 2, scope: !14, file: !3, line: 4, type: !7)
!21 = !DILocation(line: 4, column: 24, scope: !14)
!22 = !DILocation(line: 5, column: 7, scope: !14)
!23 = !DILocation(line: 5, column: 4, scope: !14)
!24 = !DILocation(line: 6, column: 7, scope: !14)
!25 = !DILocation(line: 6, column: 4, scope: !14)
!26 = !DILocation(line: 7, column: 9, scope: !14)
!27 = !DILocation(line: 7, column: 13, scope: !14)
!28 = !DILocation(line: 7, column: 11, scope: !14)
!29 = !DILocation(line: 7, column: 2, scope: !14)
