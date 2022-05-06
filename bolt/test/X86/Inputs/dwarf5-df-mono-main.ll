; clang++ -g -gdwarf-5 -gsplit-dwarf=split -emit-llvm -S  main.cpp
; void use(int * x, int * y) {
; *x += 4;
; *y -= 2;
; }
;
; int helper(int z_, int d_);
; int x = 0;
; int y = 1;
; int  main(int argc, char *argv[]) {
;    x = argc;
;    y = argc + 3;
;    use(&x, &y);
;    return helper(x, y);
; }

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global i32 0, align 4, !dbg !0
@y = dso_local global i32 1, align 4, !dbg !5

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z3usePiS_(i32* noundef %x, i32* noundef %y) #0 !dbg !14 {
entry:
  %x.addr = alloca i32*, align 8
  %y.addr = alloca i32*, align 8
  store i32* %x, i32** %x.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %x.addr, metadata !19, metadata !DIExpression()), !dbg !20
  store i32* %y, i32** %y.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %y.addr, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = load i32*, i32** %x.addr, align 8, !dbg !23
  %1 = load i32, i32* %0, align 4, !dbg !24
  %add = add nsw i32 %1, 4, !dbg !24
  store i32 %add, i32* %0, align 4, !dbg !24
  %2 = load i32*, i32** %y.addr, align 8, !dbg !25
  %3 = load i32, i32* %2, align 4, !dbg !26
  %sub = sub nsw i32 %3, 2, !dbg !26
  store i32 %sub, i32* %2, align 4, !dbg !26
  ret void, !dbg !27
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main(i32 noundef %argc, i8** noundef %argv) #2 !dbg !28 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !34, metadata !DIExpression()), !dbg !35
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !36, metadata !DIExpression()), !dbg !37
  %0 = load i32, i32* %argc.addr, align 4, !dbg !38
  store i32 %0, i32* @x, align 4, !dbg !39
  %1 = load i32, i32* %argc.addr, align 4, !dbg !40
  %add = add nsw i32 %1, 3, !dbg !41
  store i32 %add, i32* @y, align 4, !dbg !42
  call void @_Z3usePiS_(i32* noundef @x, i32* noundef @y), !dbg !43
  %2 = load i32, i32* @x, align 4, !dbg !44
  %3 = load i32, i32* @y, align 4, !dbg !45
  %call = call noundef i32 @_Z6helperii(i32 noundef %2, i32 noundef %3), !dbg !46
  ret i32 %call, !dbg !47
}

declare dso_local noundef i32 @_Z6helperii(i32 noundef, i32 noundef) #3

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 7, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 15.0.0", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "main.dwo", emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: GNU)
!3 = !DIFile(filename: "main.cpp", directory: ".", checksumkind: CSK_MD5, checksum: "1f627913a0daee879e00a3a51726f0ef")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 8, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang version 15.0.0"}
!14 = distinct !DISubprogram(name: "use", linkageName: "_Z3usePiS_", scope: !3, file: !3, line: 1, type: !15, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!18 = !{}
!19 = !DILocalVariable(name: "x", arg: 1, scope: !14, file: !3, line: 1, type: !17)
!20 = !DILocation(line: 1, column: 16, scope: !14)
!21 = !DILocalVariable(name: "y", arg: 2, scope: !14, file: !3, line: 1, type: !17)
!22 = !DILocation(line: 1, column: 25, scope: !14)
!23 = !DILocation(line: 2, column: 2, scope: !14)
!24 = !DILocation(line: 2, column: 4, scope: !14)
!25 = !DILocation(line: 3, column: 2, scope: !14)
!26 = !DILocation(line: 3, column: 4, scope: !14)
!27 = !DILocation(line: 4, column: 1, scope: !14)
!28 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 9, type: !29, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
!29 = !DISubroutineType(types: !30)
!30 = !{!7, !7, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !32, size: 64)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64)
!33 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!34 = !DILocalVariable(name: "argc", arg: 1, scope: !28, file: !3, line: 9, type: !7)
!35 = !DILocation(line: 9, column: 15, scope: !28)
!36 = !DILocalVariable(name: "argv", arg: 2, scope: !28, file: !3, line: 9, type: !31)
!37 = !DILocation(line: 9, column: 27, scope: !28)
!38 = !DILocation(line: 10, column: 8, scope: !28)
!39 = !DILocation(line: 10, column: 6, scope: !28)
!40 = !DILocation(line: 11, column: 8, scope: !28)
!41 = !DILocation(line: 11, column: 13, scope: !28)
!42 = !DILocation(line: 11, column: 6, scope: !28)
!43 = !DILocation(line: 12, column: 4, scope: !28)
!44 = !DILocation(line: 13, column: 18, scope: !28)
!45 = !DILocation(line: 13, column: 21, scope: !28)
!46 = !DILocation(line: 13, column: 11, scope: !28)
!47 = !DILocation(line: 13, column: 4, scope: !28)
