; RUN: llc -mtriple x86_64-windows-msvc -filetype obj -o %t.obj %s
; RUN: lld-link /nodefaultlib /noentry /dll /debug /out:%t.exe /pdb:%t.pdb %t.obj
; RUN: llvm-pdbutil dump -type-index=0x7c %t.pdb

; CHECK: 0x007C (char8_t) | char8_t

define dso_local i32 @main() #0 !dbg !9 {
  %1 = alloca i32, align 4
  %2 = alloca i8, align 1
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i8* %2, metadata !13, metadata !DIExpression()), !dbg !15
  store i8 0, i8* %2, align 1, !dbg !15
  %3 = load i8, i8* %2, align 1, !dbg !16
  %4 = zext i8 %3 to i32, !dbg !16
  ret i32 %4, !dbg !16
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "pdb_char8_t.cpp", directory: "C:\\src", checksumkind: CSK_MD5, checksum: "a00748d29f4e59003184945cd3e17ee3")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 1}
!8 = !{!"clang version 13.0.0"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "local", scope: !9, file: !1, line: 3, type: !14)
!14 = !DIBasicType(name: "char8_t", size: 8, encoding: DW_ATE_UTF)
!15 = !DILocation(line: 3, scope: !9)
!16 = !DILocation(line: 4, scope: !9)
