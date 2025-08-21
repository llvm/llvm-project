; ModuleID = 'clang/test/CodeGen/cfi-check-fail-debuginfo.c'
source_filename = "clang/test/CodeGen/cfi-check-fail-debuginfo.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

@llvm.used = appending global [1 x ptr] [ptr @__cfi_check_fail], section "llvm.metadata"

; Function Attrs: nounwind
define dso_local void @caller(ptr noundef %f) local_unnamed_addr #0 !dbg !7 !type !16 !type !17 !type !18 {
entry:
    #dbg_value(ptr %f, !15, !DIExpression(), !19)
  %0 = tail call i1 @llvm.type.test(ptr %f, metadata !"_ZTSFvvE"), !dbg !20, !nosanitize !24
  br i1 %0, label %cfi.cont, label %cfi.slowpath, !dbg !20, !prof !25, !nosanitize !24

cfi.slowpath:                                     ; preds = %entry
  tail call void @__cfi_slowpath(i64 9080559750644022485, ptr %f) #6, !dbg !20, !nosanitize !24
  br label %cfi.cont, !dbg !20, !nosanitize !24

cfi.cont:                                         ; preds = %cfi.slowpath, %entry
  tail call void %f() #6, !dbg !23
  ret void, !dbg !26
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.type.test(ptr, metadata) #1

declare void @__cfi_slowpath(i64, ptr) local_unnamed_addr

; Function Attrs: nounwind
define weak_odr hidden void @__cfi_check_fail(ptr noundef %0, ptr noundef %1) #0 !dbg !27 {
entry:
    #dbg_value(ptr %0, !30, !DIExpression(), !33)
    #dbg_value(ptr %1, !32, !DIExpression(), !33)
  %.not = icmp eq ptr %0, null, !dbg !33, !nosanitize !24
  br i1 %.not, label %trap, label %cont, !dbg !33, !prof !34, !nosanitize !24

trap:                                             ; preds = %entry
  tail call void @llvm.ubsantrap(i8 2) #7, !dbg !33, !nosanitize !24
  unreachable, !dbg !33, !nosanitize !24

cont:                                             ; preds = %entry
  %2 = load i8, ptr %0, align 4, !dbg !33, !nosanitize !24
  %3 = tail call i1 @llvm.type.test(ptr %1, metadata !"all-vtables"), !dbg !33, !nosanitize !24
  %4 = zext i1 %3 to i64, !dbg !33, !nosanitize !24
  switch i8 %2, label %cont10 [
    i8 0, label %handler.cfi_check_fail
    i8 1, label %trap3
    i8 2, label %handler.cfi_check_fail5
    i8 3, label %handler.cfi_check_fail7
    i8 4, label %trap9
  ], !dbg !33, !prof !35

handler.cfi_check_fail:                           ; preds = %cont
  %5 = ptrtoint ptr %0 to i64, !dbg !33, !nosanitize !24
  %6 = ptrtoint ptr %1 to i64, !dbg !33, !nosanitize !24
  tail call void @__ubsan_handle_cfi_check_fail(i64 %5, i64 %6, i64 %4) #8, !dbg !33, !nosanitize !24
  br label %cont10, !dbg !33

trap3:                                            ; preds = %cont
  tail call void @llvm.ubsantrap(i8 2) #9, !dbg !33, !nosanitize !24
  unreachable, !dbg !33, !nosanitize !24

handler.cfi_check_fail5:                          ; preds = %cont
  %7 = ptrtoint ptr %0 to i64, !dbg !33, !nosanitize !24
  %8 = ptrtoint ptr %1 to i64, !dbg !33, !nosanitize !24
  tail call void @__ubsan_handle_cfi_check_fail_abort(i64 %7, i64 %8, i64 %4) #9, !dbg !33, !nosanitize !24
  unreachable, !dbg !33, !nosanitize !24

handler.cfi_check_fail7:                          ; preds = %cont
  %9 = ptrtoint ptr %0 to i64, !dbg !33, !nosanitize !24
  %10 = ptrtoint ptr %1 to i64, !dbg !33, !nosanitize !24
  tail call void @__ubsan_handle_cfi_check_fail(i64 %9, i64 %10, i64 %4) #8, !dbg !33, !nosanitize !24
  br label %cont10, !dbg !33

trap9:                                            ; preds = %cont
  tail call void @llvm.ubsantrap(i8 2) #9, !dbg !33, !nosanitize !24
  unreachable, !dbg !33, !nosanitize !24

cont10:                                           ; preds = %handler.cfi_check_fail7, %handler.cfi_check_fail, %cont
  ret void, !dbg !33, !nosanitize !24
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.ubsantrap(i8 immarg) #2

; Function Attrs: uwtable
declare void @__ubsan_handle_cfi_check_fail(i64, i64, i64) local_unnamed_addr #3

; Function Attrs: noreturn nounwind uwtable
declare void @__ubsan_handle_cfi_check_fail_abort(i64, i64, i64) local_unnamed_addr #4

; Function Attrs: nounwind
define weak void @__cfi_check(i64 noundef %0, ptr noundef %1, ptr noundef %2) local_unnamed_addr #5 align 4096 {
entry:
    #dbg_value(ptr %2, !30, !DIExpression(), !33)
    #dbg_value(ptr %1, !32, !DIExpression(), !33)
  %.not.i = icmp eq ptr %2, null, !dbg !33, !nosanitize !24
  br i1 %.not.i, label %trap.i, label %cont.i, !dbg !33, !prof !34, !nosanitize !24

trap.i:                                           ; preds = %entry
  tail call void @llvm.ubsantrap(i8 2) #7, !dbg !33, !nosanitize !24
  unreachable, !dbg !33, !nosanitize !24

cont.i:                                           ; preds = %entry
  %3 = load i8, ptr %2, align 4, !dbg !33, !nosanitize !24
  %4 = tail call i1 @llvm.type.test(ptr %1, metadata !"all-vtables"), !dbg !33, !nosanitize !24
  %5 = zext i1 %4 to i64, !dbg !33, !nosanitize !24
  switch i8 %3, label %__cfi_check_fail.exit [
    i8 0, label %handler.cfi_check_fail.i
    i8 1, label %trap3.i
    i8 2, label %handler.cfi_check_fail5.i
    i8 3, label %handler.cfi_check_fail7.i
    i8 4, label %trap9.i
  ], !dbg !33, !prof !35

handler.cfi_check_fail.i:                         ; preds = %cont.i
  %6 = ptrtoint ptr %2 to i64, !dbg !33, !nosanitize !24
  %7 = ptrtoint ptr %1 to i64, !dbg !33, !nosanitize !24
  tail call void @__ubsan_handle_cfi_check_fail(i64 %6, i64 %7, i64 %5) #8, !dbg !33, !nosanitize !24
  br label %__cfi_check_fail.exit, !dbg !33

trap3.i:                                          ; preds = %cont.i
  tail call void @llvm.ubsantrap(i8 2) #9, !dbg !33, !nosanitize !24
  unreachable, !dbg !33, !nosanitize !24

handler.cfi_check_fail5.i:                        ; preds = %cont.i
  %8 = ptrtoint ptr %2 to i64, !dbg !33, !nosanitize !24
  %9 = ptrtoint ptr %1 to i64, !dbg !33, !nosanitize !24
  tail call void @__ubsan_handle_cfi_check_fail_abort(i64 %8, i64 %9, i64 %5) #9, !dbg !33, !nosanitize !24
  unreachable, !dbg !33, !nosanitize !24

handler.cfi_check_fail7.i:                        ; preds = %cont.i
  %10 = ptrtoint ptr %2 to i64, !dbg !33, !nosanitize !24
  %11 = ptrtoint ptr %1 to i64, !dbg !33, !nosanitize !24
  tail call void @__ubsan_handle_cfi_check_fail(i64 %10, i64 %11, i64 %5) #8, !dbg !33, !nosanitize !24
  br label %__cfi_check_fail.exit, !dbg !33

trap9.i:                                          ; preds = %cont.i
  tail call void @llvm.ubsantrap(i8 2) #9, !dbg !33, !nosanitize !24
  unreachable, !dbg !33, !nosanitize !24

__cfi_check_fail.exit:                            ; preds = %cont.i, %handler.cfi_check_fail.i, %handler.cfi_check_fail7.i
  ret void
}

attributes #0 = { nounwind "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { cold noreturn nounwind }
attributes #3 = { uwtable }
attributes #4 = { noreturn nounwind uwtable }
attributes #5 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }
attributes #8 = { nomerge nounwind }
attributes #9 = { nomerge noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "clang/test/CodeGen")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"Cross-DSO CFI", i32 1}
!5 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
!6 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!7 = distinct !DISubprogram(name: "caller", scope: !8, file: !8, line: 22, type: !9, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!8 = !DIFile(filename: "cfi-check-fail-debuginfo.c", directory: "clang/test/CodeGen")
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{!15}
!15 = !DILocalVariable(name: "f", arg: 1, scope: !7, file: !8, line: 22, type: !11)
!16 = !{i64 0, !"_ZTSFvPFvvEE"}
!17 = !{i64 0, !"_ZTSFvPvE.generalized"}
!18 = !{i64 0, i64 2451761621477796417}
!19 = !DILocation(line: 0, scope: !7)
!20 = !DILocation(line: 0, scope: !21, inlinedAt: !23)
!21 = distinct !DISubprogram(name: "__ubsan_check_cfi_icall", scope: !8, file: !8, type: !22, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !0)
!22 = !DISubroutineType(types: null)
!23 = !DILocation(line: 23, column: 3, scope: !7)
!24 = !{}
!25 = !{!"branch_weights", i32 1048575, i32 1}
!26 = !DILocation(line: 24, column: 1, scope: !7)
!27 = distinct !DISubprogram(linkageName: "__cfi_check_fail", scope: !1, file: !1, type: !28, flags: DIFlagArtificial, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29)
!28 = !DISubroutineType(types: !24)
!29 = !{!30, !32}
!30 = !DILocalVariable(arg: 1, scope: !27, type: !31, flags: DIFlagArtificial)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!32 = !DILocalVariable(arg: 2, scope: !27, type: !31, flags: DIFlagArtificial)
!33 = !DILocation(line: 0, scope: !27)
!34 = !{!"branch_weights", i32 1, i32 1048575}
!35 = !{!"branch_weights", i32 -20480, i32 4096, i32 4095, i32 4095, i32 4095, i32 4095}
