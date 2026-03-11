; RUN: split-file %s %t
; RUN: %clang %t/a.ll -o %t/a.out
; RUN: llvm-profdata merge --debug-info %t/a.out %t/a.proftext --max-debug-info-correlation-warnings 2 -o %t/a.profdata 2>&1 | FileCheck %s --implicit-check-not=warning --check-prefixes=CHECK,LIMIT
; RUN: llvm-profdata merge --debug-info %t/a.out %t/a.proftext --max-debug-info-correlation-warnings 0 -o %t/a.profdata 2>&1 | FileCheck %s --implicit-check-not=warning --check-prefixes=CHECK,NOLIMIT

; CHECK: warning: Incomplete DIE for function None:
; CHECK: warning: Incomplete DIE for function no_cfg: CFGHash=None
; NOLIMIT: warning: Incomplete DIE for function no_counter: {{.*}} NumCounters=None
; NOLIMIT: warning: Incomplete DIE for function no_profc: {{.*}} CounterPtr=None
; NOLIMIT: warning: Could not find address of function no_func
; LIMIT: warning: Suppressed 3 additional warnings

;--- a.proftext
:ir

;--- a.c
int main() { return 0; }

void removed() {}
void no_name() {}
void no_cfg() {}
void no_counter() {}
void no_profc() {}
void no_func() {}

// NOTE: After generating the IR below, manually remove the follwing pieces
// 1. Remove "@removed" function and "@__profc_removed" global
// 2. Remove "Function Name" annotation for "@no_name"
// 3. Remove "CFG Hash" annotation for "@no_cfg"
// 4. Remove "Num Counters" annotation for "@no_counter"
// 5. Remove "@__profc_no_profc"
// 6. Remove "@no_func"
;--- gen
clang --target=x86_64-unknown-linux-gnu -fprofile-generate -mllvm -profile-correlate=debug-info -S -emit-llvm -g a.c -o -

;--- a.ll
; ModuleID = 'a.c'
source_filename = "a.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$__llvm_profile_raw_version = comdat any

$__profc_main = comdat nodeduplicate

$__profc_removed = comdat nodeduplicate

$__profc_no_name = comdat nodeduplicate

$__profc_no_cfg = comdat nodeduplicate

$__profc_no_counter = comdat nodeduplicate

$__profc_no_profc = comdat nodeduplicate

$__profc_no_func = comdat nodeduplicate

$__llvm_profile_filename = comdat any

@__llvm_profile_raw_version = hidden constant i64 648518346341351434, comdat
@__profn_main = private constant [4 x i8] c"main"
@__profn_removed = private constant [7 x i8] c"removed"
@__profn_no_name = private constant [7 x i8] c"no_name"
@__profn_no_cfg = private constant [6 x i8] c"no_cfg"
@__profn_no_counter = private constant [10 x i8] c"no_counter"
@__profn_no_profc = private constant [8 x i8] c"no_profc"
@__profn_no_func = private constant [7 x i8] c"no_func"
@__profc_main = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !0
@__profc_no_name = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !19
@__profc_no_cfg = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !24
@__profc_no_counter = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !29
@__profc_no_func = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !39
@llvm.compiler.used = appending global [5 x ptr] [ptr @__profc_main, ptr @__profc_no_name, ptr @__profc_no_cfg, ptr @__profc_no_counter, ptr @__profc_no_func], section "llvm.metadata"
@__llvm_profile_filename = hidden constant [20 x i8] c"default_%m.proflite\00", comdat

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !2 {
  %1 = alloca i32, align 4
  %2 = load i64, ptr @__profc_main, align 8
  %3 = add i64 %2, 1
  store i64 %3, ptr @__profc_main, align 8
  store i32 0, ptr %1, align 4
  ret i32 0, !dbg !53
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @no_name() #0 !dbg !21 {
  %1 = load i64, ptr @__profc_no_name, align 8, !dbg !55
  %2 = add i64 %1, 1, !dbg !55
  store i64 %2, ptr @__profc_no_name, align 8, !dbg !55
  ret void, !dbg !55
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @no_cfg() #0 !dbg !26 {
  %1 = load i64, ptr @__profc_no_cfg, align 8, !dbg !56
  %2 = add i64 %1, 1, !dbg !56
  store i64 %2, ptr @__profc_no_cfg, align 8, !dbg !56
  ret void, !dbg !56
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @no_counter() #0 !dbg !31 {
  %1 = load i64, ptr @__profc_no_counter, align 8, !dbg !57
  %2 = add i64 %1, 1, !dbg !57
  store i64 %2, ptr @__profc_no_counter, align 8, !dbg !57
  ret void, !dbg !57
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @no_profc() #0 !dbg !36 {
  ret void, !dbg !58
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!46, !47, !48, !49, !50, !51, !52}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "__profc_main", scope: !2, file: !3, type: !14, isLocal: true, isDefinition: true, annotations: !44)
!2 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !7)
!3 = !DIFile(filename: "a.c", directory: "/proc/self/cwd", checksumkind: CSK_MD5, checksum: "22eee0eada6e6964fca794aa5a0966d0")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !8, splitDebugInlining: false, nameTableKind: None)
!8 = !{!0, !9, !19, !24, !29, !34, !39}
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "__profc_removed", scope: !11, file: !3, type: !14, isLocal: true, isDefinition: true, annotations: !15)
!11 = distinct !DISubprogram(name: "removed", scope: !3, file: !3, line: 3, type: !12, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !7)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "Profile Data Type")
!15 = !{!16, !17, !18}
!16 = !{!"Function Name", !"removed"}
!17 = !{!"CFG Hash", i64 742261418966908927}
!18 = !{!"Num Counters", i32 1}
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "__profc_no_name", scope: !21, file: !3, type: !14, isLocal: true, isDefinition: true, annotations: !22)
!21 = distinct !DISubprogram(name: "no_name", scope: !3, file: !3, line: 4, type: !12, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !7)
!22 = !{!17, !18}
!23 = !{!"Function Name", !"no_name"}
!24 = !DIGlobalVariableExpression(var: !25, expr: !DIExpression())
!25 = distinct !DIGlobalVariable(name: "__profc_no_cfg", scope: !26, file: !3, type: !14, isLocal: true, isDefinition: true, annotations: !27)
!26 = distinct !DISubprogram(name: "no_cfg", scope: !3, file: !3, line: 5, type: !12, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !7)
!27 = !{!28, !18}
!28 = !{!"Function Name", !"no_cfg"}
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = distinct !DIGlobalVariable(name: "__profc_no_counter", scope: !31, file: !3, type: !14, isLocal: true, isDefinition: true, annotations: !32)
!31 = distinct !DISubprogram(name: "no_counter", scope: !3, file: !3, line: 6, type: !12, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !7)
!32 = !{!33, !17}
!33 = !{!"Function Name", !"no_counter"}
!34 = !DIGlobalVariableExpression(var: !35, expr: !DIExpression())
!35 = distinct !DIGlobalVariable(name: "__profc_no_profc", scope: !36, file: !3, type: !14, isLocal: true, isDefinition: true, annotations: !37)
!36 = distinct !DISubprogram(name: "no_profc", scope: !3, file: !3, line: 7, type: !12, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !7)
!37 = !{!38, !17, !18}
!38 = !{!"Function Name", !"no_profc"}
!39 = !DIGlobalVariableExpression(var: !40, expr: !DIExpression())
!40 = distinct !DIGlobalVariable(name: "__profc_no_func", scope: !41, file: !3, type: !14, isLocal: true, isDefinition: true, annotations: !42)
!41 = distinct !DISubprogram(name: "no_func", scope: !3, file: !3, line: 8, type: !12, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !7)
!42 = !{!43, !17, !18}
!43 = !{!"Function Name", !"no_func"}
!44 = !{!45, !17, !18}
!45 = !{!"Function Name", !"main"}
!46 = !{i32 7, !"Dwarf Version", i32 5}
!47 = !{i32 2, !"Debug Info Version", i32 3}
!48 = !{i32 1, !"wchar_size", i32 4}
!49 = !{i32 8, !"PIC Level", i32 2}
!50 = !{i32 7, !"PIE Level", i32 2}
!51 = !{i32 7, !"uwtable", i32 2}
!52 = !{i32 7, !"frame-pointer", i32 2}
!53 = !DILocation(line: 1, column: 14, scope: !2)
!54 = !DILocation(line: 3, column: 17, scope: !11)
!55 = !DILocation(line: 4, column: 17, scope: !21)
!56 = !DILocation(line: 5, column: 16, scope: !26)
!57 = !DILocation(line: 6, column: 20, scope: !31)
!58 = !DILocation(line: 7, column: 18, scope: !36)
!59 = !DILocation(line: 8, column: 17, scope: !41)
