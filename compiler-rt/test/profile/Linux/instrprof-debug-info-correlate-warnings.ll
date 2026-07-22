; RUN: split-file %s %t
; RUN: %clang %t/a.ll -o %t/a.out
; RUN: llvm-profdata merge --debug-info=%t/a.out %t/a.proftext --max-debug-info-correlation-warnings=2 -o %t/a.profdata 2>&1 | FileCheck %s --implicit-check-not=warning --check-prefixes=CHECK,LIMIT
; RUN: llvm-profdata merge --debug-info=%t/a.out %t/a.proftext --max-debug-info-correlation-warnings=0 -o %t/a.profdata 2>&1 | FileCheck %s --implicit-check-not=warning --check-prefixes=CHECK,NOLIMIT

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
__attribute__((always_inline)) void no_func() {}
void use_func() { no_func(); }

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

$__profc_use_func = comdat nodeduplicate

$__llvm_profile_filename = comdat any

@__llvm_profile_raw_version = hidden constant i64 648518346341351434, comdat
@__profn_main = private constant [4 x i8] c"main"
@__profn_removed = private constant [7 x i8] c"removed"
@__profn_no_name = private constant [7 x i8] c"no_name"
@__profn_no_cfg = private constant [6 x i8] c"no_cfg"
@__profn_no_counter = private constant [10 x i8] c"no_counter"
@__profn_no_profc = private constant [8 x i8] c"no_profc"
@__profn_no_func = private constant [7 x i8] c"no_func"
@__profn_use_func = private constant [8 x i8] c"use_func"
@__profc_main = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !0
@__profc_no_name = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !22
@__profc_no_cfg = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !28
@__profc_no_counter = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !34
@__profc_no_func = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !46
@__profc_use_func = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8, !dbg !52
@llvm.compiler.used = appending global [5 x ptr] [ptr @__profc_main, ptr @__profc_no_name, ptr @__profc_no_cfg, ptr @__profc_no_counter, ptr @__profc_no_func], section "llvm.metadata"
@__llvm_profile_filename = hidden constant [20 x i8] c"default_%m.proflite\00", comdat

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !2 {
entry:
  %retval = alloca i32, align 4
  %pgocount = load i64, ptr @__profc_main, align 8
  %0 = add i64 %pgocount, 1
  store i64 %0, ptr @__profc_main, align 8
  store i32 0, ptr %retval, align 4
  ret i32 0, !dbg !64
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @no_name() #0 !dbg !24 {
entry:
  %pgocount = load i64, ptr @__profc_no_name, align 8, !dbg !66
  %0 = add i64 %pgocount, 1, !dbg !66
  store i64 %0, ptr @__profc_no_name, align 8, !dbg !66
  ret void, !dbg !66
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @no_cfg() #0 !dbg !30 {
entry:
  %pgocount = load i64, ptr @__profc_no_cfg, align 8, !dbg !67
  %0 = add i64 %pgocount, 1, !dbg !67
  store i64 %0, ptr @__profc_no_cfg, align 8, !dbg !67
  ret void, !dbg !67
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @no_counter() #0 !dbg !36 {
entry:
  %pgocount = load i64, ptr @__profc_no_counter, align 8, !dbg !68
  %0 = add i64 %pgocount, 1, !dbg !68
  store i64 %0, ptr @__profc_no_counter, align 8, !dbg !68
  ret void, !dbg !68
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @no_profc() #0 !dbg !42 {
entry:
  ret void, !dbg !69
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @use_func() #0 !dbg !54 {
entry:
  %pgocount = load i64, ptr @__profc_use_func, align 8, !dbg !71
  %0 = add i64 %pgocount, 1, !dbg !71
  store i64 %0, ptr @__profc_use_func, align 8, !dbg !71
  %pgocount.i = load i64, ptr @__profc_no_func, align 8, !dbg !72
  %1 = add i64 %pgocount.i, 1, !dbg !72
  store i64 %1, ptr @__profc_no_func, align 8, !dbg !72
  ret void, !dbg !74
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #2

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { alwaysinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!58, !59, !60, !61, !62, !63}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "__profc_main", scope: !2, file: !3, type: !9, isLocal: true, isDefinition: true, annotations: !10)
!2 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !8)
!3 = !DIFile(filename: "a.c", directory: "/proc/self/cwd", checksumkind: CSK_MD5, checksum: "7bab6089d746f793d8c0c8d39f3a691c")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!8 = !{!0}
!9 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "Profile Data Type")
!10 = !{!11, !12, !13}
!11 = !{!"Function Name", !"main"}
!12 = !{!"CFG Hash", i64 742261418966908927}
!13 = !{!"Num Counters", i32 1}
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression())
!15 = distinct !DIGlobalVariable(name: "__profc_removed", scope: !16, file: !3, type: !9, isLocal: true, isDefinition: true, annotations: !20)
!16 = distinct !DISubprogram(name: "removed", scope: !3, file: !3, line: 3, type: !17, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !{!14}
!20 = !{!21, !12, !13}
!21 = !{!"Function Name", !"removed"}
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression())
!23 = distinct !DIGlobalVariable(name: "__profc_no_name", scope: !24, file: !3, type: !9, isLocal: true, isDefinition: true, annotations: !26)
!24 = distinct !DISubprogram(name: "no_name", scope: !3, file: !3, line: 4, type: !17, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !25)
!25 = !{!22}
!26 = !{!12, !13}
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = distinct !DIGlobalVariable(name: "__profc_no_cfg", scope: !30, file: !3, type: !9, isLocal: true, isDefinition: true, annotations: !32)
!30 = distinct !DISubprogram(name: "no_cfg", scope: !3, file: !3, line: 5, type: !17, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !31)
!31 = !{!28}
!32 = !{!33, !13}
!33 = !{!"Function Name", !"no_cfg"}
!34 = !DIGlobalVariableExpression(var: !35, expr: !DIExpression())
!35 = distinct !DIGlobalVariable(name: "__profc_no_counter", scope: !36, file: !3, type: !9, isLocal: true, isDefinition: true, annotations: !38)
!36 = distinct !DISubprogram(name: "no_counter", scope: !3, file: !3, line: 6, type: !17, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !37)
!37 = !{!34}
!38 = !{!39, !12}
!39 = !{!"Function Name", !"no_counter"}
!40 = !DIGlobalVariableExpression(var: !41, expr: !DIExpression())
!41 = distinct !DIGlobalVariable(name: "__profc_no_profc", scope: !42, file: !3, type: !9, isLocal: true, isDefinition: true, annotations: !44)
!42 = distinct !DISubprogram(name: "no_profc", scope: !3, file: !3, line: 7, type: !17, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !43)
!43 = !{!40}
!44 = !{!45, !12, !13}
!45 = !{!"Function Name", !"no_profc"}
!46 = !DIGlobalVariableExpression(var: !47, expr: !DIExpression())
!47 = distinct !DIGlobalVariable(name: "__profc_no_func", scope: !48, file: !3, type: !9, isLocal: true, isDefinition: true, annotations: !50)
!48 = distinct !DISubprogram(name: "no_func", scope: !3, file: !3, line: 8, type: !17, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !49)
!49 = !{!46}
!50 = !{!51, !12, !13}
!51 = !{!"Function Name", !"no_func"}
!52 = !DIGlobalVariableExpression(var: !53, expr: !DIExpression())
!53 = distinct !DIGlobalVariable(name: "__profc_use_func", scope: !54, file: !3, type: !9, isLocal: true, isDefinition: true, annotations: !56)
!54 = distinct !DISubprogram(name: "use_func", scope: !3, file: !3, line: 9, type: !17, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !55)
!55 = !{!52}
!56 = !{!57, !12, !13}
!57 = !{!"Function Name", !"use_func"}
!58 = !{i32 7, !"Dwarf Version", i32 5}
!59 = !{i32 2, !"Debug Info Version", i32 3}
!60 = !{i32 8, !"PIC Level", i32 2}
!61 = !{i32 7, !"PIE Level", i32 2}
!62 = !{i32 7, !"uwtable", i32 2}
!63 = !{i32 7, !"frame-pointer", i32 2}
!64 = !DILocation(line: 1, column: 14, scope: !2)
!65 = !DILocation(line: 3, column: 17, scope: !16)
!66 = !DILocation(line: 4, column: 17, scope: !24)
!67 = !DILocation(line: 5, column: 16, scope: !30)
!68 = !DILocation(line: 6, column: 20, scope: !36)
!69 = !DILocation(line: 7, column: 18, scope: !42)
!70 = !DILocation(line: 8, column: 48, scope: !48)
!71 = !DILocation(line: 9, column: 19, scope: !54)
!72 = !DILocation(line: 8, column: 48, scope: !48, inlinedAt: !73)
!73 = distinct !DILocation(line: 9, column: 19, scope: !54)
!74 = !DILocation(line: 9, column: 30, scope: !54)
