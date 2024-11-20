; REQUIRES: aarch64

;;; Build the
; RUN: rm -rf %t; mkdir %t
; RUN: llc -filetype=obj %s -O3 -o %t/icf-obj-safe-thunks-dwarf.o -enable-machine-outliner=never -mtriple arm64-apple-macos -addrsig
; RUN: %lld -arch arm64 -lSystem --icf=safe_thunks -dylib -o %t/icf-safe-dwarf.dylib %t/icf-obj-safe-thunks-dwarf.o

;;; Check that we generate valid dSYM
; RUN: dsymutil %t/icf-safe-dwarf.dylib -o %t/icf-safe.dSYM
; RUN: llvm-dwarfdump --verify %t/icf-safe.dSYM | FileCheck %s --check-prefix=VERIFY-DSYM
; VERIFY-DSYM: No errors.

;;; Check that we don't generate STABS entries (N_FUN) for ICF'ed function thunks
; RUN: dsymutil -s %t/icf-safe-dwarf.dylib | FileCheck %s --check-prefix=VERIFY-STABS
; VERIFY-STABS-NOT:  N_FUN{{.*}}_func_B
; VERIFY-STABS-NOT:  N_FUN{{.*}}_func_C

;;; Check that we do generate STABS entries (N_FUN) for non-ICF'ed functions
; VERIFY-STABS:  N_FUN{{.*}}_func_A
; VERIFY-STABS:  N_FUN{{.*}}_take_func_addr


; ModuleID = 'icf-safe-thunks-dwarf.cpp'
source_filename = "icf-safe-thunks-dwarf.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx11.0.0"

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define i32 @func_A() #0 !dbg !13 {
entry:
  ret i32 1
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define i32 @func_B() #0 !dbg !18 {
entry:
  ret i32 1
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define i32 @func_C() #0 !dbg !20 {
entry:
  ret i32 1
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define i64 @take_func_addr() #0 !dbg !22 {
entry:
  %val = alloca i64, align 8
  store i64 0, ptr %val, align 8
  %0 = load i64, ptr %val, align 8
  %add = add i64 %0, ptrtoint (ptr @func_A to i64)
  store i64 %add, ptr %val, align 8
  %1 = load i64, ptr %val, align 8
  %add1 = add i64 %1, ptrtoint (ptr @func_B to i64)
  store i64 %add1, ptr %val, align 8
  %2 = load i64, ptr %val, align 8
  %add2 = add i64 %2, ptrtoint (ptr @func_C to i64)
  store i64 %add2, ptr %val, align 8
  %3 = load i64, ptr %val, align 8
  ret i64 %3
}

attributes #0 = { noinline nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "icf-safe-thunks-dwarf.cpp", directory: "/tmp/test")
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 1}
!11 = !{i32 7, !"frame-pointer", i32 1}
!12 = !{!"clang version 20.0.0"}
!13 = distinct !DISubprogram(name: "func_A", scope: !1, file: !1, line: 4, type: !14, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!14 = !DISubroutineType(types: !15)
!15 = !{}
!18 = distinct !DISubprogram(name: "func_B", scope: !1, file: !1, line: 5, type: !14, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!20 = distinct !DISubprogram(name: "func_C", scope: !1, file: !1, line: 6, type: !14, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!22 = distinct !DISubprogram(name: "take_func_addr", scope: !1, file: !1, line: 8, type: !14, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;; Generate the above LLVM IR with the below script ;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; #!/bin/bash
; set -ex
; TOOLCHAIN_BIN="llvm-project/build/Debug/bin"
;
; # Create icf-safe-thunks-dwarf.cpp file
; cat > icf-safe-thunks-dwarf.cpp <<EOF
; #define ATTR __attribute__((noinline)) extern "C"
; typedef unsigned long long ULL;
;
; ATTR int func_A() { return 1; }
; ATTR int func_B() { return 1; }
; ATTR int func_C() { return 1; }
;
; ATTR ULL take_func_addr() {
;     ULL val = 0;
;     val += (ULL)(void*)func_A;
;     val += (ULL)(void*)func_B;
;     val += (ULL)(void*)func_C;
;     return val;
; }
; EOF
;
; $TOOLCHAIN_BIN/clang -target arm64-apple-macos11.0 -S -emit-llvm -g \
;                      icf-safe-thunks-dwarf.cpp
