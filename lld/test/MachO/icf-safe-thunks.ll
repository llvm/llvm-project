; REQUIRES: aarch64

; RUN: rm -rf %t; mkdir %t
; RUN: llc -filetype=obj %s -O3 -o %t/icf-obj-safe-thunks.o -enable-machine-outliner=never -mtriple arm64-apple-macos -addrsig
; RUN: %lld -arch arm64 -lSystem --icf=safe_thunks -dylib -o %t/icf-safe.dylib -map %t/icf-safe.map %t/icf-obj-safe-thunks.o
; RUN: llvm-objdump %t/icf-safe.dylib -d --macho | FileCheck %s --check-prefixes=CHECK-ARM64
; RUN: cat %t/icf-safe.map | FileCheck %s --check-prefixes=CHECK-ARM64-MAP

;;; Check that we generate valid dSYM and that stabs entries are not created for the ICF'ed functions (thunks)
; RUN: dsymutil %t/icf-safe.dylib -o %t/icf-safe.dSYM
; RUN: llvm-dwarfdump --verify %t/icf-safe.dSYM | FileCheck %s --check-prefix=VERIFY-DSYM
; VERIFY-DSYM: No errors.

;;; Check that we don't generate STABS entries (N_FUN) for ICF'ed functions
; RUN: dsymutil -s %t/icf-safe.dylib | FileCheck %s --check-prefix=VERIFY-STABS
; VERIFY-STABS-NOT:  N_FUN {{.*}} _func_2identical_v2
; VERIFY-STABS-NOT:  N_FUN {{.*}} _func_3identical_v2
; VERIFY-STABS-NOT:  N_FUN {{.*}} _func_3identical_v3
; VERIFY-STABS-NOT:  N_FUN {{.*}} _func_3identical_v2_canmerge
; VERIFY-STABS-NOT:  N_FUN {{.*}} _func_3identical_v3_canmerge

; CHECK-ARM64:        (__TEXT,__text) section
; CHECK-ARM64-NEXT:   _func_unique_1:
; CHECK-ARM64-NEXT:        mov {{.*}}, #0x1
;
; CHECK-ARM64:        _func_unique_2_canmerge:
; CHECK-ARM64-NEXT:   _func_2identical_v1:
; CHECK-ARM64-NEXT:        mov {{.*}}, #0x2
;
; CHECK-ARM64:        _func_3identical_v1:
; CHECK-ARM64-NEXT:        mov {{.*}}, #0x3
;
; CHECK-ARM64:        _func_3identical_v1_canmerge:
; CHECK-ARM64-NEXT:   _func_3identical_v2_canmerge:
; CHECK-ARM64-NEXT:   _func_3identical_v3_canmerge:
; CHECK-ARM64-NEXT:        mov {{.*}}, #0x21
;
; CHECK-ARM64:        _call_all_funcs:
; CHECK-ARM64-NEXT:        stp  x29
;
; CHECK-ARM64:        _take_func_addr:
; CHECK-ARM64-NEXT:        adr
;
; CHECK-ARM64:        _func_2identical_v2:
; CHECK-ARM64-NEXT:        b  _func_2identical_v1
; CHECK-ARM64-NEXT:   _func_3identical_v2:
; CHECK-ARM64-NEXT:        b  _func_3identical_v1
; CHECK-ARM64-NEXT:   _func_3identical_v3:
; CHECK-ARM64-NEXT:        b  _func_3identical_v1


; CHECK-ARM64-MAP:      0x00000010 [  2] _func_unique_1
; CHECK-ARM64-MAP-NEXT: 0x00000010 [  2] _func_2identical_v1
; CHECK-ARM64-MAP-NEXT: 0x00000000 [  2] _func_unique_2_canmerge
; CHECK-ARM64-MAP-NEXT: 0x00000010 [  2] _func_3identical_v1
; CHECK-ARM64-MAP-NEXT: 0x00000010 [  2] _func_3identical_v1_canmerge
; CHECK-ARM64-MAP-NEXT: 0x00000000 [  2] _func_3identical_v2_canmerge
; CHECK-ARM64-MAP-NEXT: 0x00000000 [  2] _func_3identical_v3_canmerge
; CHECK-ARM64-MAP-NEXT: 0x00000034 [  2] _call_all_funcs
; CHECK-ARM64-MAP-NEXT: 0x00000050 [  2] _take_func_addr
; CHECK-ARM64-MAP-NEXT: 0x00000004 [  2] _func_2identical_v2
; CHECK-ARM64-MAP-NEXT: 0x00000004 [  2] _func_3identical_v2
; CHECK-ARM64-MAP-NEXT: 0x00000004 [  2] _func_3identical_v3

; ModuleID = 'icf-safe-thunks.cpp'
source_filename = "icf-safe-thunks.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx11.0.0"
@g_val = global i8 0, align 1, !dbg !0
@g_ptr = global ptr null, align 8, !dbg !7
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_unique_1() #0 !dbg !19 {
entry:
  store volatile i8 1, ptr @g_val, align 1, !dbg !22, !tbaa !23
  ret void, !dbg !26
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_unique_2_canmerge() local_unnamed_addr #0 !dbg !27 {
entry:
  store volatile i8 2, ptr @g_val, align 1, !dbg !28, !tbaa !23
  ret void, !dbg !29
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_2identical_v1() #0 !dbg !30 {
entry:
  store volatile i8 2, ptr @g_val, align 1, !dbg !31, !tbaa !23
  ret void, !dbg !32
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_2identical_v2() #0 !dbg !33 {
entry:
  store volatile i8 2, ptr @g_val, align 1, !dbg !34, !tbaa !23
  ret void, !dbg !35
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_3identical_v1() #0 !dbg !36 {
entry:
  store volatile i8 3, ptr @g_val, align 1, !dbg !37, !tbaa !23
  ret void, !dbg !38
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_3identical_v2() #0 !dbg !39 {
entry:
  store volatile i8 3, ptr @g_val, align 1, !dbg !40, !tbaa !23
  ret void, !dbg !41
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_3identical_v3() #0 !dbg !42 {
entry:
  store volatile i8 3, ptr @g_val, align 1, !dbg !43, !tbaa !23
  ret void, !dbg !44
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_3identical_v1_canmerge() local_unnamed_addr #0 !dbg !45 {
entry:
  store volatile i8 33, ptr @g_val, align 1, !dbg !46, !tbaa !23
  ret void, !dbg !47
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_3identical_v2_canmerge() local_unnamed_addr #0 !dbg !48 {
entry:
  store volatile i8 33, ptr @g_val, align 1, !dbg !49, !tbaa !23
  ret void, !dbg !50
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @func_3identical_v3_canmerge() local_unnamed_addr #0 !dbg !51 {
entry:
  store volatile i8 33, ptr @g_val, align 1, !dbg !52, !tbaa !23
  ret void, !dbg !53
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp uwtable(sync)
define void @call_all_funcs() local_unnamed_addr #1 !dbg !54 {
entry:
  tail call void @func_unique_1(), !dbg !55
  tail call void @func_unique_2_canmerge(), !dbg !56
  tail call void @func_2identical_v1(), !dbg !57
  tail call void @func_2identical_v2(), !dbg !58
  tail call void @func_3identical_v1(), !dbg !59
  tail call void @func_3identical_v2(), !dbg !60
  tail call void @func_3identical_v3(), !dbg !61
  tail call void @func_3identical_v1_canmerge(), !dbg !62
  tail call void @func_3identical_v2_canmerge(), !dbg !63
  tail call void @func_3identical_v3_canmerge(), !dbg !64
  ret void, !dbg !65
}
; Function Attrs: mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync)
define void @take_func_addr() local_unnamed_addr #0 !dbg !66 {
entry:
  store volatile ptr @func_unique_1, ptr @g_ptr, align 8, !dbg !67, !tbaa !68
  store volatile ptr @func_2identical_v1, ptr @g_ptr, align 8, !dbg !70, !tbaa !68
  store volatile ptr @func_2identical_v2, ptr @g_ptr, align 8, !dbg !71, !tbaa !68
  store volatile ptr @func_3identical_v1, ptr @g_ptr, align 8, !dbg !72, !tbaa !68
  store volatile ptr @func_3identical_v2, ptr @g_ptr, align 8, !dbg !73, !tbaa !68
  store volatile ptr @func_3identical_v3, ptr @g_ptr, align 8, !dbg !74, !tbaa !68
  ret void, !dbg !75
}
attributes #0 = { mustprogress nofree noinline norecurse nounwind ssp memory(readwrite, argmem: none) uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #1 = { mustprogress nofree noinline norecurse nounwind ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15, !16, !17}
!llvm.ident = !{!18}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_val", scope: !2, file: !3, line: 5, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 20.0.0git (https://github.com/alx32/llvm-project.git 02d6aad5cc940f17904c1288dfabc3fd2d439279)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, globals: !6, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!3 = !DIFile(filename: "icf-safe-thunks.cpp", directory: "/tmp/safe_thunks")
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!6 = !{!0, !7}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "g_ptr", scope: !2, file: !3, line: 6, type: !9, isLocal: false, isDefinition: true)
!9 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !5)
!10 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !{i32 7, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 8, !"PIC Level", i32 2}
!16 = !{i32 7, !"uwtable", i32 1}
!17 = !{i32 7, !"frame-pointer", i32 1}
!18 = !{!"clang version 20.0.0git (https://github.com/alx32/llvm-project.git 02d6aad5cc940f17904c1288dfabc3fd2d439279)"}
!19 = distinct !DISubprogram(name: "func_unique_1", scope: !3, file: !3, line: 8, type: !20, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{null}
!22 = !DILocation(line: 9, column: 11, scope: !19)
!23 = !{!24, !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !DILocation(line: 10, column: 1, scope: !19)
!27 = distinct !DISubprogram(name: "func_unique_2_canmerge", scope: !3, file: !3, line: 12, type: !20, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!28 = !DILocation(line: 13, column: 11, scope: !27)
!29 = !DILocation(line: 14, column: 1, scope: !27)
!30 = distinct !DISubprogram(name: "func_2identical_v1", scope: !3, file: !3, line: 16, type: !20, scopeLine: 16, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!31 = !DILocation(line: 17, column: 11, scope: !30)
!32 = !DILocation(line: 18, column: 1, scope: !30)
!33 = distinct !DISubprogram(name: "func_2identical_v2", scope: !3, file: !3, line: 20, type: !20, scopeLine: 20, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!34 = !DILocation(line: 21, column: 11, scope: !33)
!35 = !DILocation(line: 22, column: 1, scope: !33)
!36 = distinct !DISubprogram(name: "func_3identical_v1", scope: !3, file: !3, line: 24, type: !20, scopeLine: 24, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!37 = !DILocation(line: 25, column: 11, scope: !36)
!38 = !DILocation(line: 26, column: 1, scope: !36)
!39 = distinct !DISubprogram(name: "func_3identical_v2", scope: !3, file: !3, line: 28, type: !20, scopeLine: 28, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!40 = !DILocation(line: 29, column: 11, scope: !39)
!41 = !DILocation(line: 30, column: 1, scope: !39)
!42 = distinct !DISubprogram(name: "func_3identical_v3", scope: !3, file: !3, line: 32, type: !20, scopeLine: 32, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!43 = !DILocation(line: 33, column: 11, scope: !42)
!44 = !DILocation(line: 34, column: 1, scope: !42)
!45 = distinct !DISubprogram(name: "func_3identical_v1_canmerge", scope: !3, file: !3, line: 36, type: !20, scopeLine: 36, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!46 = !DILocation(line: 37, column: 11, scope: !45)
!47 = !DILocation(line: 38, column: 1, scope: !45)
!48 = distinct !DISubprogram(name: "func_3identical_v2_canmerge", scope: !3, file: !3, line: 40, type: !20, scopeLine: 40, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!49 = !DILocation(line: 41, column: 11, scope: !48)
!50 = !DILocation(line: 42, column: 1, scope: !48)
!51 = distinct !DISubprogram(name: "func_3identical_v3_canmerge", scope: !3, file: !3, line: 44, type: !20, scopeLine: 44, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!52 = !DILocation(line: 45, column: 11, scope: !51)
!53 = !DILocation(line: 46, column: 1, scope: !51)
!54 = distinct !DISubprogram(name: "call_all_funcs", scope: !3, file: !3, line: 48, type: !20, scopeLine: 48, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!55 = !DILocation(line: 49, column: 5, scope: !54)
!56 = !DILocation(line: 50, column: 5, scope: !54)
!57 = !DILocation(line: 51, column: 5, scope: !54)
!58 = !DILocation(line: 52, column: 5, scope: !54)
!59 = !DILocation(line: 53, column: 5, scope: !54)
!60 = !DILocation(line: 54, column: 5, scope: !54)
!61 = !DILocation(line: 55, column: 5, scope: !54)
!62 = !DILocation(line: 56, column: 5, scope: !54)
!63 = !DILocation(line: 57, column: 5, scope: !54)
!64 = !DILocation(line: 58, column: 5, scope: !54)
!65 = !DILocation(line: 59, column: 1, scope: !54)
!66 = distinct !DISubprogram(name: "take_func_addr", scope: !3, file: !3, line: 61, type: !20, scopeLine: 61, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!67 = !DILocation(line: 62, column: 11, scope: !66)
!68 = !{!69, !69, i64 0}
!69 = !{!"any pointer", !24, i64 0}
!70 = !DILocation(line: 63, column: 11, scope: !66)
!71 = !DILocation(line: 64, column: 11, scope: !66)
!72 = !DILocation(line: 65, column: 11, scope: !66)
!73 = !DILocation(line: 66, column: 11, scope: !66)
!74 = !DILocation(line: 67, column: 11, scope: !66)
!75 = !DILocation(line: 68, column: 1, scope: !66)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;; Generate the above LLVM IR with the below script ;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; #!/bin/bash
; set -ex
; TOOLCHAIN_BIN="llvm-project/build/Debug/bin"
;
; # Create icf-safe-thunks.cpp file
; cat > icf-safe-thunks.cpp <<EOF
;
; #define ATTR __attribute__((noinline)) extern "C"
; typedef unsigned long long ULL;
;
; volatile char g_val = 0;
; void *volatile g_ptr = 0;
;
; ATTR void func_unique_1() {
;     g_val = 1;
; }
;
; ATTR void func_unique_2_canmerge() {
;     g_val = 2;
; }
;
; ATTR void func_2identical_v1() {
;     g_val = 2;
; }
;
; ATTR void func_2identical_v2() {
;     g_val = 2;
; }
;
; ATTR void func_3identical_v1() {
;     g_val = 3;
; }
;
; ATTR void func_3identical_v2() {
;     g_val = 3;
; }
;
; ATTR void func_3identical_v3() {
;     g_val = 3;
; }
;
; ATTR void func_3identical_v1_canmerge() {
;     g_val = 33;
; }
;
; ATTR void func_3identical_v2_canmerge() {
;     g_val = 33;
; }
;
; ATTR void func_3identical_v3_canmerge() {
;     g_val = 33;
; }
;
; ATTR void call_all_funcs() {
;     func_unique_1();
;     func_unique_2_canmerge();
;     func_2identical_v1();
;     func_2identical_v2();
;     func_3identical_v1();
;     func_3identical_v2();
;     func_3identical_v3();
;     func_3identical_v1_canmerge();
;     func_3identical_v2_canmerge();
;     func_3identical_v3_canmerge();
; }
;
; ATTR void take_func_addr() {
;     g_ptr = (void*)func_unique_1;
;     g_ptr = (void*)func_2identical_v1;
;     g_ptr = (void*)func_2identical_v2;
;     g_ptr = (void*)func_3identical_v1;
;     g_ptr = (void*)func_3identical_v2;
;     g_ptr = (void*)func_3identical_v3;
; }
; EOF
;
; $TOOLCHAIN_BIN/clang -target arm64-apple-macos11.0 -S -emit-llvm -g \
;                      icf-safe-thunks.cpp -O3 -o icf-safe-thunks.ll
