; REQUIRES: asserts
; REQUIRES: x86-registered-target

;; Tests that accelerator table switches correctly from TU to CU when a top level TU is re-used.
;; Assert is not triggered.
;; File1
;; struct Foo {
;;   char fChar;
;; };
;; Foo fGlobal;
;; FIle2
;; struct Foo {
;;   char fChar;
;; };
;; Foo fGlobal2;
;; clang++ <file>.cpp -O0 -g2 -fdebug-types-section -gpubnames -S -emit-llvm -o <file>.ll
;; llvm-link file1.ll file2.ll -S -o thinlto-debug-names-tu-reuse.ll

; RUN: llc -O0 -dwarf-version=5 -generate-type-units -filetype=obj < %s -o %t.o
; RUN: llvm-readelf --sections %t.o | FileCheck --check-prefix=OBJ %s

; OBJ: debug_names

; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Foo = type { i8 }

@fGlobal = dso_local global %struct.Foo zeroinitializer, align 1, !dbg !0
@fGlobal2 = dso_local global %struct.Foo zeroinitializer, align 1, !dbg !9

!llvm.dbg.cu = !{!2, !11}
!llvm.ident = !{!14, !14}
!llvm.module.flags = !{!15, !16, !17, !18, !19, !20, !21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "fGlobal", scope: !2, file: !3, line: 5, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 18.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false)
!3 = !DIFile(filename: "main.cpp", directory: "/smallTUReuse", checksumkind: CSK_MD5, checksum: "4f1831504f0948b03880356fae49cb58")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !3, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !6, identifier: "_ZTS3Foo")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "fChar", scope: !5, file: !3, line: 3, baseType: !8, size: 8)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "fGlobal2", scope: !11, file: !12, line: 5, type: !5, isLocal: false, isDefinition: true)
!11 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !12, producer: "clang version 18.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !13, splitDebugInlining: false)
!12 = !DIFile(filename: "helper.cpp", directory: "/smallTUReuse", checksumkind: CSK_MD5, checksum: "014145d46991fd1eb6a2192d382feb75")
!13 = !{!9}
!14 = !{!"clang version 18.0.0git"}
!15 = !{i32 7, !"Dwarf Version", i32 5}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 8, !"PIC Level", i32 2}
!19 = !{i32 7, !"PIE Level", i32 2}
!20 = !{i32 7, !"uwtable", i32 2}
!21 = !{i32 7, !"frame-pointer", i32 2}
