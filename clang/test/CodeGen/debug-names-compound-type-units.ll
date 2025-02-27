; REQUIRES: asserts
; REQUIRES: x86-registered-target

;; Tests that we use correct accelerator table when processing nested TUs.
;; Assert is not triggered.
;; File1
;; struct Foo {
;;   char f;
;; };
;; struct Foo2 {
;;   char f;
;;   Foo f1;
;; };
;; void fooFunc() {
;; Foo2 global2;
;; }
;; clang++ <file>.cpp -O0 -g2 -fdebug-types-section -gpubnames -S -emit-llvm -o <file>.ll

; RUN: llc -O0 -dwarf-version=5 -generate-type-units -filetype=obj < %s -o %t.o
; RUN: llvm-readelf --sections %t.o | FileCheck --check-prefix=OBJ %s

; OBJ: debug_names

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Foo2 = type { i8, %struct.Foo }
%struct.Foo = type { i8 }

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z7fooFuncv() #0 !dbg !10 {
entry:
  %global2 = alloca %struct.Foo2, align 1
  call void @llvm.dbg.declare(metadata ptr %global2, metadata !14, metadata !DIExpression()), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!1 = !DIFile(filename: "main.cpp", directory: "/smallMultipleTUs", checksumkind: CSK_MD5, checksum: "70bff4d50322126f3d8ca4178afad376")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0git"}
!10 = distinct !DISubprogram(name: "fooFunc", linkageName: "_Z7fooFuncv", scope: !1, file: !1, line: 9, type: !11, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !{}
!14 = !DILocalVariable(name: "global2", scope: !10, file: !1, line: 10, type: !15)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo2", file: !1, line: 4, size: 16, flags: DIFlagTypePassByValue, elements: !16, identifier: "_ZTS4Foo2")
!16 = !{!17, !19}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !15, file: !1, line: 5, baseType: !18, size: 8)
!18 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !15, file: !1, line: 6, baseType: !20, size: 8, offset: 8)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !21, identifier: "_ZTS3Foo")
!21 = !{!22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !20, file: !1, line: 2, baseType: !18, size: 8)
!23 = !DILocation(line: 10, column: 6, scope: !10)
!24 = !DILocation(line: 11, column: 1, scope: !10)
