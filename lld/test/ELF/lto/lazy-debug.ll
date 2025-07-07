; REQUIRES: aarch64
;; Regression test for a case in lazy debuginfo loading.
;; The bug would cause ld.lld to crash.

; RUN: split-file %s %t
; RUN: llvm-as %t/hda_codec.s -o %t/hda_codec.o
; RUN: llvm-as %t/hda_bind.s -o %t/hda_bind.o
; RUN: ld.lld -EL -maarch64elf -r %t/hda_bind.o %t/hda_codec.o -o %t/hda_codec

;--- hda_codec.s
; ModuleID = 'hda_codec.o'
source_filename = "hda_codec.i"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.anon = type { i32 }

@hda_set_power_state_codec = hidden local_unnamed_addr global %struct.anon zeroinitializer, align 4, !dbg !0

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define hidden void @snd_hda_codec_shutdown() local_unnamed_addr #0 !dbg !19 {
entry:
  ret void, !dbg !22
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+v8a,-fmv" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "hda_set_power_state_codec", scope: !2, file: !5, line: 3, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 21.0.0git (git@github.com:llvm/llvm-project.git 93849a39c432827473ca6c676f1500da69b3aaa0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "hda_codec.i", directory: "/tmp")
!4 = !{!0}
!5 = !DIFile(filename: "hda_codec.i", directory: "/tmp", checksumkind: CSK_MD5, checksum: "c192644b468953345ff9647026173a7b")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !5, line: 1, size: i64 32, offset: i64 0, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "mfg", scope: !6, file: !5, line: 2, baseType: !9, size: i64 32, offset: i64 0)
!9 = !DIBasicType(name: "int", size: i64 32, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"frame-pointer", i32 1}
!17 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!18 = !{!"clang version 21.0.0git (git@github.com:llvm/llvm-project.git 93849a39c432827473ca6c676f1500da69b3aaa0)"}
!19 = distinct !DISubprogram(name: "snd_hda_codec_shutdown", scope: !5, file: !5, line: 4, type: !20, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{null}
!22 = !DILocation(line: 4, column: 32, scope: !19)

^0 = module: (path: "hda_codec.o", hash: (1120894731, 3099354915, 309166549, 2100129435, 1932081428))
^1 = gv: (name: "snd_hda_codec_shutdown", summaries: (function: (module: ^0, flags: (linkage: external, visibility: hidden, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1, funcFlags: (readNone: 1, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0)))) ; guid = 1539195202824839354
^2 = gv: (name: "hda_set_power_state_codec", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: hidden, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), varFlags: (readonly: 1, writeonly: 1, constant: 0)))) ; guid = 10300548032946263328
^3 = flags: 8
^4 = blockcount: 0

;--- hda_bind.s
; ModuleID = 'hda_bind.o'
source_filename = "hda_bind.i"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define hidden void @hda_codec_driver_shutdown() local_unnamed_addr #0 !dbg !11 {
entry:
  tail call void @snd_hda_codec_shutdown() #2, !dbg !15
  ret void, !dbg !16
}

declare void @snd_hda_codec_shutdown(...) local_unnamed_addr #1

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+v8a,-fmv" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git (git@github.com:llvm/llvm-project.git 93849a39c432827473ca6c676f1500da69b3aaa0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "hda_bind.i", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 1}
!9 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!10 = !{!"clang version 21.0.0git (git@github.com:llvm/llvm-project.git 93849a39c432827473ca6c676f1500da69b3aaa0)"}
!11 = distinct !DISubprogram(name: "hda_codec_driver_shutdown", scope: !12, file: !12, line: 2, type: !13, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DIFile(filename: "hda_bind.i", directory: "/tmp", checksumkind: CSK_MD5, checksum: "5907dd04e8964940b57448f37db201c6")
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DILocation(line: 2, column: 36, scope: !11)
!16 = !DILocation(line: 2, column: 62, scope: !11)

^0 = module: (path: "hda_bind.o", hash: (1958332034, 2012675483, 855691486, 2017350850, 2779827776))
^1 = gv: (name: "snd_hda_codec_shutdown") ; guid = 1539195202824839354
^2 = gv: (name: "hda_codec_driver_shutdown", summaries: (function: (module: ^0, flags: (linkage: external, visibility: hidden, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^1, tail: 1))))) ; guid = 12817427500962331703
^3 = flags: 8
^4 = blockcount: 0
