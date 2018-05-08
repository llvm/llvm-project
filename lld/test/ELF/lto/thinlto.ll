; REQUIRES: x86

; First ensure that the ThinLTO handling in lld handles
; bitcode without summary sections gracefully and generates index file.
; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/thinlto.ll -o %t2.o
; RUN: ld.lld -m elf_x86_64 --plugin-opt=thinlto-index-only -shared %t.o %t2.o -o %t3
; RUN: ls %t2.o.thinlto.bc
; RUN: not test -e %t3
; RUN: ld.lld -m elf_x86_64 -shared %t.o %t2.o -o %t4
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM

; Basic ThinLTO tests.
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o %t4.o

; Ensure lld generates an index and not a binary if requested.
; RUN: ld.lld -m elf_x86_64 --plugin-opt=thinlto-index-only -shared %t.o %t2.o -o %t3
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump %t2.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND2
; RUN: not test -e %t3

; Ensure lld generates an index even if the file is wrapped in --start-lib/--end-lib
; RUN: rm -f %t2.o.thinlto.bc
; RUN: ld.lld -m elf_x86_64 --plugin-opt=thinlto-index-only -shared %t.o %t4.o --start-lib %t2.o --end-lib -o %t5
; RUN: ls %t2.o.thinlto.bc

; Ensure lld writes linked files to linked objects file
; RUN: ld.lld -m elf_x86_64 --plugin-opt=thinlto-index-only=%tlinkedobjfile -shared %t.o %t2.o %t4.o -o %t5
; RUN: cat %tlinkedobjfile 2>&1 | FileCheck %s --check-prefix=IN1
; IN1: {{.*}}thinlto.ll.tmp.o
; IN1-NEXT: {{.*}}thinlto.ll.tmp2.o
; IN1-NEXT: {{.*}}thinlto.ll.tmp4.o

; Ensure lld generates error if unable to write to index file
; RUN: rm -f %t4.o.thinlto.bc
; RUN: touch %t4.o.thinlto.bc
; RUN: chmod 400 %t4.o.thinlto.bc
; RUN: not ld.lld -m elf_x86_64 --plugin-opt=thinlto-index-only -shared %t.o %t4.o -o %t5 2>&1 | FileCheck %s --check-prefix=ERR
; ERR: cannot open {{.*}}4.o.thinlto.bc: {{P|p}}ermission denied

; Ensure lld doesn't generates index files when thinlto-index-only is not enabled
; RUN: rm -f %t.o.thinlto.bc
; RUN: rm -f %t2.o.thinlto.bc
; RUN: rm -f %t4.o.thinlto.bc
; RUN: ld.lld -m elf_x86_64 -shared %t.o %t2.o %t4.o -o %t5
; RUN: not ls %t.o.thinlto.bc
; RUN: not ls %t2.o.thinlto.bc
; RUN: not ls %t4.o.thinlto.bc

; First force single-threaded mode
; RUN: rm -f %t.lto.o %t1.lto.o
; RUN: ld.lld -save-temps --thinlto-jobs=1 -shared %t.o %t2.o -o %t
; RUN: llvm-nm %t1.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t2.lto.o | FileCheck %s --check-prefix=NM2

; Next force multi-threaded mode
; RUN: rm -f %t2.lto.o %t21.lto.o
; RUN: ld.lld -save-temps --thinlto-jobs=2 -shared %t.o %t2.o -o %t2
; RUN: llvm-nm %t21.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t22.lto.o | FileCheck %s --check-prefix=NM2

; Then check without --thinlto-jobs (which currently default to hardware_concurrency)
; We just check that we don't crash or fail (as it's not sure which tests are
; stable on the final output file itself.
; RUN: ld.lld -shared %t.o %t2.o -o %t2

; Test to ensure that thinlto-index-only with obj-path creates the file.
; RUN: rm -f %t5.o
; RUN: ld.lld --plugin-opt=thinlto-index-only --plugin-opt=obj-path=%t5.o -shared %t.o %t2.o -o %t4
; RUN: llvm-readobj -h %t5.o | FileCheck %s --check-prefix=FORMAT
; RUN: llvm-nm %t5.o | count 0

; NM: T f
; NM1: T f
; NM1-NOT: U g
; NM2: T g
; FORMAT: Format: ELF64-x86-64

; The backend index for this module contains summaries from itself and
; Inputs/thinlto.ll, as it imports from the latter.
; BACKEND1: <MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}thinlto.ll.tmp{{.*}}.o'
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}thinlto.ll.tmp{{.*}}.o'
; BACKEND1-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND1: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND1: <VERSION
; BACKEND1: <FLAGS
; BACKEND1: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1: <VALUE_GUID op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1: <COMBINED
; BACKEND1: <COMBINED
; BACKEND1: </GLOBALVAL_SUMMARY_BLOCK

; The backend index for Input/thinlto.ll contains summaries from itself only,
; as it does not import anything.
; BACKEND2: <MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <ENTRY {{.*}} record string = '{{.*}}thinlto.ll.tmp2.o'
; BACKEND2-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VERSION
; BACKEND2-NEXT: <FLAGS
; BACKEND2-NEXT: <VALUE_GUID op0=1 op1=-5300342847281564238
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
