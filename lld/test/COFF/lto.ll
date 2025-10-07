; REQUIRES: x86
; RUN: mkdir -p %t.dir
; RUN: llvm-as -o %t.dir/main.lto.obj %s
; RUN: llvm-as -o %t.dir/foo.lto.obj %S/Inputs/lto-dep.ll
; RUN: rm -f %t.dir/foo.lto.lib
; RUN: llvm-ar cru %t.dir/foo.lto.lib %t.dir/foo.lto.obj

; RUN: llc -filetype=obj -o %t.dir/main.obj %s
; RUN: llc -filetype=obj -o %t.dir/foo.obj %S/Inputs/lto-dep.ll
; RUN: rm -f %t.dir/foo.lib
; RUN: llvm-ar cru %t.dir/foo.lib %t.dir/foo.obj

; RUN: lld-link /out:%t.dir/main.exe /entry:main /include:f2 /subsystem:console %t.dir/main.lto.obj %t.dir/foo.lto.obj
; RUN: llvm-readobj --file-headers %t.dir/main.exe | FileCheck -check-prefix=HEADERS-11 %s
; RUN: llvm-objdump --no-print-imm-hex -d %t.dir/main.exe | FileCheck --check-prefix=TEXT-11 %s
; RUN: lld-link /out:%t.dir/main.exe /entry:main /include:f2 /subsystem:console %t.dir/main.lto.obj %t.dir/foo.lto.lib /verbose 2>&1 | FileCheck -check-prefix=VERBOSE %s
; RUN: llvm-readobj --file-headers %t.dir/main.exe | FileCheck -check-prefix=HEADERS-11 %s
; RUN: llvm-objdump --no-print-imm-hex -d %t.dir/main.exe | FileCheck --check-prefix=TEXT-11 %s

; RUN: lld-link /out:%t.dir/main.exe /entry:main /subsystem:console %t.dir/main.obj %t.dir/foo.lto.obj
; RUN: llvm-readobj --file-headers %t.dir/main.exe | FileCheck -check-prefix=HEADERS-01 %s
; RUN: llvm-objdump --no-print-imm-hex -d %t.dir/main.exe | FileCheck --check-prefix=TEXT-01 %s
; RUN: lld-link /out:%t.dir/main.exe /entry:main /subsystem:console %t.dir/main.obj %t.dir/foo.lto.lib
; RUN: llvm-readobj --file-headers %t.dir/main.exe | FileCheck -check-prefix=HEADERS-01 %s
; RUN: llvm-objdump --no-print-imm-hex -d %t.dir/main.exe | FileCheck --check-prefix=TEXT-01 %s

; RUN: lld-link /out:%t.dir/main.exe /entry:main /subsystem:console %t.dir/main.lto.obj %t.dir/foo.obj
; RUN: llvm-readobj --file-headers %t.dir/main.exe | FileCheck -check-prefix=HEADERS-10 %s
; RUN: llvm-objdump --no-print-imm-hex -d %t.dir/main.exe | FileCheck --check-prefix=TEXT-10 %s
; RUN: lld-link /out:%t.dir/main.exe /entry:main /subsystem:console %t.dir/main.lto.obj %t.dir/foo.lib
; RUN: llvm-readobj --file-headers %t.dir/main.exe | FileCheck -check-prefix=HEADERS-10 %s
; RUN: llvm-objdump --no-print-imm-hex -d %t.dir/main.exe | FileCheck --check-prefix=TEXT-10 %s

; VERBOSE: foo.lto.lib({{.*}}foo.lto.obj)

; HEADERS-11: AddressOfEntryPoint: 0x1000
; TEXT-11: Disassembly of section .text:
; TEXT-11-EMPTY:
; TEXT-11-NEXT: <.text>:
; TEXT-11-NEXT: xorl	%eax, %eax
; TEXT-11-NEXT: retq
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: int3
; TEXT-11-NEXT: movl	$2, %eax
; TEXT-11-NEXT: retq

; HEADERS-01: AddressOfEntryPoint: 0x1000
; TEXT-01: Disassembly of section .text:
; TEXT-01-EMPTY:
; TEXT-01-NEXT: <.text>:
; TEXT-01-NEXT: subq	$40, %rsp
; TEXT-01-NEXT: callq	0x140001020
; TEXT-01-NEXT: xorl	%eax, %eax
; TEXT-01-NEXT: addq	$40, %rsp
; TEXT-01-NEXT: retq
; TEXT-01-NEXT: retq
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: int3
; TEXT-01-NEXT: retq

; HEADERS-10: AddressOfEntryPoint: 0x1020
; TEXT-10: Disassembly of section .text:
; TEXT-10-EMPTY:
; TEXT-10-NEXT: <.text>:
; TEXT-10-NEXT: retq
; TEXT-10-NEXT: nopw %cs:(%rax,%rax)
; TEXT-10-NEXT: nopl (%rax,%rax)
; TEXT-10-NEXT: retq
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: int3
; TEXT-10-NEXT: subq	$40, %rsp
; TEXT-10-NEXT: callq	0x140001000
; TEXT-10-NEXT: xorl	%eax, %eax
; TEXT-10-NEXT: addq	$40, %rsp
; TEXT-10-NEXT: retq

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define i32 @main() {
  call void @foo()
  ret i32 0
}

declare void @foo()

$f1 = comdat any
define i32 @f1() comdat($f1) {
  ret i32 1
}

$f2 = comdat any
define i32 @f2() comdat($f2) {
  ret i32 2
}

define internal void @internal() {
  ret void
}
