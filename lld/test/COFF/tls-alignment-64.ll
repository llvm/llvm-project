; REQUIRES: x86

; This test is to make sure that the necessary alignment for thread locals
; gets reflected in the TLS Directory of the generated executable on x86-64.
;
; aligned_thread_local specifies 'align 64' and so the generated
; exe should reflect that with a value of IMAGE_SCN_ALIGN_64BYTES
; in the Characteristics field of the IMAGE_TLS_DIRECTORY

; RUN: llc -filetype=obj %S/Inputs/tlssup-64.ll -o %t.tlssup.obj
; RUN: llc -filetype=obj %s -o %t.obj
; RUN: lld-link %t.tlssup.obj %t.obj -entry:main -nodefaultlib -out:%t.exe
; RUN: llvm-readobj --coff-tls-directory %t.exe | FileCheck %s

; CHECK: TLSDirectory {
; CHECK: Characteristics [ (0x700000)
; CHECK-NEXT: IMAGE_SCN_ALIGN_64BYTES (0x700000)

target triple = "x86_64-pc-windows-msvc"

@aligned_thread_local = thread_local global i32 42, align 64

define i32 @main() {
  %t = load i32, ptr @aligned_thread_local
  ret i32 %t
}
