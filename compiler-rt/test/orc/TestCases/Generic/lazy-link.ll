; Check that files passed with the -lazy option aren't linked unless they're
; needed. The foo-ret-42.ll file, which contains only code, should not be
; needed in this -noexec case, whereas x.o, which contains a global variable
; referenced by main, should be linked (despite being passed with -lazy).
;
; RUN: rm -rf %t && mkdir -p %t
; RUN: %clang -c -o %t/foo.o %S/Inputs/foo-ret-42.ll
; RUN: %clang -c -o %t/x.o %S/Inputs/var-x-42.ll
; RUN: %clang -c -o %t/main.o %s
; RUN: %llvm_jitlink -noexec -show-linked-files %t/main.o -lazy %t/foo.o \
; RUN:     -lazy %t/x.o | FileCheck %s
;
; UNSUPPORTED: system-windows
; REQUIRES: target={{(arm|aarch|x86_)64.*}}
;
; CHECK: Linking {{.*}}main.o
; CHECK-DAG: Linking <indirect stubs graph #1>
; CHECK-DAG: Linking {{.*}}x.o
; CHECK-NOT: Linking {{.*}}foo.o

declare i32 @foo()
@x = external global i32

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %foo_result = call i32 @foo()
  %x_val = load i32, ptr @x
  %result = add nsw i32 %foo_result, %x_val
  ret i32 %result
}
