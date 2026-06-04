; Tests clang-sycl-linker linking behavior.
;
; REQUIRES: spirv-registered-target
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/foo.ll -o %t/foo.bc
; RUN: llvm-as %t/bar.ll -o %t/bar.bc
; RUN: llvm-as %t/baz.ll -o %t/baz.bc
; RUN: llvm-as %t/libfoo.ll -o %t/libfoo.bc
; RUN: llvm-as %t/addFive.ll -o %t/addFive.bc
; RUN: llvm-as %t/unusedFunc.ll -o %t/unusedFunc.bc
; RUN: rm -f %t/libfoo.a %t/libdevice.a
; RUN: llvm-ar rc %t/libfoo.a %t/libfoo.bc
; RUN: llvm-ar rc %t/libdevice.a %t/addFive.bc %t/unusedFunc.bc
;
; Test linking two input files.
; RUN: clang-sycl-linker %t/foo.bc %t/bar.bc --dry-run -o a.spv --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-SIMPLE
; CHECK-SIMPLE: define {{.*}}foo_func1{{.*}}
; CHECK-SIMPLE: define {{.*}}foo_func2{{.*}}
; CHECK-SIMPLE: define {{.*}}bar_func1{{.*}}
; CHECK-SIMPLE-NOT: define {{.*}}addFive{{.*}}
; CHECK-SIMPLE-NOT: define {{.*}}unusedFunc{{.*}}
;
; Test that multiply defined symbols are reported as errors.
; RUN: not clang-sycl-linker %t/bar.bc %t/baz.bc --dry-run -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-MULTIPLE-DEFS
; CHECK-MULTIPLE-DEFS: error: Linking globals named {{.*}}bar_func1{{.*}} symbol multiply defined!
;
; Test lazy linking with an archive library: only needed members are extracted.
; foo.bc references addFive, so addFive.bc is extracted from libdevice.a.
; unusedFunc.bc is not needed, so it should NOT be extracted.
; RUN: clang-sycl-linker %t/foo.bc %t/bar.bc -l device -L %t --dry-run -o a.spv --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-LAZY-LINK
; CHECK-LAZY-LINK: define {{.*}}foo_func1{{.*}}
; CHECK-LAZY-LINK: define {{.*}}foo_func2{{.*}}
; CHECK-LAZY-LINK: define {{.*}}bar_func1{{.*}}
; CHECK-LAZY-LINK: define {{.*}}addFive{{.*}}
; CHECK-LAZY-LINK-NOT: define {{.*}}unusedFunc{{.*}}
;
; Test linking with an archive library file using -l:libname.a syntax.
; Archive linking extracts members at file granularity, so all functions in libfoo.bc are included.
; RUN: clang-sycl-linker %t/foo.bc %t/bar.bc -l :libfoo.a -L %t --dry-run -o a.spv --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE-LIB
; CHECK-DEVICE-LIB: define {{.*}}foo_func1{{.*}}
; CHECK-DEVICE-LIB: define {{.*}}foo_func2{{.*}}
; CHECK-DEVICE-LIB: define {{.*}}bar_func1{{.*}}
; CHECK-DEVICE-LIB: define {{.*}}addFive{{.*}}
; CHECK-DEVICE-LIB: define {{.*}}unusedFunc{{.*}}
;
; Test that an absolute path as a positional argument still performs lazy member extraction.
; libdevice.a has two members (addFive.bc and unusedFunc.bc).
; Since foo.bc needs addFive, only addFive.bc member is extracted; unusedFunc.bc is not.
; RUN: clang-sycl-linker %t/foo.bc %t/bar.bc %t/libdevice.a --dry-run -o a.spv --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE-LIB-POS
; CHECK-DEVICE-LIB-POS: define {{.*}}foo_func1{{.*}}
; CHECK-DEVICE-LIB-POS: define {{.*}}foo_func2{{.*}}
; CHECK-DEVICE-LIB-POS: define {{.*}}bar_func1{{.*}}
; CHECK-DEVICE-LIB-POS: define {{.*}}addFive{{.*}}
; CHECK-DEVICE-LIB-POS-NOT: define {{.*}}unusedFunc{{.*}}

;--- foo.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @foo_func1(i32 %a, i32 %b) {
entry:
  %call = tail call spir_func i32 @addFive(i32 %b)
  %res = tail call spir_func i32 @bar_func1(i32 %a, i32 %call)
  ret i32 %res
}

declare spir_func i32 @bar_func1(i32, i32)

declare spir_func i32 @addFive(i32)

define spir_func i32 @foo_func2(i32 %c, i32 %d) {
entry:
  %res = add nsw i32 %c, %d
  ret i32 %res
}

;--- bar.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @bar_func1(i32 %a, i32 %b) {
entry:
  %res = add nsw i32 %b, %a
  ret i32 %res
}

;--- baz.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @bar_func1(i32 %a, i32 %b) {
entry:
  %mul = shl nsw i32 %a, 1
  %res = add nsw i32 %mul, %b
  ret i32 %res
}


;--- libfoo.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @addFive(i32 %a) {
entry:
  %res = add nsw i32 %a, 5
  ret i32 %res
}

define spir_func i32 @unusedFunc(i32 %a) {
entry:
  %res = mul nsw i32 %a, 5
  ret i32 %res
}

;--- addFive.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @addFive(i32 %a) {
entry:
  %res = add nsw i32 %a, 5
  ret i32 %res
}

;--- unusedFunc.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @unusedFunc(i32 %a) {
entry:
  %res = mul nsw i32 %a, 5
  ret i32 %res
}
