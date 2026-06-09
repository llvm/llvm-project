; Additional archive-handling edge cases for clang-sycl-linker.
;
; REQUIRES: spirv-registered-target
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/main.ll -o %t/main.bc
; RUN: llvm-as %t/dup.ll -o %t/dup.bc
; RUN: llvm-as %t/incl.ll -o %t/incl.bc
; RUN: llvm-as %t/extra.ll -o %t/extra.bc
; RUN: llvm-as %t/otherarch.ll -o %t/otherarch.bc
; RUN: llvm-ar rc %t/libdup.a %t/dup.bc
; RUN: llvm-ar rc %t/libincl.a %t/incl.bc
; RUN: llvm-ar rc %t/libextra.a %t/extra.bc
;
; A multiply-defined symbol inside an archive member is harmless while the
; member stays lazy: main.bc already defines bar_func1, so dup.bc is never
; extracted and there is no multiply-defined error.
; RUN: clang-sycl-linker %t/main.bc -l dup -L %t --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-LAZY-OK
; CHECK-LAZY-OK: define {{.*}}bar_func1{{.*}}
; CHECK-LAZY-OK-NOT: error:
;
; Forcing the same member in with --whole-archive extracts dup.bc and the
; conflicting definition now triggers a multiply-defined error.
; RUN: not clang-sycl-linker %t/main.bc --whole-archive -l dup -L %t --dry-run -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-WHOLE-CONFLICT
; CHECK-WHOLE-CONFLICT: error: Linking globals named {{.*}}bar_func1{{.*}} symbol multiply defined!
;
; --no-whole-archive after --whole-archive restores lazy behavior for a later
; -l: libincl is whole-archived (inclFunc is pulled in unconditionally), while
; libextra is lazy and nothing references extraFunc, so it is not pulled.
; RUN: clang-sycl-linker %t/main.bc --whole-archive -l incl --no-whole-archive -l extra -L %t --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-NO-WHOLE
; CHECK-NO-WHOLE: define {{.*}}bar_func1{{.*}}
; CHECK-NO-WHOLE: define {{.*}}inclFunc{{.*}}
; CHECK-NO-WHOLE-NOT: define {{.*}}extraFunc{{.*}}
;
; -L search must skip a directory whose name matches the requested library and
; fall through to a later -L path that holds the real archive. Here %t/dir1
; contains a *directory* named libincl.a, while %t (the second -L) has the real
; archive; the real one must be found rather than erroring on the directory.
; RUN: mkdir -p %t/dir1/libincl.a
; RUN: clang-sycl-linker %t/main.bc --whole-archive -l incl -L %t/dir1 -L %t --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-DIR-FALLTHROUGH
; CHECK-DIR-FALLTHROUGH: define {{.*}}inclFunc{{.*}}
;
; A whole-archive member built for a different target triple is silently skipped
; rather than producing a "conflicting target triples" error. otherarch.bc has
; triple spirv32; main.bc is spirv64, so otherarch's member is dropped while
; inclFunc (spirv64) is still linked.
; RUN: llvm-ar rc %t/libother.a %t/incl.bc %t/otherarch.bc
; RUN: clang-sycl-linker %t/main.bc --whole-archive -l other -L %t --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-TRIPLE-SKIP
; CHECK-TRIPLE-SKIP: define {{.*}}inclFunc{{.*}}
; CHECK-TRIPLE-SKIP-NOT: define {{.*}}otherArchFunc{{.*}}
; CHECK-TRIPLE-SKIP-NOT: error:

;--- main.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @bar_func1(i32 %a, i32 %b) {
entry:
  %res = add nsw i32 %b, %a
  ret i32 %res
}

;--- dup.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @bar_func1(i32 %a, i32 %b) {
entry:
  %mul = shl nsw i32 %a, 1
  %res = add nsw i32 %mul, %b
  ret i32 %res
}

;--- incl.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @inclFunc(i32 %a) {
entry:
  %res = add nsw i32 %a, 7
  ret i32 %res
}

;--- extra.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @extraFunc(i32 %a) {
entry:
  %res = mul nsw i32 %a, 3
  ret i32 %res
}

;--- otherarch.ll
target triple = "spirv32"

define spir_func i32 @otherArchFunc(i32 %a) {
entry:
  %res = sub nsw i32 %a, 1
  ret i32 %res
}
