; Test weak symbol resolution semantics for clang-sycl-linker.
;
; REQUIRES: spirv-registered-target
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/main.ll -o %t/main.bc
; RUN: llvm-as %t/weak-archive.ll -o %t/weak-archive.bc
; RUN: llvm-as %t/strong-archive.ll -o %t/strong-archive.bc
; RUN: llvm-as %t/weak-main.ll -o %t/weak-main.bc
; RUN: llvm-as %t/strong-main.ll -o %t/strong-main.bc
; RUN: llvm-as %t/another-weak.ll -o %t/another-weak.bc
; RUN: llvm-ar rc %t/libweak.a %t/weak-archive.bc
; RUN: llvm-ar rc %t/libstrong.a %t/strong-archive.bc
; RUN: llvm-ar rc %t/libmixed.a %t/weak-archive.bc %t/strong-archive.bc
; RUN: llvm-ar rc %t/libanother.a %t/another-weak.bc
;
; Strong definition in main input takes precedence over weak in lazy archive.
; The weak definition in libweak.a should NOT be extracted because main.bc already
; defines commonFunc (strongly), so there's no undefined reference to resolve.
; RUN: clang-sycl-linker %t/main.bc -l weak -L %t --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-STRONG-WINS
; CHECK-STRONG-WINS: define{{.*}}i32 @commonFunc{{.*}} {
; CHECK-STRONG-WINS-NEXT: ret i32 42
; CHECK-STRONG-WINS-NOT: ret i32 999
;
; Weak definition in main, strong in lazy archive.
; When weak-main.bc references commonFunc weakly, the strong definition in
; libstrong.a should be extracted and take precedence.
; RUN: clang-sycl-linker %t/weak-main.bc -u commonFunc -l strong -L %t --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-STRONG-FROM-ARCHIVE
; CHECK-STRONG-FROM-ARCHIVE: define{{.*}}i32 @commonFunc{{.*}} {
; CHECK-STRONG-FROM-ARCHIVE-NEXT: ret i32 100
; CHECK-STRONG-FROM-ARCHIVE-NOT: ret i32 999
;
; Two weak definitions from different lazy archives.
; Both archives provide weak definitions. The first one encountered (by -L/-l order)
; should be taken. Here libweak.a comes before libanother.a, so weak-archive's
; version (ret 999) should be extracted when -u forces the symbol.
; RUN: clang-sycl-linker %t/strong-main.bc -u commonFunc -l weak -l another -L %t --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-FIRST-WEAK-WINS
; CHECK-FIRST-WEAK-WINS: define{{.*}}weak{{.*}}i32 @commonFunc{{.*}} {
; CHECK-FIRST-WEAK-WINS-NEXT: ret i32 999
; CHECK-FIRST-WEAK-WINS-NOT: ret i32 777
;
; Whole-archive with mixed weak and strong definitions in same archive.
; When --whole-archive forces extraction of all members, the strong definition
; should override the weak one. libmixed.a contains both weak-archive.bc and
; strong-archive.bc; the strong one should win.
; RUN: clang-sycl-linker %t/strong-main.bc --whole-archive -l mixed -L %t --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-WHOLE-STRONG-WINS
; CHECK-WHOLE-STRONG-WINS: define{{.*}}i32 @commonFunc{{.*}} {
; CHECK-WHOLE-STRONG-WINS-NEXT: ret i32 100
; CHECK-WHOLE-STRONG-WINS-NOT: ret i32 999
;
; Strong definition in one input, weak in another non-archive input.
; Both are non-lazy; the strong definition should be kept.
; RUN: clang-sycl-linker %t/strong-main.bc %t/weak-main.bc --dry-run -o /dev/null --print-linked-module 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-NON-LAZY-STRONG
; CHECK-NON-LAZY-STRONG: define{{.*}}i32 @mainFunc{{.*}}
; CHECK-NON-LAZY-STRONG: define{{.*}}i32 @commonFunc{{.*}}
; CHECK-NON-LAZY-STRONG-NOT: define{{.*}}weak{{.*}}@commonFunc

;--- main.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @mainFunc() {
  ret i32 0
}

define spir_func i32 @commonFunc() {
  ret i32 42
}

;--- weak-archive.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define weak spir_func i32 @commonFunc() {
  ret i32 999
}

;--- strong-archive.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @commonFunc() {
  ret i32 100
}

;--- weak-main.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define weak spir_func i32 @commonFunc() {
  ret i32 999
}

define spir_func i32 @weakMainFunc() {
  ret i32 1
}

;--- strong-main.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @mainFunc() {
  ret i32 2
}

;--- another-weak.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define weak spir_func i32 @commonFunc() {
  ret i32 777
}
