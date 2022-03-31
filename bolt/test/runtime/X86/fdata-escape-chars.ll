@var = dso_local global i32 0, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @symb_w_whitespace() #0 {
  store volatile i32 1, i32* @var, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @symb_backslash_b() #0 {
  call void @symb_w_whitespace()
  store volatile i32 2, i32* @var, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @static_symb_backslash_b()
  call void @symb_w_whitespace()
  store i32 0, i32* %2, align 4
  br label %3

3:                                                ; preds = %7, %0
  %4 = load i32, i32* %2, align 4
  %5 = icmp slt i32 %4, 2
  br i1 %5, label %6, label %10

6:                                                ; preds = %3
  call void @symb_backslash_b()
  br label %7

7:                                                ; preds = %6
  %8 = load i32, i32* %2, align 4
  %9 = add nsw i32 %8, 1
  store i32 %9, i32* %2, align 4
  br label %3

10:                                               ; preds = %3
  %11 = load i32, i32* %1, align 4
  ret i32 %11
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @static_symb_backslash_b() #0 {
  call void @symb_w_whitespace()
  store volatile i32 3, i32* @var, align 4
  ret void
}

; REQUIRES: system-linux,bolt-runtime

; RUN: llc %s -o %t.s
; RUN: %clang %cflags -O0 %t.s -o %t.exe -Wl,-q
; RUN: llvm-objcopy --redefine-syms=%p/Inputs/fdata-escape-chars-syms.txt %t.exe
;
; RUN: llvm-bolt %t.exe -o %t.exe.instrumented -instrument  \
; RUN:   -instrumentation-file=%t.fdata
; RUN: %t.exe.instrumented
; RUN: cat %t.fdata | \
; RUN:   FileCheck --check-prefix="FDATA_CHECK" %s
; RUN: llvm-bolt %t.exe -o %t.fdata.exe -data %t.fdata -print-finalized | \
; RUN:   FileCheck --check-prefix="INSTR_CHECK" %s
;
; RUN: link_fdata %p/Inputs/fdata-escape-chars.txt %t.exe %t.pre "PREAGR"
; RUN: perf2bolt %t.exe -o %t.pre.fdata -pa -p %t.pre
; RUN: cat %t.pre.fdata | FileCheck --check-prefix="PREAGR_FDATA_CHECK" %s
; RUN: llvm-bolt %t.exe -o %t.pre.fdata.exe -data %t.pre.fdata -print-finalized | \
; RUN:   FileCheck --check-prefix="PREAGR_CHECK" %s

; FDATA_CHECK: 1 symb\ backslash\\ {{([[:xdigit:]]+)}} 1 symb\ whitespace 0 0 2
; FDATA_CHECK: 1 main {{([[:xdigit:]]+)}} 1 symb\ whitespace 0 0 1
; FDATA_CHECK: 1 main {{([[:xdigit:]]+)}} 1 symb\ backslash\\ 0 0 2

; INSTR_CHECK: Binary Function "symb whitespace"
; INSTR_CHECK: Exec Count  : 4
; INSTR_CHECK: Binary Function "symb backslash\"
; INSTR_CHECK: Exec Count  : 2
; INSTR_CHECK: {{([[:xdigit:]]+)}}:   callq   "symb whitespace" # Count: 2
; INSTR_CHECK: Binary Function "main"
; INSTR_CHECK: Exec Count  : 1
; INSTR_CHECK: {{([[:xdigit:]]+)}}:   callq   "symb whitespace" # Count: 1
; INSTR_CHECK: {{([[:xdigit:]]+)}}:   callq   "symb backslash\" # Count: 2
; INSTR_CHECK: Binary Function "static symb backslash\/1(*2)"
; INSTR_CHECK: Exec Count  : 1
; INSTR_CHECK: {{([[:xdigit:]]+)}}:   callq   "symb whitespace" # Count: 1

; PREAGR_FDATA_CHECK: 1 symb\ backslash\\ 0 1 symb\ whitespace 0 0 2
; PREAGR_FDATA_CHECK: 1 main 0 1 static\ symb\ backslash\\/1 0 0 1
; PREAGR_FDATA_CHECK: 1 main 0 1 symb\ whitespace 0 0 1
; PREAGR_FDATA_CHECK: 1 main 0 1 symb\ backslash\\ 0 0 2
; PREAGR_FDATA_CHECK: 1 static\ symb\ backslash\\/1 0 1 symb\ whitespace 0 0 1

; PREAGR_CHECK: Binary Function "symb whitespace"
; PREAGR_CHECK-DAG: Exec Count  : 4
; PREAGR_CHECK: Binary Function "symb backslash\"
; PREAGR_CHECK-DAG: Exec Count  : 2
; PREAGR_CHECK: Binary Function "static symb backslash\/1(*2)"
; PREAGR_CHECK-DAG: Exec Count  : 1
