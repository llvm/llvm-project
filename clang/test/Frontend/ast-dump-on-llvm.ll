; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump=json %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP-EQ-JSON
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump=default %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP-EQ-DEFAULT
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump-all %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP-ALL
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump-all=json %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP-ALL-EQ-JSON
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump-all=default %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP-ALL-EQ-DEFAULT

; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-print %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-PRINT
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-view %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-VIEW
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-list %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-LIST
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump-lookups %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP-LOOKUP
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump-filter=FunctionDecl %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP-FILTER-EQ
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump-decl-types %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AST-DUMP-DECL-TYPES
; RUN: not %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-SYNTAX-ONLY


; CHECK-AST-DUMP: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-DUMP-EQ-JSON: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-DUMP-EQ-DEFAULT: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-DUMP-ALL: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-DUMP-ALL-EQ-JSON: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-DUMP-ALL-EQ-DEFAULT: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-PRINT: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-VIEW: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-LIST: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-DUMP-LOOKUP: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-DUMP-FILTER-EQ: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-AST-DUMP-DECL-TYPES: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
; CHECK-SYNTAX-ONLY: fatal error: cannot apply AST actions to LLVM IR file '{{.*}}'
