// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O3 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O3 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O3 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

union YYSTYPE {
  void *yt_casestring;
  void *yt_ID;
};

extern YYSTYPE yylval;

static int consume(YYSTYPE v) { return v.yt_casestring != nullptr; }

int test_shift(YYSTYPE *yyvsp) {
  yylval.yt_casestring = reinterpret_cast<void *>(0x42);
  *++yyvsp = yylval;
  return consume(yyvsp[0]);
}

// CIR-LABEL: cir.func{{.*}} @_Z10test_shiftP7YYSTYPE
// CIR-NOT: cir.call{{.*}}@_ZN7YYSTYPEaSERKS_
// CIR: cir.copy

// LLVM-LABEL: define{{.*}} @_Z10test_shiftP7YYSTYPE(
// LLVM: store{{.*}}@yylval
// LLVM: store i64 66
// LLVM-NOT: readonly
// LLVM: ret i32 1

// OGCG-LABEL: define{{.*}} @_Z10test_shiftP7YYSTYPE(
// OGCG: store{{.*}}@yylval
// OGCG: store i64 66
// OGCG: ret i32 1
