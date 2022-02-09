// RUN: c-index-test core -print-source-symbols -- %s | FileCheck %s
// RUN: %clang_cc1 -emit-pch %s -o %t.pch
// RUN: c-index-test core -print-source-symbols -module-file %t.pch | FileCheck %s

// CHECK: [[@LINE+1]]:6 | function/C | test1 | [[TEST1_USR:.*]] | [[TEST1_CG:.*]] | Decl | rel: 0
void test1(void);

// CHECK: [[@LINE+1]]:20 | function/C | test2 | [[TEST2_USR:.*]] | {{.*}} | Def | rel: 0
static inline void test2(void) {
  // CHECK: [[@LINE+2]]:3 | function/C | test1 | [[TEST1_USR]] | [[TEST1_CG]] | Ref,Call,RelCall,RelCont | rel: 1
  // CHECK-NEXT: RelCall,RelCont | test2 | [[TEST2_USR]]
  test1();
}
