// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -funwind-tables=2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR-ASYNC
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -funwind-tables=2 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM-ASYNC
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -funwind-tables=2 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM-ASYNC

// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -funwind-tables=1 -emit-cir %s -o %t-sync.cir
// RUN: FileCheck --input-file=%t-sync.cir %s -check-prefix=CIR-SYNC
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -funwind-tables=1 -emit-llvm %s -o %t-sync.ll
// RUN: FileCheck --input-file=%t-sync.ll %s -check-prefix=LLVM-SYNC
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -funwind-tables=1 -emit-llvm %s -o %t-sync-ogcg.ll
// RUN: FileCheck --input-file=%t-sync-ogcg.ll %s -check-prefix=LLVM-SYNC
//
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -funwind-tables=0 -emit-cir %s -o %t-none.cir
// RUN: FileCheck --input-file=%t-none.cir %s -check-prefix=CIR-NONE
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -funwind-tables=0 -emit-llvm %s -o %t-none.ll
// RUN: FileCheck --input-file=%t-none.ll %s -check-prefix=LLVM-NONE
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -funwind-tables=0 -emit-llvm %s -o %t-none-ogcg.ll
// RUN: FileCheck --input-file=%t-none-ogcg.ll %s -check-prefix=LLVM-NONE

void normal() {}
// CIR-ASYNC: cir.func{{.*}}@_Z6normalv() attributes {{.*}}uwtable = #cir.uwtable<async>
// LLVM-ASYNC: define {{.*}}@_Z6normalv(){{.*}}#[[NORM_ATTR:[0-9]+]]

// CIR-SYNC: cir.func{{.*}}@_Z6normalv() attributes {{.*}}uwtable = #cir.uwtable<sync>
// LLVM-SYNC: define {{.*}}@_Z6normalv(){{.*}}#[[NORM_ATTR:[0-9]+]]

// CIR-NONE: cir.func{{.*}}@_Z6normalv()
// CIR-NONE-NOT: attributes {{.*}}uwtable = 
// LLVM-NONE: define {{.*}}@_Z6normalv(){{.*}}#[[NORM_ATTR:[0-9]+]]

[[clang::nouwtable]] void suppressed() {}
// CIR-ASYNC: cir.func{{.*}}@_Z10suppressedv() 
// CIR-ASYNC-NOT: attributes {{.*}}uwtable = 
// LLVM-ASYNC: define{{.*}}@_Z10suppressedv(){{.*}} #[[SUPP_ATTR:[0-9]+]]

// CIR-SYNC: cir.func{{.*}}@_Z10suppressedv() 
// CIR-SYNC-NOT: attributes {{.*}}uwtable = 
// LLVM-SYNC: define{{.*}}@_Z10suppressedv(){{.*}} #[[SUPP_ATTR:[0-9]+]]

// CIR-NONE: cir.func{{.*}}@_Z10suppressedv() 
// CIR-NONE-NOT: attributes {{.*}}uwtable = 
// LLVM-NONE: define{{.*}}@_Z10suppressedv(){{.*}} #[[SUPP_ATTR:[0-9]+]]

// LLVM-ASYNC: attributes #[[NORM_ATTR]] ={{.*}}uwtable
// LLVM-ASYNC-NOT: attributes #[[SUPP_ATTR]] ={{.*}}uwtable

// LLVM-SYNC: attributes #[[NORM_ATTR]] ={{.*}}uwtable(sync)
// LLVM-SYNC-NOT: attributes #[[SUPP_ATTR]] ={{.*}}uwtable

// LLVM-SYNC-NOT: attributes #[[NORM_ATTR]] ={{.*}}uwtable
// LLVM-SYNC-NOT: attributes #[[SUPP_ATTR]] ={{.*}}uwtable
