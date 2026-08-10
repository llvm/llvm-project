// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefixes=CHECK,NOUNWIND %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fexceptions -fcxx-exceptions -funwind-tables=1 -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefixes=CHECK,SYNCUNWIND %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fexceptions -fcxx-exceptions -funwind-tables=2 -emit-llvm -o - %s | \
// RUN:   FileCheck --check-prefixes=CHECK,ASYNCUNWIND %s

void caller(void callback()) noexcept { callback(); }

// CHECK: define {{.*}}void @__clang_call_terminate({{[^)]*}}) #[[#ATTRNUM:]]
// CHECK: attributes #[[#ATTRNUM]] = {
// NOUNWIND-NOT: uwtable
// NOUNWIND-SAME: }
// SYNCUNWIND-SAME: uwtable(sync)
// ASYNCUNWIND-SAME: uwtable{{ }}
