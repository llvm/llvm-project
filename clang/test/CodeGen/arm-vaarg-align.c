// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple arm -target-abi aapcs %s -emit-llvm -o - | FileCheck -check-prefix=AAPCS %s
// RUN: %clang_cc1 -triple arm -target-abi apcs-gnu %s -emit-llvm -o - | FileCheck -check-prefix=APCS-GNU %s
/*
 * Check that va_arg accesses stack according to ABI alignment
 * long long and double require 8-byte alignment under AAPCS
 * however, they only require 4-byte alignment under APCS
 */
long long t1(int i, ...) {
    // AAPCS: t1
    // APCS-GNU: t1
    __builtin_va_list ap;
    __builtin_va_start(ap, i);
    // AAPCS: [[ADD:%.*]] = getelementptr inbounds i8, ptr %argp.cur, i32 7
    // AAPCS = call ptr @llvm.ptrmask.p0.i32(ptr [[ADD]], i32 -8)

    // APCS-GNU-NOT: getelementptr {{.*}}, i32 7
    // APCS-GNU-NOT: llvm.ptrmask.p0
    long long ll = __builtin_va_arg(ap, long long);
    __builtin_va_end(ap);
    return ll;
}
double t2(int i, ...) {
    // AAPCS: t2
    // APCS-GNU: t2
    __builtin_va_list ap;
    __builtin_va_start(ap, i);
    // AAPCS: [[ADD:%.*]] = getelementptr inbounds i8, ptr %argp.cur, i32 7
    // AAPCS = call ptr @llvm.ptrmask.p0.i32(ptr [[ADD]], i32 -8)

    // APCS-GNU-NOT: getelementptr {{.*}}, i32 7
    // APCS-GNU-NOT: llvm.ptrmask.p0
    double ll = __builtin_va_arg(ap, double);
    __builtin_va_end(ap);
    return ll;
}
