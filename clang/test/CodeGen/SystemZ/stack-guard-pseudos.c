// RUN: %clang_cc1 -S -mllvm -stop-after=systemz-isel -stack-protector 1 -triple=s390x-ibm-linux < %s -o - | FileCheck -check-prefix=CHECK-DAGCOMBINE %s
// RUN: %clang_cc1 -S -mllvm -stop-after=finalize-isel -stack-protector 1 -triple=s390x-ibm-linux < %s -o - | FileCheck -check-prefix=CHECK-CUSTOMINSERT %s
// RUN: not %clang_cc1 -S -stack-protector 1 -mstack-protector-guard-record -triple=s390x-ibm-linux < %s -o - 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s 
// CHECK-DAGCOMBINE:   bb.0.entry:
// CHECK-DAGCOMBINE:     MOVE_STACK_GUARD_DAG %stack.0.StackGuardSlot, 0
// CHECK-DAGCOMBINE:     COMPARE_STACK_GUARD_DAG %stack.0.StackGuardSlot, 0, implicit-def $cc
// CHECK-CUSTOMINSERT: bb.0.entry
// CHECK-CUSTOMINSERT:   early-clobber %10:addr64bit = MOVE_STACK_GUARD %stack.0.StackGuardSlot, 0
// CHECK_CUSTOMINSERT: bb.3.entry
// CHECK-CUSTOMINSERT: early-clobber %14:addr64bit = COMPARE_STACK_GUARD %stack.0.StackGuardSlot, 0, implicit-def $cc
extern char *strcpy (char * D, const char * S);
int main(int argc, char *argv[])
{
    char Buffer[8] = {0};
    strcpy(Buffer, argv[1]);
    return 0;
}

// CHECK-OPTS: error: option '-mstack-protector-guard-record' cannot be specified without '-mstack-protector-guard=global'
