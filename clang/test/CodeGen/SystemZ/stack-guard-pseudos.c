// RUN: %clang_cc1 -S -mllvm -stop-after=systemz-isel -stack-protector 1 -triple=s390x-ibm-linux < %s -o - | FileCheck -check-prefix=CHECK-PSEUDOS %s
// RUN: not %clang_cc1 -S -stack-protector 1 -mstack-protector-guard-record -triple=s390x-ibm-linux < %s -o - 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s 
// CHECK-PSEUDOS:   bb.0.entry:
// CHECK-PSEUDOS:     %3:addr64bit = LOAD_STACK_GUARD_ADDRESS
// CHECK-PSEUDOS:     MOVE_STACK_GUARD %stack.0.StackGuardSlot, 0, %3
// CHECK-PSEUDOS:     COMPARE_STACK_GUARD %stack.0.StackGuardSlot, 0, %3, implicit-def $cc

extern char *strcpy (char * D, const char * S);
int main(int argc, char *argv[])
{
    char Buffer[8] = {0};
    strcpy(Buffer, argv[1]);
    return 0;
}

// CHECK-OPTS: error: option '-mstack-protector-guard-record' cannot be specified without '-mstack-protector-guard=global'
