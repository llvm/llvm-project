// RUN: %clang_cc1 -S -mllvm -stop-after=systemz-isel -stack-protector 1 -triple=s390x-ibm-linux < %s -o - | FileCheck %s

// CHECK:   bb.0.entry:
// CHECK:     MOVE_STACK_GUARD %stack.0.StackGuardSlot, 0
// CHECK:     COMPARE_STACK_GUARD %stack.0.StackGuardSlot, implicit-def $cc

extern char *strcpy (char * D, const char * S);
int main(int argc, char *argv[])
{
    char Buffer[8] = {0};
    strcpy(Buffer, argv[1]);
    return 0;
}
