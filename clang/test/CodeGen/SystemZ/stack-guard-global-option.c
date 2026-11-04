// RUN: not %clang_cc1 -S -stack-protector 1 -mstack-protector-guard-record -triple=s390x-ibm-linux < %s -o - 2>&1 | FileCheck -check-prefix=CHECK-OPTS %s 
extern char *strcpy (char * D, const char * S);
int main(int argc, char *argv[])
{
    char Buffer[8] = {0};
    strcpy(Buffer, argv[1]);
    return 0;
}

// CHECK-OPTS: error: option '-mstack-protector-guard-record' cannot be specified without '-mstack-protector-guard=global'
