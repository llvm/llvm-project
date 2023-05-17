// RUN: c-index-test -test-load-source all %s 2>&1 | FileCheck %s

#include "fixit-sys-header.h"
#include "fixit-user-header.h"

int main(int argc, char const *argv[])
{
    char* str;{};
    
    func_in_sys_header(str, str + 10);
    // CHECK: Number FIX-ITs = 0
    // CHECK-NEXT: candidate function not viable: no known conversion from 'char *' to 'unsigned long' for 2nd argument; dereference the argument with *
    // CHECK-NEXT: Number FIX-ITs = 0
    
    func_in_user_header(str, str + 10);
    // CHECK: Number FIX-ITs = 0
    // CHECK-NEXT: candidate function not viable: no known conversion from 'char *' to 'unsigned long' for 2nd argument; dereference the argument with *
    // CHECK-NEXT: Number FIX-ITs = 2

    return 0;
}
