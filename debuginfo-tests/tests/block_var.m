// RUN: %clang %target_itanium_abi_host_triple -O0 -g %s -c -o %t.o
// RUN: %clang %target_itanium_abi_host_triple %t.o -o %t.out -framework Foundation
// RUN: %test_debuginfo %s %t.out

// REQUIRES: system-darwin

// DEBUGGER: break 24
// DEBUGGER: r
// DEBUGGER: p result
// CHECK: ${{[0-9]}} = 42

void doBlock(void (^block)(void))
{
    block();
}

int I(int n)
{
    __block int result;
    int i = 2;
    doBlock(^{
        result = n;
    });
    return result + i; /* Check value of 'result' */
}


int main (int argc, const char * argv[]) {
  return I(42);
}


