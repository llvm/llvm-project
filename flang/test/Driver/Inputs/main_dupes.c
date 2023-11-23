#include <stdio.h>

int main(int argc, char * argv[]) {
    // Irrelevant what to do in here.
    // Test is supposed to fail at link time.
    printf("Hello from C [%s]\n", __FUNCTION__);
    return 0;
}
