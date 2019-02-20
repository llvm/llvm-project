#include <stdio.h>

int main(int argc, char const *argv[]) {
    printf("Print a formatted string so that GCC does not optimize this printf call: %s\n", argv[0]);
    return 0;
}
