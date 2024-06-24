#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void g() {
    printf("In function g\n");
}
void h(int *p) {
    printf("In function h, p = %d\n", *p);
}
void f(int n) {
    int *p = NULL;
    if (n > 2) {
        p = (int *)malloc(sizeof(int));
        *p = n;
    } else if (n < 2) {
        p = &n;
    }
    printf("In function f, n = %d\n", n);
    g();
    h(p);
}
int main(int argc, char **argv) {
    f(argc);
    return 0;
}
