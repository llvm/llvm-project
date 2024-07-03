#include <cstdlib>

int *nonNull() { return new int; }

int *null() { return nullptr; }

int *mayNull() {
    if (rand() % 2 == 0) {
        return new int;
    } else {
        return nullptr;
    }
}

int main() {
    // 形如 p = null，会包括
    int *x1 = 0;
    int *x2 = NULL;
    int *x3 = nullptr;

    // 不会包括
    int *y1 = (int *)malloc(sizeof(int));
    int *y2 = new int;
    int *y3 = x1;
    int *y4 = y1;

    // 形如 p = foo()，会包括 foo() 中有 return NULL 的
    int *z1 = nonNull();
    int *z2 = null();
    int *z3 = mayNull();
}
