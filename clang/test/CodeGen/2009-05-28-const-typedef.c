// RUN: %clang_cc1 -emit-llvm %s -o -
// PR4281

typedef struct {
        int i;
} something;

typedef const something const_something;

something fail(void);

int
main(int argc, char *argv[])
{
        const_something R = fail();
}

