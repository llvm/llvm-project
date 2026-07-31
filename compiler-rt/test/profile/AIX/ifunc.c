// RUN: %clang_pgogen %s -fno-integrated-as -o %t.out && %t.out

// candidates
__attribute__((visibility("hidden"))) int my_foo() { return 4; }
static int my_foo2() { return 5; }

// resolver
extern int x;
static void *foo_resolver() { return x ? &my_foo : &my_foo2; };

// ifunc
__attribute__((ifunc("foo_resolver"))) int foo();

int x = 1;
int main() { return foo() - 4; }
