// RUN: %clang_cc1 -fblocks -emit-llvm -o - %s
// rdar : // 6182276

template <typename T> T foo(T t)
{
    void (^block)(int);
    return 1;
}

int test1(void)
{
    int i = 1;
    int b = 2;
    i = foo(b);
    return 0;
}

template <typename T, typename T1> void foo(T t, T1 r)
{
    T block_arg;
    __block T1 byref_block_arg;

    T1 (^block)(char, T, T1, double) =  
	^ T1 (char ch, T arg, T1 arg2, double d1) { byref_block_arg = arg2;
           					    return byref_block_arg + block_arg + arg; };

    void (^block2)() = ^{};
}

void test2(void)
{
    foo(100, 'a');
}

namespace rdar6182276 {
extern "C" {
int printf(const char *, ...);
}

template <typename T> T foo(T t)
{
    void (^testing)(int) = ^(int bar) { printf("bar is %d\n", bar); };
    printf("bar is\n");
    return 1;
}

template <typename T> void gorf(T t)
{
    foo(t);
}


void test(void)
{
    gorf(2);
}
}


