#ifdef _MSC_VER
#include <intrin.h>
#define BREAKPOINT_INTRINSIC()    __debugbreak()
#else
#define BREAKPOINT_INTRINSIC()    __asm__ __volatile__ ("int3")
#endif

int
bar(int const *foo)
{
    int count = 0;
    for (int i = 0; i < 10; ++i)
    {
        count += 1;
        BREAKPOINT_INTRINSIC();
        count += 1;
    }
    return *foo;
}

int
main(int argc, char **argv)
{
    int foo = 42;
    bar(&foo);
    return 0;
}


