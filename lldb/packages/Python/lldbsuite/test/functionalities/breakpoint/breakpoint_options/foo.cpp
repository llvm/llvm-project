
namespace ns {
    int func(void)
    {
        return 0;
    }
}

extern "C" int foo(void)
{
    return ns::func();
}
