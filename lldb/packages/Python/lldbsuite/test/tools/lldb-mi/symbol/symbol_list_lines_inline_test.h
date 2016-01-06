namespace ns
{
inline int
ifunc(int i)
{ // FUNC_ifunc
    return i;
}
struct S
{
    int a;
    int b;
    S()
        : a(3)
        , b(4)
    {
    }
    int
    mfunc()
    { // FUNC_mfunc
        return a + b;
    }
};
extern S s;
}
