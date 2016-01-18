namespace ns
{
inline int
ifunc(int i)
{
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
    {
        return a + b;
    }
};
extern S s;
}
