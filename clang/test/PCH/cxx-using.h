// Header for PCH test cxx-using.cpp






struct B {
    void f(char c);
};

struct D : B 
{
    using B::f;
    void f(int);
};
