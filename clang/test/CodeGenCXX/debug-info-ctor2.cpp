// RUN: %clang -fverbose-asm -g -S %s -o - | grep AT_explicit


class MyClass
{
public:
    explicit MyClass (int i) : 
        m_i (i)
    {}
private:
    int m_i;
};

MyClass m(1);

