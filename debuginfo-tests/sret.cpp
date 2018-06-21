// RUN: %clangxx %target_itanium_abi_host_triple -O0 -g %s -c -o %t.o
// RUN: %clangxx %target_itanium_abi_host_triple %t.o -o %t.out
// RUN: %test_debuginfo %s %t.out 
// Radar 8775834
// DEBUGGER: break 62
// DEBUGGER: r
// DEBUGGER: p a
// CHECK: ${{[0-9]+}} =
// LLDB does not print artificial members.
// CHECK:  {{(_vptr\$A =)?.*}}m_int = 12

class A
{
public:
    A (int i=0);
    A (const A& rhs);
    const A&
    operator= (const A& rhs);
    virtual ~A() {}

    int get_int();

protected:
    int m_int;
};

A::A (int i) : 
    m_int(i)
{
}

A::A (const A& rhs) :
    m_int (rhs.m_int)
{
}

const A &
A::operator =(const A& rhs)
{
    m_int = rhs.m_int;
    return *this;
}

int A::get_int()
{
    return m_int;
}

class B
{
public:
    B () {}
    
    A AInstance();
};

A 
B::AInstance()
{
    A a(12);
    return a;
}

int main (int argc, char const *argv[])
{
    B b;
    int return_val = b.AInstance().get_int();
    
    A a(b.AInstance());
    return return_val;
}
