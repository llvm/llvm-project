//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

static int g_next_value = 12345;

class VBase
{
public:
    VBase() : m_value(g_next_value++) {}
    virtual ~VBase() {}
    void Print() 
    {
        printf("%p: %s\n%p: m_value = 0x%8.8x\n", this, __PRETTY_FUNCTION__, &m_value, m_value);
    }
    int m_value;
};

class Derived1 : public virtual VBase
{
public:
    Derived1() {};
    void Print ()
    {
        printf("%p: %s\n", this, __PRETTY_FUNCTION__);
        VBase::Print();
    }

};

class Derived2 : public virtual VBase
{
public:
    Derived2() {};
    
    void Print ()
    {
        printf("%p: %s\n", this, __PRETTY_FUNCTION__);
        VBase::Print();
    }
};

class Joiner1 : public Derived1, public Derived2
{
public:
    Joiner1() : 
        m_joiner1(3456), 
        m_joiner2(6789) {}
    void Print ()
    {
        printf("%p: %s \n%p: m_joiner1 = 0x%8.8x\n%p: m_joiner2 = 0x%8.8x\n",
               this,
               __PRETTY_FUNCTION__,
               &m_joiner1,
               m_joiner1,
               &m_joiner2,
               m_joiner2);
        Derived1::Print();
        Derived2::Print();
    }
    int m_joiner1;
    int m_joiner2;
};

class Joiner2 : public Derived2
{
    int m_stuff[32];
};

int main(int argc, const char * argv[])
{
    Joiner1 j1;
    Joiner2 j2;
    j1.Print();
    j2.Print();
    Derived2 *d = &j1;
    d = &j2;  // breakpoint 1
    return 0; // breakpoint 2
}
