//===-- sbvalue-cast.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifdef DO_VIRTUAL_INHERITANCE
#define VIRTUAL virtual
#else
#define VIRTUAL 
#endif

#include <stdio.h>

class Base
{
public:
    Base(int val) : m_base_val (val) {}
    virtual ~Base() {}

    virtual void
    forcast(int input) {
        int future_val = m_base_val + input * 1;
        printf("Forcasting %d\n", future_val);
    }

protected:
    int m_base_val;
};

class DerivedA : public VIRTUAL Base
{
public:
    DerivedA(int val) : Base(val*2), m_a_val(val) {
        printf("DerivedA::ctor()->\n");
        printf("m_base_val=%d\n", m_base_val);
        printf("m_a_val=%d\n", m_a_val);
    }
    virtual ~DerivedA() {}

private:
    int m_a_val;
};

class DerivedB : public VIRTUAL Base
{
public:
    DerivedB(int val) : Base(val), m_b_val(val*3) {
        printf("DerivedB::ctor()->\n");
        printf("m_base_val=%d\n", m_base_val);
        printf("m_b_val=%d\n", m_b_val);
    }
    virtual ~DerivedB() {}
    
    virtual void
    forcast(int input) {
        int future_val = m_b_val + input * 2;
        printf("Forcasting %d\n", future_val);
    }

private:
    int m_b_val;
};

int
main(int argc, char **argv)
{
	DerivedA* dA = new DerivedA(10);
	DerivedB* dB = new DerivedB(12);
	Base *array[2] = {dA, dB};
    Base *teller = NULL;
    for (int i = 0; i < 2; ++i) {
        teller = array[i];
        teller->forcast(i); // Set breakpoint here.
    }

    return 0;
}
