//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdint.h>

namespace a {
    class c {
    public:
        c();
        ~c();
        void func1() 
        {
            puts (LLVM_PRETTY_FUNCTION);
        }
        void func2() 
        {
            puts (LLVM_PRETTY_FUNCTION);
        }
        void func3() 
        {
            puts (LLVM_PRETTY_FUNCTION);
        }
    };

    c::c() {}
    c::~c() {}
}

namespace b {
    class c {
    public:
        c();
        ~c();
        void func1() 
        {
            puts (LLVM_PRETTY_FUNCTION);
        }
        void func3() 
        {
            puts (LLVM_PRETTY_FUNCTION);
        }
    };

    c::c() {}
    c::~c() {}
}

namespace c {
    class d {
    public:
        d () {}
        ~d() {}
        void func2() 
        {
            puts (LLVM_PRETTY_FUNCTION);
        }
        void func3() 
        {
            puts (LLVM_PRETTY_FUNCTION);
        }
    };
}

int main (int argc, char const *argv[])
{
    a::c ac;
    b::c bc;
    c::d cd;
    ac.func1();
    ac.func2();
    ac.func3();
    bc.func1();
    bc.func3();
    cd.func2();
    cd.func3();
    return 0;
}
