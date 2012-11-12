// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s
// Radar 9239104
class Class
{
public:
//CHECK: DW_TAG_const_type
    int foo (int p) const {
        return p+m_int;
    }  

protected:
    int m_int;
};

Class c;
