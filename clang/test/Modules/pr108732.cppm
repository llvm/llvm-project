// RUN: %clang_cc1 -std=c++20 %s -ast-dump | FileCheck %s
export module mod;

extern "C++" {
class C
{
public:
bool foo() const {
    return true;
}
};
}

// CHECK: foo {{.*}}implicit-inline
