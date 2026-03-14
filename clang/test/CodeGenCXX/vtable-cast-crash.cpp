// RUN: %clang_cc1 -emit-llvm-only -triple %itanium_abi_triple %s
struct A
{
A();    
virtual ~A();
};

struct B: A
{
  B();
  ~B();
};

B::B()
{
}

B::~B()
{
}
                         
