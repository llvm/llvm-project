// RUN: %clang_cc1 %s -emit-llvm-only

class Base {
public:
   virtual ~Base();
};

Base::~Base()
{
}

class Foo : public Base {
public:
   virtual ~Foo();
};

Foo::~Foo()
{
}
