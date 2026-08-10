// RUN: %clang_cc1 -emit-llvm %s -o /dev/null

struct Foo  {
    Foo();
    virtual ~Foo();
};

struct Bar  {
    Bar();
    virtual ~Bar();
    virtual bool test(bool) const;
};

struct Baz : public Foo, public Bar  {
    Baz();
    virtual ~Baz();
    virtual bool test(bool) const;
};

bool Baz::test(bool) const  {
    return true;
}
