// RUN: %check_clang_tidy %s bugprone-derived-method-shadowing-base-method %t

class Base 
{
    void method();
    void methodWithArg(int I);

    virtual Base* getThis() = 0;
};

class A : public Base
{
public:
    void method();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'A::method' shadows method with the same name in class 'Base' [bugprone-derived-method-shadowing-base-method]
// CHECK-MESSAGES: :5:5: note: previous definition of 'method' is here
};

// only declaration should be checked
void A::method()
{    
}

class B
{
public:
    void method();
};

class D: public Base
{

};

// test indirect inheritance
class E : public D
{
public:
    void method();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'E::method' shadows method with the same name in class 'Base' [bugprone-derived-method-shadowing-base-method]
};

class H : public Base
{
public:
    Base* getThis() override;
    Base const* getThis() const;
};

class I : public Base
{
public:
    // test with inline implementation
    void method()
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'I::method' shadows method with the same name in class 'Base' [bugprone-derived-method-shadowing-base-method]
    {

    }
};

class J : public Base
{
public:
    Base* getThis() final;
};

template<typename T>
class TemplateBase
{
public:
   virtual void size() const = 0;
};
    
template<typename T>
class K : public TemplateBase<T>
{
public:
    void size() const final;
};

class L : public Base
{
public:
// not same signature (take const ref) but still ambiguous
    void methodWithArg(int const& I);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'L::methodWithArg' shadows method with the same name in class 'Base' [bugprone-derived-method-shadowing-base-method]

    void methodWithArg(int const I);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'L::methodWithArg' shadows method with the same name in class 'Base' [bugprone-derived-method-shadowing-base-method]

    void methodWithArg(int *I);
    void methodWithArg(int const* I);
};

class M : public Base
{
public:
    static void method();
};

class N : public Base
{
public:
    template<typename T>
    void methodWithArg(T I);
    // TODO:  Templates are not handled yet
    template<> void methodWithArg<int>(int I);
};

namespace std{
    struct thread{
        void join();
    };
}

struct O: public std::thread{
    void join();
};

struct P: public std::thread, Base{
    void join();
    void method();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'P::method' shadows method with the same name in class 'Base' [bugprone-derived-method-shadowing-base-method]
};

class Q : public Base
{
public:
    typedef int MyInt;
// not same signature (take const ref) but still ambiguous
    void methodWithArg(MyInt const& I);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'Q::methodWithArg' shadows method with the same name in class 'Base' [bugprone-derived-method-shadowing-base-method]

    void methodWithArg(MyInt const I);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'Q::methodWithArg' shadows method with the same name in class 'Base' [bugprone-derived-method-shadowing-base-method]

    void methodWithArg(MyInt *I);
    void methodWithArg(MyInt const* I);
};
