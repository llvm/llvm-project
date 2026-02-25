// RUN: %check_clang_tidy %s misc-forbid-non-virtual-base-dtor %t

// should warn -> non-virtual base + derived has data
class A {};
class B: public A{
    int x;
};
// CHECK-MESSAGES: warning: class 'B' inherits from 'A' which has a non-virtual destructor [misc-forbid-non-virtual-base-dtor]

//shouldn't warn -> derived has no data
class C : public A{};

// shouldn't warn -> base has virtual destructor
class D {
    public:
    virtual ~D(){};
};
class E : public D{
    int y;
};

//shouldn't crash -> incomplete base
class F;
class F {};
class G: public F{
    int z;
};
// CHECK-MESSAGES: warning: class 'G' inherits from 'F' which has a non-virtual destructor [misc-forbid-non-virtual-base-dtor]
