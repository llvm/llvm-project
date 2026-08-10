// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

class CppAbstractBase {
public:
    virtual void testA() = 0;
    virtual void testB() = 0; // expected-note {{unimplemented pure virtual method 'testB' in 'CppConcreteSub}}
    int a;
};

class CppConcreteSub : public CppAbstractBase {
    virtual void testA() { }
};

@interface Objc  {
    CppConcreteSub _concrete; // expected-error{{instance variable type 'CppConcreteSub' is an abstract class}}
}
- (CppAbstractBase*)abstract;
@property (nonatomic, readonly) const CppConcreteSub Prop;  // expected-note {{property declared here}}
@end

@implementation Objc
- (CppAbstractBase*)abstract {
    return &_concrete;
}
@synthesize Prop; // expected-error {{synthesized instance variable type 'const CppConcreteSub' is an abstract class}}
@end

class Cpp {
public:
    CppConcreteSub sub; // expected-error {{field type 'CppConcreteSub' is an abstract class}}
};
