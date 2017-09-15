// RUN: %clang_cc1 %s -verify -fsyntax-only

class ClassA {};

class ClassB {
public:
  ClassB(ClassA* parent=0);
  ~ClassB();
};

@interface NSObject
@end

@interface InterfaceA : NSObject
@property(nonatomic, assign) ClassA *m_prop1; // expected-note {{here}}
@property(nonatomic, assign) ClassB *m_prop2;
@end

@implementation InterfaceA
- (id)test {
  self.m_prop2 = new ClassB(m_prop1); // expected-error {{use of undeclared identifier 'm_prop1'; did you mean '_m_prop1'?}}
}
@end

// rdar://30310772

@interface InvalidNameInIvarAndPropertyBase
{
@public
  float  _a;
}
@property float _b;
@end

void invalidNameInIvarAndPropertyBase() {
  float a = ((InvalidNameInIvarAndPropertyBase*)node)->_a; // expected-error {{use of undeclared identifier 'node'}}
  float b = ((InvalidNameInIvarAndPropertyBase*)node)._b; // expected-error {{use of undeclared identifier 'node'}}
}

// rdar://problem/33102722
// Typo correction for a property when it has as correction candidates
// synthesized ivar and a class name, both at the same edit distance.
@class TypoCandidate;

@interface PropertyType : NSObject
@property int x;
@end

@interface InterfaceC : NSObject
@property(assign) PropertyType *typoCandidate; // expected-note {{'_typoCandidate' declared here}}
@end

@implementation InterfaceC
-(void)method {
  typoCandidate.x = 0; // expected-error {{use of undeclared identifier 'typoCandidate'; did you mean '_typoCandidate'?}}
}
@end
