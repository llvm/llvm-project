// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface Test {}
+ (Test*)one;
- (int)two;
@end

int main (void)
{
  return Test.one.two;
}

__attribute__((objc_root_class))
@interface RootClass { 
  Class isa; 
}

@property int property;
-(int)method;
- (void) setMethod : (int)arg;
+(int)classMethod;
@end

@interface Subclass : RootClass @end
void Test1(void) { 
    // now okay
    (void)RootClass.property;
    (void)Subclass.property;
    (void)RootClass.method;
    (void)Subclass.method;

    RootClass.property = 1;
    Subclass.property = 2;
    RootClass.method = 3;
    Subclass.method = 4;

    // okay
    (void)RootClass.classMethod;
    (void)Subclass.classMethod;

    // also okay
    (void)[RootClass property];
    (void)[Subclass property];
    [RootClass method];
    [Subclass method];
    [RootClass classMethod];
    [Subclass classMethod];

    // also okay
    [RootClass setProperty : 1];
    [Subclass setProperty : 2];
    [RootClass setMethod : 3];
    [Subclass setMethod : 4];
}
