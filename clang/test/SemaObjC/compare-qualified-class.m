// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://8191774

@protocol SomeProtocol
@end

@protocol SomeProtocol1
@end

@interface SomeObject <SomeProtocol>
@end

int main () {
    Class <SomeProtocol> classA;
    Class <SomeProtocol> classB;
    Class <SomeProtocol, SomeProtocol1> classC;
    Class <SomeProtocol1> classD;
    void * pv = 0;
    Class c = (Class)0;;
    if (pv)
      return classA == pv;

    if (c)
      return classA == c;
    
    return classA == classB  || classA == classC ||
           classC == classA ||
           classA == classD; // expected-warning {{comparison of distinct pointer types ('Class<SomeProtocol> *' and 'Class<SomeProtocol1> *')}}
}

