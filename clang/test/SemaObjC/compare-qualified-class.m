// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol SomeProtocol
@end

@protocol SomeProtocol1
@end

@interface SomeObject <SomeProtocol>
@end

int main (void) {
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
           classA == classD; // expected-warning {{comparison of distinct pointer types ('Class<SomeProtocol>' and 'Class<SomeProtocol1>')}}
}

@protocol NSObject @end

@interface NSObject @end
@protocol ProtocolX <NSObject>
@end

@protocol ProtocolY <NSObject>
@end

@interface ClassA : NSObject
@end

@interface ClassB : ClassA <ProtocolY, ProtocolX>
@end

@interface OtherClass : NSObject
@property (nonatomic, copy) ClassB<ProtocolX> *aProperty;
- (ClassA<ProtocolY> *)aMethod;
- (ClassA<ProtocolY> *)anotherMethod;
@end

@implementation OtherClass
- (ClassA<ProtocolY> *)aMethod {
    // This does not work, even though ClassB subclasses from A and conforms to Y
    // because the property type explicitly adds ProtocolX conformance
    // even though ClassB already conforms to ProtocolX
    return self.aProperty;
}
- (ClassA<ProtocolY> *)anotherMethod {
    // This works, even though all it is doing is removing an explicit
    // protocol conformance that ClassB already conforms to
    return (ClassB *)self.aProperty;
}
@end
