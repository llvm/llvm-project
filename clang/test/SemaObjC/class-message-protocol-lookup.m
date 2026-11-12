// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s

@interface RandomObject {
@private
}
+ (id)alloc;
@end

@protocol TestProtocol
- (void)nothingInteresting;
@end

@protocol Test2Protocol
+ (id)alloc;
- (id)alloc2; // expected-note 2 {{method 'alloc2' declared here}}
@end

@implementation RandomObject
- (void) Meth {
    Class<TestProtocol> c = [c alloc]; //  expected-warning {{class method '+alloc' not found (return type defaults to 'id')}}
    Class<Test2Protocol> c1 = [c1 alloc2]; //  expected-warning {{instance method 'alloc2' found instead of class method 'alloc2'}}
    Class<Test2Protocol> c2 = [c2 alloc]; //  ok
}
+ (id)alloc { return 0; }
@end

int main (void)
{
    Class<TestProtocol> c = [c alloc]; //  expected-warning {{class method '+alloc' not found (return type defaults to 'id')}}
    Class<Test2Protocol> c1 = [c1 alloc2]; //  expected-warning {{instance method 'alloc2' found instead of class method 'alloc2'}}
    Class<Test2Protocol> c2 = [c2 alloc]; //  ok
    return 0;
}

@protocol NSObject

- (int)respondsToSelector:(SEL)aSelector;

@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>

@end

@protocol OtherProto

- (void)otherInstanceMethod; // expected-note {{method 'otherInstanceMethod' declared here}}

@end

@protocol MyProto <NSObject, OtherProto>
@end

void allowInstanceMethodsFromRootProtocols(Class<MyProto> c) {
  [c respondsToSelector: @selector(instanceMethod)]; // no warning
  [c otherInstanceMethod]; //  expected-warning {{instance method 'otherInstanceMethod' found instead of class method 'otherInstanceMethod'}}
}
