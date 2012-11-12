// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify %s
// rdar://11618852

@protocol TestProtocol 
- (void)newProtocolMethod;
- (void)deprecatedProtocolMethod __attribute__((deprecated)); // expected-note 2 {{method 'deprecatedProtocolMethod' declared here}}
@end

@interface NSObject @end

@interface TestClass : NSObject <TestProtocol>

- (void)newInstanceMethod;
- (void)deprecatedInstanceMethod __attribute__((deprecated)); // expected-note {{method 'deprecatedInstanceMethod' declared here}}

@end

int main(int argc, const char * argv[])
{

    TestClass *testObj = (TestClass*)0;
    [testObj newInstanceMethod];
    [testObj deprecatedInstanceMethod]; // expected-warning {{'deprecatedInstanceMethod' is deprecated}}

    [testObj newProtocolMethod];
    [testObj deprecatedProtocolMethod]; // expected-warning {{'deprecatedProtocolMethod' is deprecated}}

    id <TestProtocol> testProto = testObj;
    [testProto newProtocolMethod];
    [testProto deprecatedProtocolMethod]; // expected-warning {{'deprecatedProtocolMethod' is deprecated}}
    return 0;
}
