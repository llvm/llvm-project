// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://16462586

__attribute__((objc_runtime_name("MySecretNamespace.Protocol")))
@protocol Protocol
@end

__attribute__((objc_runtime_name("MySecretNamespace.Message")))
@interface Message <Protocol> { 
__attribute__((objc_runtime_name("MySecretNamespace.Message"))) // expected-error {{'objc_runtime_name' attribute only applies to Objective-C interfaces and Objective-C protocols}}
  id MyIVAR;
}
__attribute__((objc_runtime_name("MySecretNamespace.Message")))
@property int MyProperty; // expected-error {{prefix attribute must be followed by an interface or protocol}}}}

- (int) getMyProperty __attribute__((objc_runtime_name("MySecretNamespace.Message"))); // expected-error {{'objc_runtime_name' attribute only applies to}}

- (void) setMyProperty : (int) arg __attribute__((objc_runtime_name("MySecretNamespace.Message"))); // expected-error {{'objc_runtime_name' attribute only applies to}}

@end

__attribute__((objc_runtime_name("MySecretNamespace.ForwardClass")))
@class ForwardClass; // expected-error {{prefix attribute must be followed by an interface or protocol}}

__attribute__((objc_runtime_name("MySecretNamespace.ForwardProtocol")))
@protocol ForwardProtocol;

__attribute__((objc_runtime_name("MySecretNamespace.Message")))
@implementation Message // expected-error {{prefix attribute must be followed by an interface or protocol}}
__attribute__((objc_runtime_name("MySecretNamespace.Message")))
- (id) MyMethod {
  return MyIVAR;
}
@end
