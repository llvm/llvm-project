// RUN: %clang_cc1 -fsyntax-only -verify %s

@class Protocol;

@protocol fproto; // expected-note {{'fproto' declared here}}

@protocol p1 
@end

@class cl;

int main()
{
	Protocol *proto = @protocol(p1);
        Protocol *fproto = @protocol(fproto); // expected-error {{@protocol is using a forward protocol declaration of 'fproto'}}
	Protocol *pp = @protocol(i); // expected-error {{cannot find protocol declaration for 'i'}}
	Protocol *p1p = @protocol(cl); // expected-error {{cannot find protocol declaration for 'cl'}}
}

// rdar://17768630
@protocol SuperProtocol; // expected-note {{'SuperProtocol' declared here}}
@protocol TestProtocol; // expected-note {{'TestProtocol' declared here}}

@interface I
- (int) conformsToProtocol : (Protocol *)protocl;
@end

int doesConform(id foo) {
  return [foo conformsToProtocol:@protocol(TestProtocol)]; // expected-error {{@protocol is using a forward protocol declaration of 'TestProtocol'}}
}

int doesConformSuper(id foo) {
  return [foo conformsToProtocol:@protocol(SuperProtocol)]; // expected-error {{@protocol is using a forward protocol declaration of 'SuperProtocol'}}
}
