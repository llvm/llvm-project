// RUN: %clang_cc1  -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak  -Wdirect-ivar-access -verify -Wno-objc-root-class %s
// rdar://6505197

__attribute__((objc_root_class)) @interface MyObject {
@public
    id _myLeader;
    id _isTickledPink; // expected-error {{existing instance variable '_isTickledPink' for property 'isTickledPink'}}
    int _myIntProp;
}
@property(retain) id myLeader;
@property(assign) id isTickledPink; // expected-note {{property declared here}}
@property int myIntProp;
@end

@implementation MyObject

@synthesize myLeader = _myLeader;
@synthesize isTickledPink = _isTickledPink; // expected-note {{property synthesized here}}
@synthesize myIntProp = _myIntProp;

- (void) doSomething {
    _myLeader = _isTickledPink; // expected-warning {{instance variable '_myLeader' is being directly accessed}} \
    // expected-warning {{instance variable '_isTickledPink' is being directly accessed}}
}

- (id) init {
    _myLeader=0;
    return _myLeader;
}
- (void) dealloc { _myLeader = 0; }
@end

MyObject * foo (void)
{
	MyObject* p=0;
        p.isTickledPink = p.myLeader;	// ok
	p->_isTickledPink = (*p)._myLeader; // expected-warning {{instance variable '_isTickledPink' is being directly accessed}} \
        // expected-warning {{instance variable '_myLeader' is being directly accessed}}
        if (p->_myIntProp) // expected-warning {{instance variable '_myIntProp' is being directly accessed}}
          p->_myIntProp = 0; // expected-warning {{instance variable '_myIntProp' is being directly accessed}}
	return p->_isTickledPink; // expected-warning {{instance variable '_isTickledPink' is being directly accessed}}
}

@interface ITest32 {
@public
 id ivar;
}
@end

id Test32(__weak ITest32 *x) {
  __weak ITest32 *y;
  x->ivar = 0; // expected-error {{dereferencing a __weak pointer is not allowed}}
  return y ? y->ivar     // expected-error {{dereferencing a __weak pointer is not allowed}}
           : (*x).ivar;  // expected-error {{dereferencing a __weak pointer is not allowed}}
}

// rdar://13142820
@protocol PROTOCOL
@property (copy, nonatomic) id property_in_protocol;
@end

__attribute__((objc_root_class)) @interface INTF <PROTOCOL>
@property (copy, nonatomic) id foo;
- (id) foo;
@end

@interface INTF()
@property (copy, nonatomic) id foo1;
- (id) foo1;
@end

@implementation INTF
- (id) foo { return _foo; }
- (id) property_in_protocol { return _property_in_protocol; } // expected-warning {{instance variable '_property_in_protocol' is being directly accessed}}
- (id) foo1 { return _foo1; }
@synthesize property_in_protocol = _property_in_protocol;
@end

