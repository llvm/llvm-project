// RUN: %clang_cc1  -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak  -Wdirect-ivar-access -verify -Wno-objc-root-class %s
// rdar://6505197

__attribute__((objc_root_class)) @interface MyObject {
@public
    id _myMaster;
    id _isTickledPink;
    int _myIntProp;
}
@property(retain) id myMaster;
@property(assign) id isTickledPink; // expected-note {{property declared here}}
@property int myIntProp;
@end

@implementation MyObject

@synthesize myMaster = _myMaster;
@synthesize isTickledPink = _isTickledPink; // expected-error {{existing instance variable '_isTickledPink' for property 'isTickledPink'}}
@synthesize myIntProp = _myIntProp;

- (void) doSomething {
    _myMaster = _isTickledPink; // expected-warning {{instance variable '_myMaster' is being directly accessed}} \
    // expected-warning {{instance variable '_isTickledPink' is being directly accessed}}
}

- (id) init {
    _myMaster=0;
    return _myMaster;
}
- (void) dealloc { _myMaster = 0; }
@end

MyObject * foo ()
{
	MyObject* p=0;
        p.isTickledPink = p.myMaster;	// ok
	p->_isTickledPink = (*p)._myMaster; // expected-warning {{instance variable '_isTickledPink' is being directly accessed}} \
        // expected-warning {{instance variable '_myMaster' is being directly accessed}}
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

