// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fsyntax-only -verify -Wno-objc-root-class %s

class TCPPObject
{
public:
	TCPPObject(const TCPPObject& inObj);
	TCPPObject();
	~TCPPObject();
	
	TCPPObject& operator=(const TCPPObject& inObj)const ; // expected-note {{'operator=' declared here}}

	void* Data();
	
private:
	void* fData;
};


typedef const TCPPObject& CREF_TCPPObject;

@interface TNSObject
@property (assign, readwrite, nonatomic) CREF_TCPPObject cppObjectNonAtomic;
@property (assign, readwrite) CREF_TCPPObject cppObjectAtomic;
@property (assign, readwrite, nonatomic) const TCPPObject& cppObjectDynamic;
@end


@implementation TNSObject

@synthesize cppObjectNonAtomic;
@synthesize cppObjectAtomic; // expected-error{{atomic property of reference type 'CREF_TCPPObject' (aka 'const TCPPObject &') cannot have non-trivial assignment operator}}
@dynamic cppObjectDynamic;

- (const TCPPObject&) cppObjectNonAtomic
{
	return cppObjectNonAtomic;
}

- (void) setCppObjectNonAtomic: (const TCPPObject&)cppObject
{
	cppObjectNonAtomic = cppObject;
}
@end


@interface NSObject
+ alloc;
- init;
- class;
@end

template<typename T> void f() {
  NSObject *o = [NSObject.alloc init];
  [o class];
}

template void f<int>();

// Make sure that the default-argument checker looks through
// pseudo-object expressions correctly.  The default argument
// needs to force l2r to test this effectively because the checker
// is syntactic and runs before placeholders are handled.
@interface Test13602832
- (int) x;
@end
namespace test13602832 {
  template <int N> void foo(Test13602832 *a, int limit = a.x + N) {} // expected-error {{default argument references parameter 'a'}}

  void test(Test13602832 *a) {
    // FIXME: this is a useless cascade error.
    foo<1024>(a); // expected-error {{no matching function}}
  }
}
