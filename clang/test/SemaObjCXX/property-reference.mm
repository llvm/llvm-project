// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://9070460

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


// <rdar://problem/11052352>
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
