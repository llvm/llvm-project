// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@protocol NSObject @end
@interface NSObject
- (id)init;
- (id) alloc;
- (id) autorelease;
@end

template<class T>
class TNSAutoRef
{
public:
	TNSAutoRef(T t)
		:	fRef(t)
		{ }

	~TNSAutoRef()
		{ }

	operator T() const
		{ return fRef; }

private:
	T fRef;
};


#pragma mark -


@protocol TFooProtocol <NSObject>

- (void) foo;
@end


#pragma mark -


@interface TFoo : NSObject

- (void) setBlah: (id<TFooProtocol>)blah;
@end


#pragma mark -


@implementation TFoo

- (void) setBlah: (id<TFooProtocol>)blah
	{ }
@end


#pragma mark -


@interface TBar : NSObject

- (void) setBlah: (id)blah;
@end

#pragma mark -


@implementation TBar

- (void) setBlah: (id)blah
	{ }
@end


#pragma mark -


int main (int argc, const char * argv[]) {

	NSObject* object1 = [[[NSObject alloc] init] autorelease];
	TNSAutoRef<NSObject*> object2([[NSObject alloc] init]);
	TNSAutoRef<TBar*> bar([[TBar alloc] init]);
	[bar setBlah: object1];				// <== Does not compile.  It should.
        if (object1 == object2)
	  [bar setBlah: object2];				// <== Does not compile.  It should.
	return 0;
}
