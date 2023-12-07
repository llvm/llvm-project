// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<class T>
class TNSAutoRef
{
public:
	TNSAutoRef(T t)
		:	fRef(t)
		{ }

	~TNSAutoRef()
		{  }

	operator T() const
		{ return fRef; }
	
	T Get() const
		{ return fRef; }

private:
	T fRef;
};

@interface NSObject
- (id) alloc;
- (id)init;
@end

@interface TFoo : NSObject
- (void) foo;
@end

@implementation TFoo
- (void) foo {}
@end

@interface TBar : NSObject
- (void) foo;
@end

@implementation TBar 
- (void) foo {}
@end

int main () {
	TNSAutoRef<TBar*> bar([[TBar alloc] init]);
	[bar foo];
	return 0;
}
