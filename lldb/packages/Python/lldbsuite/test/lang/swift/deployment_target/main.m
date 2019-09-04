#include <Foundation/Foundation.h>
#include <dlfcn.h>

@protocol FooProtocol
+ (void)f;
@end

int main() {
	dlopen("libNewerTarget.dylib", RTLD_LAZY);
	Class<FooProtocol> fooClass = NSClassFromString(@"NewerTarget.Foo");
	[[[fooClass alloc] init] f];
	return 0;
}

