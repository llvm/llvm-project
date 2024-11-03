// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

void *sel_registerName(const char *);

typedef void (^FooBlock) (int foo, int bar, int baz);
	
@interface Foo { }
@property (readwrite, copy, nonatomic) FooBlock fooBlock;
@end
	
static void Bar (Foo * foo) {
	foo.fooBlock (1,2,3);
}
