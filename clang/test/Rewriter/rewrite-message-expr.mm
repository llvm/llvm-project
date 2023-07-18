// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix CHECK-LP --input-file=%t-rw.cpp %s
// radar 7617047

@interface Baz
- (id)y;
+ (id)z;
@end

@interface Foo {
@public
	int bar;
}
@end

extern Foo* x(id a);

int f(Baz *baz) {
	int i = x([Baz z])->bar;
        int j = ((Foo*)[Baz z])->bar;
        int k = x([baz y])->bar;
        return i+j+k;
}

// CHECK-LP: ((struct Foo_IMPL *)x(((id (*)(id, SEL))(void *)objc_msgSend)(objc_getClass("Baz"), sel_registerName("z"))))->bar
