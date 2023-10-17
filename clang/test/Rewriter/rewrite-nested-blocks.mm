// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp

typedef unsigned long size_t;
void f(void (^block)(void));

@interface X {
	int y;
}
- (void)foo;
@end

@implementation X
- (void)foo {
    f(^{
  f(^{
    f(^{
      y=42;
    });
  });
});

}
@end

struct S {
  int y;
};

void foo () {
	struct S *SELF;
	f(^{
		f(^{
			SELF->y = 42;
		});
	});
}

@interface Bar
@end

void f(Bar *);
void q(void (^block)(void));

void x() {
        void (^myblock)(Bar *b) = ^(Bar *b) {
                q(^{
                        f(b);   
                });
        };
        
        Bar *b = (Bar *)42;
        myblock(b);
}
