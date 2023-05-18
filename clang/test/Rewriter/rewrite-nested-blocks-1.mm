// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp

typedef unsigned long size_t;
void *sel_registerName(const char *);

void f(void (^block)(void));
void f2(id);
void f3(int);
char f4(id, id);

@interface Baz
- (void)b:(void (^)(void))block;
@end

@interface Bar
@end

@interface Foo {
	int _x;
}
@end

@implementation Foo
- (void)method:(Bar *)up {
    Baz *down;
	int at;
    id cq;
    __block char didit = 'a';
    __block char upIsFinished = 'b';
    f(^{
            id old_cq;
			f2(cq);
            [down b:^{
                    [down b:^{
                            f(^{
                                    didit = f4(up, down);
									upIsFinished = 'c';
                                    self->_x++;
                                });
                        }];
                }];
				f2(old_cq);
			f3(at);
    });
}
@end
