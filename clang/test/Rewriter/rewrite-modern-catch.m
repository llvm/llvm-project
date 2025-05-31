// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions  -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

void foo(id arg);

@interface NSException
@end

@interface Foo
@end

@implementation Foo
- (void)bar {
    @try {
    } @catch (NSException *e) {
	foo(e);
    }
    @catch (Foo *f) {
    }
    @catch (...) {
      @try {
      }
      @catch (Foo *f1) {
	foo(f1);
      }
      @catch (id pid) {
	foo(pid);
      }
    }
}
@end
