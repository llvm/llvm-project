// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions  -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;

void FINALLY();
void TRY();
void INNER_FINALLY();
void INNER_TRY();
void CHECK();

@interface Foo
@end

@implementation Foo
- (void)bar {
    @try {
	TRY();
    } 
    @finally {
	FINALLY();
    }
    CHECK();
    @try {
	TRY();
    } 
    @finally {
      @try {
        INNER_TRY();
      }
      @finally {
        INNER_FINALLY();
      }
      FINALLY();
    }
}
@end
