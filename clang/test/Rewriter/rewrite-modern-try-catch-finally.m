// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions  -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;

extern int printf(const char *, ...);

int main() {
  @try {
  } 
  @finally {
  }
  while (1) {
    @try {
      printf("executing try");
      break;
    } @finally {
      printf("executing finally");
    }
    printf("executing after finally block");
  }
  @try {
    printf("executing try");
  } @finally {
    printf("executing finally");
  }
  return 0;
}

void test2_try_with_implicit_finally() {
    @try {
        return;
    } @catch (id e) {
        
    }
}

void FINALLY();
void TRY();
void CATCH();

@interface NSException
@end

@interface Foo
@end

@implementation Foo
- (void)bar {
    @try {
	TRY();
    } 
    @catch (NSException *e) {
	CATCH();
    }
    @finally {
	FINALLY();
    }
}
@end
