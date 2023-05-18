// RUN: %clang_analyze_cc1 -fobjc-arc -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached(void);

extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));

#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))


@protocol NSObject
+ (nonnull instancetype)alloc;
- (nonnull instancetype)init;
@end
@interface NSObject <NSObject> {}
@end

@interface NSError : NSObject {
@public
  int x;
}
@end


@interface SomeClass : NSObject
+ (int)doSomethingWithError:(NSError *__autoreleasing *)error;
@end

@implementation SomeClass
+ (int)doSomethingWithError:(NSError *__autoreleasing *)error {
    if (error) {
        NSError *e = [[NSError alloc] init];
        assert(e);
        e->x = 5;
        *error = e;
        clang_analyzer_eval(*error != 0); // expected-warning{{TRUE}}
    }
    return 0;
}
@end

void testStrongOutParam(void) {
  NSError *error;
  clang_analyzer_eval(error != 0); // expected-warning{{FALSE}}
  int ok = [SomeClass doSomethingWithError:&error];
  clang_analyzer_eval(ok);         // expected-warning{{FALSE}}
  clang_analyzer_eval(error != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(error->x == 5); // expected-warning{{TRUE}}
}

void testAutoreleasingOutParam(void) {
  NSError *__autoreleasing error;
  clang_analyzer_eval(error != 0); // expected-warning{{FALSE}}
  int ok = [SomeClass doSomethingWithError:&error];
  clang_analyzer_eval(ok);         // expected-warning{{FALSE}}
  clang_analyzer_eval(error != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(error->x == 5); // expected-warning{{TRUE}}
}

void testNilOutParam(void) {
    int ok = [SomeClass doSomethingWithError:(void *)0];
    clang_analyzer_eval(ok);  // expected-warning{{FALSE}}
}

