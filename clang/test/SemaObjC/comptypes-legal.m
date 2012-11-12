// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

@protocol NSObject
@end
@interface NSObject <NSObject> {
}
@end
@interface NSString : NSObject
@end
void __setRetained(id *ivar, id value, NSObject **o) {
    *ivar = value;
}
static NSString *_logProcessPrefix = 0;
void func() {
  __setRetained(&_logProcessPrefix, _logProcessPrefix, &_logProcessPrefix);
}
@implementation NSObject (ScopeAdditions)
+ (void)setObjectLogProcessPrefix:(NSString *)processPrefix {
    __setRetained(&_logProcessPrefix, processPrefix, &_logProcessPrefix);
}
@end

@class Derived;

NSObject *ExternFunc (NSObject *filePath, NSObject *key);
typedef id FuncSignature (NSObject *arg1, Derived *arg2);

@interface Derived: NSObject
+ (void)registerFunc:(FuncSignature *)function; // expected-note{{passing argument to parameter 'function' here}}
@end

void foo(void)
{
  // GCC currently allows this (it has some fiarly new support for covariant return types and contravariant argument types).
  // Since registerFunc: expects a Derived object as it's second argument, I don't know why this would be legal.
  [Derived registerFunc: ExternFunc];  // expected-warning{{incompatible pointer types sending 'NSObject *(NSObject *, NSObject *)' to parameter of type 'FuncSignature *'}}
}
