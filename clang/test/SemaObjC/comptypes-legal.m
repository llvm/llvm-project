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
void func(void) {
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
  // GCC currently allows this (it has some fairly new support for covariant return types and contravariant argument types).
  // Since registerFunc: expects a Derived object as it's second argument, I don't know why this would be legal.
  [Derived registerFunc: ExternFunc];  // expected-error{{incompatible function pointer types sending 'NSObject *(NSObject *, NSObject *)' to parameter of type 'FuncSignature *' (aka 'id (*)(NSObject *, Derived *)')}}
}

// rdar://10751015
@protocol NSCopying @end
@interface I
- (void) Meth : (id <NSCopying>)aKey; // expected-note {{passing argument to parameter 'aKey' here}}
@end

@class ForwarClass; // expected-note 3 {{conformance of forward class 'ForwarClass' to protocol 'NSCopying' can not be confirmed}}

ForwarClass *Test10751015 (I* pi, ForwarClass *ns_forward) {

  [pi Meth : ns_forward ]; // expected-warning {{sending 'ForwarClass *' to parameter of incompatible type 'id<NSCopying>'}}

  id <NSCopying> id_ns = ns_forward; // expected-warning {{initializing 'id<NSCopying>' with an expression of incompatible type 'ForwarClass *'}}

  return id_ns; // expected-warning {{returning 'id<NSCopying>' from a function with incompatible result type 'ForwarClass *'}}
}
