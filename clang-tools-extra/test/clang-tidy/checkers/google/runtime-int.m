// RUN: clang-tidy -checks=-*,google-runtime-int %s 2>&1 -- | count 0
// RUN: clang-tidy -checks=-*,google-runtime-int %s 2>&1 -- -x objective-c++ | count 0

typedef long NSInteger;
typedef unsigned long NSUInteger;

@interface NSString
@property(readonly) NSInteger integerValue;
@property(readonly) long long longLongValue;
@property(readonly) NSUInteger length;
@end

NSInteger Foo(NSString *s) {
  return [s integerValue];
}

long long Bar(NSString *s) {
  return [s longLongValue];
}

NSUInteger Baz(NSString *s) {
  return [s length];
}

unsigned short NSSwapShort(unsigned short inv);

long DoSomeMath(long a, short b) {
  short c = NSSwapShort(b);
  long a2 = a * 5L;
  return a2 + c;
}

