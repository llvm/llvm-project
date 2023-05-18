// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics
struct HasValueType {
  typedef int value_type;
};

__attribute__((objc_root_class))
@interface Foo
{
@protected
    HasValueType foo;
}

@property (nonatomic) HasValueType bar;
@end

@implementation Foo
@synthesize bar;

- (void)test {
  decltype(foo)::value_type vt1;
  decltype(self->foo)::value_type vt2;
  decltype(self.bar)::value_type vt3;
}
@end
