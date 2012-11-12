// RUN: %clang_cc1 -S -g -masm-verbose -x objective-c < %s | grep DW_AT_name
@interface Foo {
  int i;
}
@property int i;
@end

@implementation Foo
@synthesize i;
@end

int bar(Foo *f) {
  int i = 1;
  f.i = 2;
  i = f.i;
  return i;
}
