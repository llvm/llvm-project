// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang_cc1 -triple %itanium_abi_triple -S -debug-info-kind=limited -x objective-c < %s | grep DW_AT_name
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
