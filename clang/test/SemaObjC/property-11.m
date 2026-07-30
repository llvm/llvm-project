// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface NSSound
@end
@interface NSFont
@end

@interface NSSound (Adds)
@end

@implementation NSSound (Adds)
- foo {
  return self;
}
- (void)setFoo:obj {
}
@end

@implementation NSFont (Adds)

- xx {
  NSSound *x;
  id o;

  // GCC does *not* warn about the following. Since foo/setFoo: are not in the
  // class or category interface for NSSound, the compiler shouldn't find them.
  // For now, we will support GCC's behavior (sigh).
  o = [x foo];
  o = x.foo;
  [x setFoo:o];
  x.foo = o;
  return 0;
}

@end

