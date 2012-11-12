// Objective-C recovery
// RUN: cp %s %t
// RUN: not %clang_cc1 -fixit -x objective-c %t
// RUN: %clang_cc1 -fsyntax-only -Werror -x objective-c %t

// Objective-C++ recovery
// RUN: cp %s %t
// RUN: not %clang_cc1 -fixit -x objective-c++ %t
// RUN: %clang_cc1 -fsyntax-only -Werror -x objective-c++ %t
// rdar://9603056

@interface S @end

@interface NSArray
{
@public
  S iS;
}
+ (id) arrayWithObjects;
@end

NSArray func() {
  NSArray P;
  return P;
}

int main() {
  	NSArray pluginNames = [NSArray arrayWithObjects];
}
