// Objective-C recovery
// RUN: cp %s %t
// RUN: not %clang_cc1 -fixit -x objective-c %t
// RUN: %clang_cc1 -fsyntax-only -Werror -x objective-c %t

// Objective-C++ recovery
// RUN: cp %s %t
// RUN: not %clang_cc1 -fixit -x objective-c++ %t -std=c++11
// RUN: %clang_cc1 -fsyntax-only -Werror -x objective-c++ %t  -std=c++11
// rdar://9603056

@interface S @end

@interface NSArray
{
@public
  S iS;
}
+ (id) arrayWithObjects;
@end

NSArray func(void) {
  NSArray P;
  return P;
}

NSArray (func2)(void) { return 0; }

#ifdef __cplusplus
void test_result_type() {
  auto l1 = [] () -> NSArray { return 0; };
}
#endif

int main(void) {
  	NSArray pluginNames = [NSArray arrayWithObjects];
}
