// RUN: cp %s %t
// RUN: %clang_cc1 -x objective-c -fixit %t
// RUN: diff %t %s

#define nil (void *)0

@interface NSObject
- (void)testDataSource:(id)object withMultipleArguments:(id)arguments;
@end

int main(void) {
  id obj;
  [obj TestDataSource:nil withMultipleArguments:nil];
}
