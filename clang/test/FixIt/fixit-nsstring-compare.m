// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck %s

typedef unsigned char BOOL;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@interface NSString<NSObject>
@end

int main(void) {
  NSString *stringA = @"stringA";

  BOOL comparison = stringA==@"stringB";

}

// CHECK: {15:21-15:21}:"["
// CHECK: {15:28-15:30}:" isEqual:"
// CHECK: {15:40-15:40}:"]"
