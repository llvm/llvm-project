// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck %s
// rdar://12716301

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

// CHECK: {16:21-16:21}:"["
// CHECK: {16:28-16:30}:" isEqual:"
// CHECK: {16:40-16:40}:"]"
