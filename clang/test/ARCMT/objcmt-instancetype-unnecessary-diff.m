// RUN: %clang_cc1 -objcmt-migrate-instancetype %s -triple x86_64-apple-darwin11 -fobjc-arc -migrate -o %t.remap
// RUN: FileCheck %s -input-file=%t.remap

// Make sure we don't create an edit unnecessarily.
// CHECK-NOT: instancetype

@class NSString;
@interface NSDictionary
+(instancetype) dictionaryWithURLEncodedString:(NSString *)urlEncodedString;
@end
