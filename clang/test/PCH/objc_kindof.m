// RUN: %clang_cc1 -emit-pch %s -o %t
// RUN: %clang_cc1 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED
@protocol NSObject
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
@end

@interface NSString : NSObject <NSCopying>
@end

@interface NSMutableString : NSString
@end

@interface NSNumber : NSObject <NSCopying>
@end

extern __kindof NSObject <NSCopying> *kindof_NSObject_NSCopying;

#else
void testPrettyPrint(int *ip) {
  ip = kindof_NSObject_NSCopying; // expected-warning{{from '__kindof NSObject<NSCopying> *'}}
}

#endif
