// RUN: %clang_cc1 -std=c++11 %s -verify

// expected-no-diagnostics
@protocol NSObject
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
@end

@interface NSString : NSObject
@end

// --------------------------------------------------------------------------
// Parsing parameterized classes.
// --------------------------------------------------------------------------
@interface PC1<T, U, V> : NSObject
@end

// --------------------------------------------------------------------------
// Parsing type arguments.
// --------------------------------------------------------------------------
typedef PC1<::NSString *, NSString *, id<NSCopying>> typeArgs1;
