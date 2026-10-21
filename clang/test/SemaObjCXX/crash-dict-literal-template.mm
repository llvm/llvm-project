// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-12.0.0 -verify %s

// expected-no-diagnostics

typedef unsigned int NSUInteger;
@interface NSObject
@end
@interface NSNumber
+ numberWithUnsignedLongLong:(unsigned long long)value;
@end
@interface NSDictionary<KeyType, ObjectType> : NSObject
+ dictionaryWithObjects:(ObjectType[_Nullable])objects
                forKeys:(KeyType[_Nullable])keys
                  count:(NSUInteger)cnt;
@end

template <typename T> class C {
  C() {
    @{
      // The GNU statement-expression in a dependent context is always
      // value-dependent and instantiation-dependent. This produces a
      // somewhat unusual situation where the NSDictionary key isn't dependent
      // but the value is.
      @"key" : @(
        (({}), ((1ULL))) / 1024 // no-crash
      )
    };
  }
};
