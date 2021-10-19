// UNSUPPORTED: -zos, -aix
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-pch -x objective-c++ -std=c++0x -o %t %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -include-pch %t -x objective-c++ -std=c++0x -verify %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -include-pch %t -x objective-c++ -std=c++0x -ast-print %s | FileCheck -check-prefix=CHECK-PRINT %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -include-pch %t -x objective-c++ -std=c++0x -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-IR %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

typedef unsigned char BOOL;

@interface NSNumber @end

@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithUnsignedChar:(unsigned char)value;
+ (NSNumber *)numberWithShort:(short)value;
+ (NSNumber *)numberWithUnsignedShort:(unsigned short)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithUnsignedInt:(unsigned int)value;
+ (NSNumber *)numberWithLong:(long)value;
+ (NSNumber *)numberWithUnsignedLong:(unsigned long)value;
+ (NSNumber *)numberWithLongLong:(long long)value;
+ (NSNumber *)numberWithUnsignedLongLong:(unsigned long long)value;
+ (NSNumber *)numberWithFloat:(float)value;
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(BOOL)value;
@end

@interface NSArray
@end

@interface NSArray (NSArrayCreation)
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
@end

template<typename T, typename U>
struct pair {
  T first;
  U second;
};

template<typename T, typename U>
pair<T, U> make_pair(const T& first, const U& second) {
  return { first, second };
}

// CHECK-IR: define linkonce_odr {{.*}}void @_Z29variadic_dictionary_expansionIJP8NSStringS1_EJP8NSNumberS3_EEvDp4pairIT_T0_E
template<typename ...Ts, typename ... Us>
void variadic_dictionary_expansion(pair<Ts, Us>... key_values) {
  // CHECK-PRINT: id dict = @{ key_values.first : key_values.second... };
  // CHECK-IR: {{call.*objc_msgSend}}
  // CHECK-IR: ret void
  id dict = @{ key_values.first : key_values.second ... };
}

#else
void test_all() {
  variadic_dictionary_expansion(make_pair(@"Seventeen", @17), 
                                make_pair(@"YES", @true));
}
#endif
