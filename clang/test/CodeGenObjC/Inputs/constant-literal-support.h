#ifndef OBJC_CONSTANT_LITERAL_SUPPORT_H
#define OBJC_CONSTANT_LITERAL_SUPPORT_H

typedef unsigned long NSUInteger;
typedef long NSInteger;
typedef signed char BOOL;

@protocol NSCopying
@end

@interface NSNumber <NSCopying>
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
+ (NSNumber *)numberWithInteger:(NSInteger)value;
+ (NSNumber *)numberWithUnsignedInteger:(NSUInteger)value;
@end

@interface NSArray
+ (id)arrayWithObjects:(const id _Nonnull[_Nonnull])objects
                 count:(NSUInteger)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id[])objects
                    forKeys:(const id<NSCopying>[])keys
                      count:(NSUInteger)cnt;
@end

@interface NSString <NSCopying>
@end

@interface NSConstantIntegerNumber : NSNumber {
@public
  char const *const _encoding;
  long long const _value;
}
@end

@interface NSConstantFloatNumber : NSNumber {
@public
  float const _value;
}
@end

@interface NSConstantDoubleNumber : NSNumber {
@public
  double const _value;
}
@end

@interface NSConstantArray : NSArray {
@public
  unsigned long long const _count;
  id const *const _objects;
}
@end

@interface NSConstantDictionary : NSDictionary {
@public
  unsigned int const _hashOptions;
  unsigned int const _count;
  id const *const _keys;
  id const *const _objects;
}
@end

#endif // OBJC_CONSTANT_LITERAL_SUPPORT_H
