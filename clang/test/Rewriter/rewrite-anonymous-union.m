// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  -o - %s
// rdar://6948022

typedef unsigned int uint32_t;

typedef struct {
    union {
        uint32_t daysOfWeek;
        uint32_t dayOfMonth;
    };
    uint32_t nthOccurrence;
} OSPatternSpecificData;

@interface NSNumber
+ (NSNumber *)numberWithLong:(long)value;
@end

@interface OSRecurrence  {
    OSPatternSpecificData _pts;
}
- (void)_setTypeSpecificInfoOnRecord;
@end

@implementation OSRecurrence
- (void)_setTypeSpecificInfoOnRecord
{
    [NSNumber numberWithLong:(_pts.dayOfMonth >= 31 ? -1 : _pts.dayOfMonth)];
}
@end

