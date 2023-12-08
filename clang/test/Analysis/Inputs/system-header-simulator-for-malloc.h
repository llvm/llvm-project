// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.
#pragma clang system_header

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *calloc(size_t, size_t);
void free(void *);


#if __OBJC__

#import "system-header-simulator-objc.h"

@interface Wrapper : NSData
- (id)initWithBytesNoCopy:(void *)bytes length:(NSUInteger)len;
@end

@implementation Wrapper
- (id)initWithBytesNoCopy:(void *)bytes length:(NSUInteger)len {
  return [self initWithBytesNoCopy:bytes length:len freeWhenDone:1]; // no-warning
}
@end

@interface CustomData : NSData
+ (id)somethingNoCopy:(char *)bytes;
+ (id)somethingNoCopy:(void *)bytes length:(NSUInteger)length freeWhenDone:(BOOL)freeBuffer;
+ (id)something:(char *)bytes freeWhenDone:(BOOL)freeBuffer;
@end

#endif
