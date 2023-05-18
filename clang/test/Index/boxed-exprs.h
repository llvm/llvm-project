
@interface NSString @end

@interface NSString (NSStringExtensionMethods)
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end

static inline void infoo(const char *cs) {
  NSString *s = @(cs);
}
