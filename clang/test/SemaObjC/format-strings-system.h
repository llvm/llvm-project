
#pragma clang system_header

@class NSString;

// Do not emit warnings when using NSLocalizedString
extern NSString *GetLocalizedString(NSString *str);
#define NSLocalizedString(key) GetLocalizedString(key)

#define NSAssert(fmt, arg) NSLog(fmt, arg, 0, 0)
