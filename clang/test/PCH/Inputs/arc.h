// Header for Objective-C ARC-related PCH tests

typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

CFTypeRef CFCreateSomething();
CFStringRef CFCreateString();
CFTypeRef CFGetSomething();
CFStringRef CFGetString();

@interface NSString
@end

id CreateSomething();
NSString *CreateNSString();

#if __has_feature(objc_arc)
#define BRIDGE __bridge
#else
#define BRIDGE
#endif

typedef int array0[sizeof((BRIDGE id)CFCreateSomething())];
typedef int array1[sizeof((BRIDGE CFTypeRef)CreateSomething())];


