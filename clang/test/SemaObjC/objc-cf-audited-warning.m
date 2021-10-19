// RUN: %clang_cc1  -fobjc-arc -verify %s
// rdar://18222007

#if __has_feature(arc_cf_code_audited)
#define CF_IMPLICIT_BRIDGING_ENABLED _Pragma("clang arc_cf_code_audited begin")
#define CF_IMPLICIT_BRIDGING_DISABLED _Pragma("clang arc_cf_code_audited end")
#endif
#define CF_BRIDGED_TYPE(T)              __attribute__((objc_bridge(T)))

typedef const struct CF_BRIDGED_TYPE(NSURL) __CFURL * CFURLRef;
typedef signed long long CFIndex;
typedef unsigned char           Boolean;
typedef unsigned char                   UInt8;
typedef const struct __CFAllocator * CFAllocatorRef;
const CFAllocatorRef kCFAllocatorDefault;

CF_IMPLICIT_BRIDGING_ENABLED
CFURLRef CFURLCreateFromFileSystemRepresentation(CFAllocatorRef allocator, const UInt8 *buffer, CFIndex bufLen, Boolean isDirectory); // expected-note {{passing argument to parameter 'buffer' here}}
CF_IMPLICIT_BRIDGING_DISABLED

void saveImageToJPG(const char *filename)
{
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault, filename, 10, 0); // expected-warning {{passing 'const char *' to parameter of type 'const UInt8 *' (aka 'const unsigned char *') converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
}
