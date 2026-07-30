// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount,alpha.core -verify %s
// expected-no-diagnostics

typedef unsigned char Boolean;
typedef signed long CFIndex;
typedef const void * CFTypeRef;
typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
typedef struct {} CFAllocatorContext;
extern void CFRelease(CFTypeRef cf);
typedef struct {}
CFDictionaryKeyCallBacks;
extern const CFDictionaryKeyCallBacks kCFTypeDictionaryKeyCallBacks;
typedef struct {}
CFDictionaryValueCallBacks;
extern const CFDictionaryValueCallBacks kCFTypeDictionaryValueCallBacks;
typedef const struct __CFDictionary * CFDictionaryRef;
extern CFDictionaryRef CFDictionaryCreate(CFAllocatorRef allocator, const void **keys, const void **values, CFIndex numValues, const CFDictionaryKeyCallBacks *keyCallBacks, const CFDictionaryValueCallBacks *valueCallBacks);
enum { kCFNumberSInt8Type = 1,     kCFNumberSInt16Type = 2,     kCFNumberSInt32Type = 3,     kCFNumberSInt64Type = 4,     kCFNumberFloat32Type = 5,     kCFNumberFloat64Type = 6,      kCFNumberCharType = 7,     kCFNumberShortType = 8,     kCFNumberIntType = 9,     kCFNumberLongType = 10,     kCFNumberLongLongType = 11,     kCFNumberFloatType = 12,     kCFNumberDoubleType = 13,      kCFNumberCFIndexType = 14,      kCFNumberNSIntegerType = 15,     kCFNumberCGFloatType = 16,     kCFNumberMaxType = 16    };
typedef CFIndex CFNumberType;
typedef const struct __CFNumber * CFNumberRef;
extern CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);
typedef struct __CFNotificationCenter * CFNotificationCenterRef;
extern CFNotificationCenterRef CFNotificationCenterGetDistributedCenter(void);
extern void CFNotificationCenterPostNotification(CFNotificationCenterRef center, CFStringRef name, const void *object, CFDictionaryRef userInfo, Boolean deliverImmediately);

// This test case was reported in PR2519 as a false positive (_value was
// reported as being leaked).

int main(int argc, char **argv) {
 CFStringRef _key = ((CFStringRef) __builtin___CFStringMakeConstantString ("" "Process identifier" ""));
 int pid = 42;

 CFNumberRef _value = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &pid);
 CFDictionaryRef userInfo = CFDictionaryCreate(kCFAllocatorDefault, (const void **)&_key, (const void **)&_value, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
 CFRelease(_value); // no-warning
 CFNotificationCenterPostNotification(CFNotificationCenterGetDistributedCenter(),
           ((CFStringRef) __builtin___CFStringMakeConstantString ("" "GrowlPreferencesChanged" "")),
           ((CFStringRef) __builtin___CFStringMakeConstantString ("" "GrowlUserDefaults" "")),
           userInfo, 0);
 CFRelease(userInfo); // no-warning

 return 0;
}

