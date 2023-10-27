// RUN: %clang_cc1 -Wcstring-format-directive -verify -fsyntax-only %s

typedef __builtin_va_list __darwin_va_list;
typedef __builtin_va_list va_list;

va_list argList;

typedef const struct __CFString * CFStringRef;
typedef struct __CFString * CFMutableStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;


typedef const struct __CFDictionary * CFDictionaryRef;

CFStringRef CFSTR ( const char *cStr );


extern
CFStringRef CStringCreateWithFormat(CFAllocatorRef alloc, CFDictionaryRef formatOptions, const char* format, ...) __attribute__((format(os_trace, 3, 4)));

extern
CFStringRef CStringCreateWithFormatAndArguments(CFAllocatorRef alloc, CFDictionaryRef formatOptions, const char* format, va_list arguments) __attribute__((format(os_trace, 3, 0)));

extern
void CStringAppendFormat(CFMutableStringRef theString, CFDictionaryRef formatOptions, const char* format, ...) __attribute__((format(os_trace, 3, 4)));

extern
void CStringAppendFormatAndArguments(CFMutableStringRef theString, CFDictionaryRef formatOptions, const char* format, va_list arguments) __attribute__((format(os_trace, 3, 0)));

void Test1(va_list argList) {
  CFAllocatorRef alloc;
  CStringCreateWithFormatAndArguments (alloc, 0, "%s\n", argList);
  CStringAppendFormatAndArguments ((CFMutableStringRef)@"AAAA", 0, "Hello %s there %d\n", argList);
  CStringCreateWithFormatAndArguments (alloc, 0, "%c\n", argList);
  CStringAppendFormatAndArguments ((CFMutableStringRef)@"AAAA", 0, "%d\n", argList);
}

extern void MyOSLog(const char* format, ...) __attribute__((format(os_trace, 1, 2)));
extern void MyFStringCreateWithFormat(const char *format, ...) __attribute__((format(os_trace, 1, 2)));
extern void XMyOSLog(int, const char* format, ...) __attribute__((format(os_trace, 2, 3)));
extern void os_trace(const char *format, ...) __attribute__((format(os_trace, 1, 2)));

void Test2(void) {
  MyOSLog("%s\n", "Hello");

  MyFStringCreateWithFormat("%s", "Hello"); 
  XMyOSLog(4, "%s\n", "Hello");

  os_trace("testing %@, %s, %d, %@, %m", CFSTR("object"), "string", 3, "it"); // expected-warning {{format specifies type 'id' but the argument has type 'char *'}}

  os_trace("testing %@, %s, %d, %@, %m", CFSTR("object"), "string", 3, @"ok");
}

