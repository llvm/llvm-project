// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -verify %s
// expected-no-diagnostics

// It tests path-sensitivity with respect to '!(cfstring != 0)' (negation of inequality).

int printf(const char *restrict,...);
typedef unsigned long UInt32;
typedef signed long SInt32;
typedef SInt32  OSStatus;
typedef unsigned char Boolean;
enum { noErr = 0};
typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;
typedef const struct __CFAllocator *CFAllocatorRef;
extern void     CFRelease(CFTypeRef cf);
typedef UInt32  CFStringEncoding;
enum { kCFStringEncodingMacRoman = 0, kCFStringEncodingWindowsLatin1 = 0x0500,
       kCFStringEncodingISOLatin1 = 0x0201, kCFStringEncodingNextStepLatin = 0x0B01,
       kCFStringEncodingASCII = 0x0600, kCFStringEncodingUnicode = 0x0100,
       kCFStringEncodingUTF8 = 0x08000100, kCFStringEncodingNonLossyASCII = 0x0BFF,
       kCFStringEncodingUTF16 = 0x0100, kCFStringEncodingUTF16BE = 0x10000100,
       kCFStringEncodingUTF16LE = 0x14000100, kCFStringEncodingUTF32 = 0x0c000100,
       kCFStringEncodingUTF32BE = 0x18000100, kCFStringEncodingUTF32LE = 0x1c000100};
extern CFStringRef CFStringCreateWithCString(CFAllocatorRef alloc, const char *cStr, CFStringEncoding encoding);

enum { memROZWarn = -99, memROZError = -99, memROZErr = -99, memFullErr = -108,
       nilHandleErr = -109, memWZErr = -111, memPurErr = -112, memAdrErr = -110,
       memAZErr = -113, memPCErr = -114, memBCErr = -115, memSCErr = -116, memLockedErr = -117};

#define DEBUG1

void            DebugStop(const char *format,...);
void            DebugTraceIf(unsigned int condition, const char *format,...);
Boolean         DebugDisplayOSStatusMsg(OSStatus status, const char *statusStr, const char *fileName, unsigned long lineNumber);

#define Assert(condition)if (!(condition)) { DebugStop("Assertion failure: %s [File: %s, Line: %lu]", #condition, __FILE__, __LINE__); }
#define AssertMsg(condition, message)if (!(condition)) { DebugStop("Assertion failure: %s (%s) [File: %s, Line: %lu]", #condition, message, __FILE__, __LINE__); }
#define Require(condition)if (!(condition)) { DebugStop("Assertion failure: %s [File: %s, Line: %lu]", #condition, __FILE__, __LINE__); }
#define RequireAction(condition, action)if (!(condition)) { DebugStop("Assertion failure: %s [File: %s, Line: %lu]", #condition, __FILE__, __LINE__); action }
#define RequireActionSilent(condition, action)if (!(condition)) { action }
#define AssertNoErr(err){ DebugDisplayOSStatusMsg((err), #err, __FILE__, __LINE__); }
#define RequireNoErr(err, action){ if( DebugDisplayOSStatusMsg((err), #err, __FILE__, __LINE__) ) { action }}

void DebugStop(const char *format,...); /* Not an abort function. */

int main(int argc, char *argv[]) {
  CFStringRef     cfString;
  OSStatus        status = noErr;
  cfString = CFStringCreateWithCString(0, "hello", kCFStringEncodingUTF8);
  RequireAction(cfString != 0, return memFullErr;) //no - warning
    printf("cfstring %p\n", cfString);
  Exit:
  CFRelease(cfString);
  return 0;
}
