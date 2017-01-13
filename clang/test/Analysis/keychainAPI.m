// RUN: %clang_cc1 -analyze -analyzer-checker=osx.SecKeychainAPI -fblocks %s -verify

#include "Inputs/system-header-simulator-objc.h"

// Fake typedefs.
typedef unsigned int OSStatus;
typedef unsigned int SecKeychainAttributeList;
typedef unsigned int SecKeychainItemRef;
typedef unsigned int SecItemClass;
typedef unsigned int UInt32;
typedef unsigned int SecProtocolType;
typedef unsigned int SecAuthenticationType;
typedef unsigned int SecKeychainAttributeInfo;
enum {
  noErr                      = 0,
  GenericError               = 1
};

// Functions that allocate data.
OSStatus SecKeychainItemCopyContent (
    SecKeychainItemRef itemRef,
    SecItemClass *itemClass,
    SecKeychainAttributeList *attrList,
    UInt32 *length,
    void **outData
);
OSStatus SecKeychainFindGenericPassword (
    CFTypeRef keychainOrArray,
    UInt32 serviceNameLength,
    const char *serviceName,
    UInt32 accountNameLength,
    const char *accountName,
    UInt32 *passwordLength,
    void **passwordData,
    SecKeychainItemRef *itemRef
);
OSStatus SecKeychainFindInternetPassword (
    CFTypeRef keychainOrArray,
    UInt32 serverNameLength,
    const char *serverName,
    UInt32 securityDomainLength,
    const char *securityDomain,
    UInt32 accountNameLength,
    const char *accountName,
    UInt32 pathLength,
    const char *path,
    UInt16 port,
    SecProtocolType protocol,
    SecAuthenticationType authenticationType,
    UInt32 *passwordLength,
    void **passwordData,
    SecKeychainItemRef *itemRef
);
OSStatus SecKeychainItemCopyAttributesAndData (
   SecKeychainItemRef itemRef,
   SecKeychainAttributeInfo *info,
   SecItemClass *itemClass,
   SecKeychainAttributeList **attrList,
   UInt32 *length,
   void **outData
);

// Functions which free data.
OSStatus SecKeychainItemFreeContent (
    SecKeychainAttributeList *attrList,
    void *data
);
OSStatus SecKeychainItemFreeAttributesAndData (
   SecKeychainAttributeList *attrList,
   void *data
);

void errRetVal() {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *outData;
  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData);
  if (st == GenericError)
    SecKeychainItemFreeContent(ptr, outData);
} // expected-warning{{Allocated data is not released: missing a call to 'SecKeychainItemFreeContent'}}

// If null is passed in, the data is not allocated, so no need for the matching free.
void fooDoNotReportNull() {
    unsigned int *ptr = 0;
    OSStatus st = 0;
    UInt32 *length = 0;
    void **outData = 0;
    SecKeychainItemCopyContent(2, ptr, ptr, 0, 0);
    SecKeychainItemCopyContent(2, ptr, ptr, length, outData);
}// no-warning

void doubleAlloc() {
    unsigned int *ptr = 0;
    OSStatus st = 0;
    UInt32 length;
    void *outData;
    st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData);
    st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData); // expected-warning {{Allocated data should be released before another call to the allocator:}}
    if (st == noErr)
      SecKeychainItemFreeContent(ptr, outData);
}

// Do not warn if undefined value is passed to a function.
void fooOnlyFreeUndef() {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *outData;
  SecKeychainItemFreeContent(ptr, outData);
}// no-warning

// Do not warn if the address is a parameter in the enclosing function.
void fooOnlyFreeParam(void *attrList, void* X) {
    SecKeychainItemFreeContent(attrList, X); 
}// no-warning

// If we are returning the value, do not report.
void* returnContent() {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *outData;
  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData);
  return outData;
} // no-warning

// Password was passed in as an argument and does not have to be deleted.
OSStatus getPasswordAndItem(void** password, UInt32* passwordLength) {
  OSStatus err;
  SecKeychainItemRef item;
  err = SecKeychainFindGenericPassword(0, 3, "xx", 3, "xx",
                                       passwordLength, password, &item);
  return err;
} // no-warning

// Make sure we do not report an error if we call free only if password != 0.
// Also, do not report double allocation if first allocation returned an error.
OSStatus testSecKeychainFindGenericPassword(UInt32* passwordLength,
                        CFTypeRef keychainOrArray, SecProtocolType protocol, 
                        SecAuthenticationType authenticationType) {
  OSStatus err;
  SecKeychainItemRef item;
  void *password;
  err = SecKeychainFindGenericPassword(0, 3, "xx", 3, "xx",
                                       passwordLength, &password, &item);
  if( err == GenericError ) {
    err = SecKeychainFindInternetPassword(keychainOrArray, 
                                  16, "server", 16, "domain", 16, "account",
                                  16, "path", 222, protocol, authenticationType,
                                  passwordLength, &(password), 0);
  }

  if (err == noErr && password) {
    SecKeychainItemFreeContent(0, password);
  }
  return err;
}

int apiMismatch(SecKeychainItemRef itemRef, 
         SecKeychainAttributeInfo *info,
         SecItemClass *itemClass) {
  OSStatus st = 0;
  SecKeychainAttributeList *attrList;
  UInt32 length;
  void *outData;
  
  st = SecKeychainItemCopyAttributesAndData(itemRef, info, itemClass, 
                                            &attrList, &length, &outData); 
  if (st == noErr)
    SecKeychainItemFreeContent(attrList, outData); // expected-warning{{Deallocator doesn't match the allocator}}
  return 0;
}

int ErrorCodesFromDifferentAPISDoNotInterfere(SecKeychainItemRef itemRef, 
                                              SecKeychainAttributeInfo *info,
                                              SecItemClass *itemClass) {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *outData;
  OSStatus st2 = 0;
  SecKeychainAttributeList *attrList;
  UInt32 length2;
  void *outData2;

  st2 = SecKeychainItemCopyAttributesAndData(itemRef, info, itemClass, 
                                             &attrList, &length2, &outData2);
  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData);  
  if (st == noErr) {
    SecKeychainItemFreeContent(ptr, outData);
    if (st2 == noErr) {
      SecKeychainItemFreeAttributesAndData(attrList, outData2);
    }
  } 
  return 0; // expected-warning{{Allocated data is not released: missing a call to 'SecKeychainItemFreeAttributesAndData'}}
}

int foo(CFTypeRef keychainOrArray, SecProtocolType protocol, 
        SecAuthenticationType authenticationType, SecKeychainItemRef *itemRef) {
  unsigned int *ptr = 0;
  OSStatus st = 0;

  UInt32 length;
  void *outData[5];

  st = SecKeychainFindInternetPassword(keychainOrArray, 
                                       16, "server", 16, "domain", 16, "account",
                                       16, "path", 222, protocol, authenticationType,
                                       &length, &(outData[3]), itemRef);
  if (length == 5) {
    if (st == noErr)
      SecKeychainItemFreeContent(ptr, outData[3]);
  }
  if (length) { // TODO: We do not report a warning here since the symbol is no longer live, but it's not marked as dead.
    length++;
  }
  return 0;
}

int testErrorCodeAsLHS(CFTypeRef keychainOrArray, SecProtocolType protocol,
        SecAuthenticationType authenticationType, SecKeychainItemRef *itemRef) {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *outData;
  st = SecKeychainFindInternetPassword(keychainOrArray,
                                       16, "server", 16, "domain", 16, "account",
                                       16, "path", 222, protocol, authenticationType,
                                       &length, &outData, itemRef);
  if (noErr == st)
    SecKeychainItemFreeContent(ptr, outData);

  return 0;
}

void free(void *ptr);
void deallocateWithFree() {
    unsigned int *ptr = 0;
    OSStatus st = 0;
    UInt32 length;
    void *outData;
    st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData);
    if (st == noErr)
      free(outData); // expected-warning{{Deallocator doesn't match the allocator: 'SecKeychainItemFreeContent' should be used}}
}

// Typesdefs for CFStringCreateWithBytesNoCopy.
typedef char uint8_t;
typedef signed long CFIndex;
typedef UInt32 CFStringEncoding;
typedef unsigned Boolean;
typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
extern const CFAllocatorRef kCFAllocatorSystemDefault;
extern const CFAllocatorRef kCFAllocatorMalloc;
extern const CFAllocatorRef kCFAllocatorMallocZone;
extern const CFAllocatorRef kCFAllocatorNull;
extern const CFAllocatorRef kCFAllocatorUseContext;
CFStringRef CFStringCreateWithBytesNoCopy(CFAllocatorRef alloc, const uint8_t *bytes, CFIndex numBytes, CFStringEncoding encoding, Boolean externalFormat, CFAllocatorRef contentsDeallocator);

void DellocWithCFStringCreate1(CFAllocatorRef alloc) {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *bytes;
  char * x;
  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &bytes);
  if (st == noErr) {
    CFStringRef userStr = CFStringCreateWithBytesNoCopy(alloc, bytes, length, 5, 0, kCFAllocatorDefault); // expected-warning{{Deallocator doesn't match the allocator:}} 
    CFRelease(userStr);
  }
}

void DellocWithCFStringCreate2(CFAllocatorRef alloc) {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *bytes;
  char * x;
  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &bytes);
  if (st == noErr) {
    CFStringRef userStr = CFStringCreateWithBytesNoCopy(alloc, bytes, length, 5, 0, kCFAllocatorNull); // expected-warning{{Allocated data is not released}}
    CFRelease(userStr); 
  }
}

void DellocWithCFStringCreate3(CFAllocatorRef alloc) {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *bytes;
  char * x;
  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &bytes);
  if (st == noErr) {
    CFStringRef userStr = CFStringCreateWithBytesNoCopy(alloc, bytes, length, 5, 0, kCFAllocatorUseContext);
    CFRelease(userStr);
  }
}

void DellocWithCFStringCreate4(CFAllocatorRef alloc) {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *bytes;
  char * x;
  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &bytes);
  if (st == noErr) {
    CFStringRef userStr = CFStringCreateWithBytesNoCopy(alloc, bytes, length, 5, 0, 0); // expected-warning{{Deallocator doesn't match the allocator:}} 
    CFRelease(userStr);
  }
}

static CFAllocatorRef gKeychainDeallocator = 0;

static CFAllocatorRef GetKeychainDeallocator() {  
  return gKeychainDeallocator;
}

CFStringRef DellocWithCFStringCreate5(CFAllocatorRef alloc) {
  unsigned int *ptr = 0;
  OSStatus st = 0;
  UInt32 length;
  void *bytes;
  char * x;
  st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &bytes);
  if (st == noErr) {
    return CFStringCreateWithBytesNoCopy(alloc, bytes, length, 5, 0, GetKeychainDeallocator()); // no-warning
  }
  return 0;
}

void radar10508828() {
  UInt32 pwdLen = 0;
  void*  pwdBytes = 0;
  OSStatus rc = SecKeychainFindGenericPassword(0, 3, "foo", 3, "bar", &pwdLen, &pwdBytes, 0);
#pragma unused(rc)
  if (pwdBytes)
    SecKeychainItemFreeContent(0, pwdBytes);
}

void radar10508828_20092614() {
  UInt32 pwdLen = 0;
  void*  pwdBytes = 0;
  OSStatus rc = SecKeychainFindGenericPassword(0, 3, "foo", 3, "bar", &pwdLen, &pwdBytes, 0);
  SecKeychainItemFreeContent(0, pwdBytes);
}

//Example from bug 10797.
__inline__ static
const char *__WBASLLevelString(int level) {
  return "foo";
}

static int *bug10798(int *p, int columns, int prevRow) {
  int *row = 0;
  row = p + prevRow * columns;
  prevRow += 2;
  do {
    ++prevRow;
    row+=columns;
  } while(10 >= row[1]);
  return row;
}

// Test inter-procedural behaviour.

void my_FreeParam(void *attrList, void* X) {
    SecKeychainItemFreeContent(attrList, X); 
}

void *my_AllocateReturn(OSStatus *st) {
  unsigned int *ptr = 0;
  UInt32 length;
  void *outData;
  *st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData);
  return outData;
}

OSStatus my_Allocate_Param(void** password, UInt32* passwordLength) {
  OSStatus err;
  SecKeychainItemRef item;
  err = SecKeychainFindGenericPassword(0, 3, "xx", 3, "xx",
                                       passwordLength, password, &item);
  return err;
}

void allocAndFree1() {
    unsigned int *ptr = 0;
    OSStatus st = 0;
    UInt32 length;
    void *outData;
    st = SecKeychainItemCopyContent(2, ptr, ptr, &length, &outData);
    if (st == noErr)
      my_FreeParam(ptr, outData);
}

void consumeChar(char);

void allocNoFree2(int x) {
    OSStatus st = 0;
    void *outData = my_AllocateReturn(&st); 
    if (x) {
      consumeChar(*(char*)outData); // expected-warning{{Allocated data is not released:}}
      return;
    } else {
      consumeChar(*(char*)outData);
    }
    return;
}

void allocAndFree2(void *attrList) {
    OSStatus st = 0;
    void *outData = my_AllocateReturn(&st);
    if (st == noErr)
      my_FreeParam(attrList, outData);
}

void allocNoFree3() {
    UInt32 length = 32;
    void *outData;    
    void *outData2;
    OSStatus st = my_Allocate_Param(&outData, &length); // expected-warning{{Allocated data is not released}}
    st = my_Allocate_Param(&outData2, &length); // expected-warning{{Allocated data is not released}}
}

void allocAndFree3(void *attrList) {
    UInt32 length = 32;
    void *outData;
    OSStatus st = my_Allocate_Param(&outData, &length); 
    if (st == noErr)
      SecKeychainItemFreeContent(attrList, outData);
}

typedef struct AuthorizationValue {
    int length;
    void *data;
} AuthorizationValue;
typedef struct AuthorizationCallback {
    OSStatus (*SetContextVal)(AuthorizationValue *inValue);
} AuthorizationCallback;
static AuthorizationCallback cb;
int radar_19196494() {
  @autoreleasepool {
    AuthorizationValue login_password = {};
    UInt32 passwordLength;
    void *passwordData = 0;
    OSStatus err = SecKeychainFindGenericPassword(0, 0, "", 0, "", (UInt32 *)&login_password.length, (void**)&login_password.data, 0);
    cb.SetContextVal(&login_password);
    if (err == noErr) {
      SecKeychainItemFreeContent(0, login_password.data);
    }
  }
  return 0;
}
