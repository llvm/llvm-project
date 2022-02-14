// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=osx.SecKeychainAPI -analyzer-store=region -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the default SecKeychainAPI checker.

typedef unsigned int OSStatus;
typedef unsigned int SecKeychainAttributeList;
typedef unsigned int SecKeychainItemRef;
typedef unsigned int SecItemClass;
typedef unsigned int UInt32;
enum {
    noErr                      = 0,
    GenericError               = 1
};
OSStatus SecKeychainItemCopyContent (
                                     SecKeychainItemRef itemRef,
                                     SecItemClass *itemClass,
                                     SecKeychainAttributeList *attrList,
                                     UInt32 *length,
                                     void **outData
                                     );

void DellocWithCFStringCreate4(void) {
    unsigned int *ptr = 0;
    OSStatus st = 0;
    UInt32 length;
    char *bytes;
    char *x;
    st = SecKeychainItemCopyContent(2, ptr, ptr, &length, (void **)&bytes); // expected-note {{Data is allocated here}}
    x = bytes;
    if (st == noErr) // expected-note {{Assuming 'st' is equal to noErr}} // expected-note{{Taking true branch}}
        x = bytes;;
  
    length++; // expected-warning {{Allocated data is not released}} // expected-note{{Allocated data is not released}}
}

