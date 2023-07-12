// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=struct objc_object*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// FIXME: Should be able to pipe into clang, but code is not yet correct for
// other reasons.

void *sel_registerName(const char *);
void objc_enumerationMutation(id);

@class NSNotification;
@class NSMutableArray;

void foo(NSMutableArray *notificationArray, id X) {
  for (NSNotification *notification in notificationArray)
    [X postNotification:notification];
}

