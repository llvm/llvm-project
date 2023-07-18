// RUN: %clang_cc1  -fsyntax-only -triple x86_64-apple-macosx10.11 -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -triple x86_64-apple-macosx10.11 -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://20408445
 
@protocol NSFileManagerDelegate @end

@interface NSFileManager 
@property (assign) id <NSFileManagerDelegate> delegate;
@end

@interface NSFontManager
@property (assign) id delegate __attribute__((availability(macosx,introduced=10.0 ,deprecated=10.11,message="" "NSFontManager doesn't have any delegate method. This property should not be used.")));

@end

id Test20408445(id p) {
        return [p delegate];
}
