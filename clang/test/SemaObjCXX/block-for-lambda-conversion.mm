// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -std=c++11 %s

enum NSEventType {
  NSEventTypeFlagsChanged = 12
};

enum NSEventMask {
  NSEventMaskLeftMouseDown = 1
};

static const NSEventType NSFlagsChanged = NSEventTypeFlagsChanged; // expected-note {{'NSFlagsChanged' declared here}}

@interface NSObject
@end
@interface NSEvent : NSObject {
}
+ (nullable id)
addMonitor:(NSEventMask)mask handler:(NSEvent *_Nullable (^)(NSEvent *))block; // expected-note {{passing argument to parameter 'mask' here}}
@end

void test(id weakThis) {
  id m_flagsChangedEventMonitor = [NSEvent
      addMonitor:NSFlagsChangedMask //expected-error {{use of undeclared identifier 'NSFlagsChangedMask'}} \
                                      expected-error {{cannot initialize a parameter of type 'NSEventMask' with an lvalue of type 'const NSEventType'}}
         handler:[weakThis](NSEvent *flagsChangedEvent) {
             return flagsChangedEvent;
         }];
}
