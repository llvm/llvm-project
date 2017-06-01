// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -std=c++11 %s

enum NSEventType {
  NSEventTypeFlagsChanged = 12
};

enum NSEventMask {
  NSEventMaskLeftMouseDown = 1
};

static const NSEventType NSFlagsChanged = NSEventTypeFlagsChanged;

@interface NSObject
@end
@interface NSEvent : NSObject {
}
+ (nullable id)
addMonitor:(NSEventMask)mask handler:(NSEvent *_Nullable (^)(NSEvent *))block;
@end

void test(id weakThis) {
  id m_flagsChangedEventMonitor = [NSEvent
      addMonitor:NSFlagsChangedMask //expected-error {{use of undeclared identifier 'NSFlagsChangedMask'}}
         handler:[weakThis](NSEvent *flagsChangedEvent) {
             return flagsChangedEvent;
         }];
}
