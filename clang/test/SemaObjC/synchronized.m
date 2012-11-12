// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface PBXTrackableTaskManager @end

@implementation PBXTrackableTaskManager
- (id) init { return 0; }
- (void) unregisterTask:(id) task {
  @synchronized (self) {
  id taskID = [task taskIdentifier];  // expected-warning {{method '-taskIdentifier' not found (return type defaults to 'id')}}
  }
}
@end


struct x { int a; } b;

void test1() {
  @synchronized (b) {  // expected-error {{@synchronized requires an Objective-C object type ('struct x' invalid)}}
  }

  @synchronized (42) {  // expected-error {{@synchronized requires an Objective-C object type ('int' invalid)}}
  }
}
