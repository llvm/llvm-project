// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface PBXTrackableTaskManager @end
@implementation PBXTrackableTaskManager @end

struct x {
  operator PBXTrackableTaskManager *() const { return 0; }
} a;

struct y {
  operator int *() const { return 0; }
} b;

void test1() {
  @synchronized (a) {
  }

  @synchronized (b) {  // expected-error {{@synchronized requires an Objective-C object type ('struct y' invalid)}}
  }
}
