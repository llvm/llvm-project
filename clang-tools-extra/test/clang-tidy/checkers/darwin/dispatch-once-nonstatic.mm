// RUN: %check_clang_tidy %s darwin-dispatch-once-nonstatic %t

typedef int dispatch_once_t;
extern void dispatch_once(dispatch_once_t *pred, void(^block)(void));


void bad_dispatch_once(dispatch_once_t once, void(^block)(void)) {}
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: dispatch_once_t variables must have static or global storage duration; function parameters should be pointer references [darwin-dispatch-once-nonstatic]

// file-scope dispatch_once_ts have static storage duration.
dispatch_once_t global_once;
static dispatch_once_t file_static_once;
namespace {
dispatch_once_t anonymous_once;
} // end anonymous namespace

int Correct(void) {
  static int value;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    value = 1;
  });
  return value;
}

int Incorrect(void) {
  static int value;
  dispatch_once_t once;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: dispatch_once_t variables must have static or global storage duration [darwin-dispatch-once-nonstatic]
  // CHECK-FIXES: static dispatch_once_t once;
  dispatch_once(&once, ^{
    value = 1;
  });
  return value;
}

struct OnceStruct {
  static dispatch_once_t staticOnce; // Allowed
  int value;
  dispatch_once_t once;  // Allowed (at this time)
};

@interface MyObject {
  dispatch_once_t _once;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: dispatch_once_t variables must have static or global storage duration and cannot be Objective-C instance variables [darwin-dispatch-once-nonstatic]
  // CHECK-FIXES: dispatch_once_t _once;
}
@end
