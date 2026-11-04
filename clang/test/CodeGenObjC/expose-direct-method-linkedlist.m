// RUN: %clang_cc1 -emit-llvm -fobjc-arc -fblocks -fobjc-runtime-has-weak \
// RUN:   -triple arm64-apple-macos11.0 \
// RUN:   -fobjc-direct-precondition-thunk %s -o - | FileCheck %s

#define nil ((id)0)

__attribute__((objc_root_class))
@interface LinkedList
@property(direct, readonly, nonatomic) int v;
@property(direct, strong, nonatomic) LinkedList* next;
@property(direct, readonly, nonatomic) int instanceId;
@property(strong, nonatomic, direct) void ( ^ printBlock )( void );

+ (instancetype)alloc;
- (instancetype)initWithV:(int)v Next:(id)next __attribute__((objc_direct));
- (instancetype)clone __attribute__((objc_direct));
- (void)print __attribute__((objc_direct));
- (instancetype) reverseWithPrev:(id) prev __attribute__((objc_direct));
- (int) size __attribute__((objc_direct));
- (int) sum __attribute__((objc_direct));
- (double) avg __attribute__((objc_direct));
- (int) sumWith:(LinkedList *) __attribute__((ns_consumed)) other __attribute__((objc_direct));
@end

@implementation LinkedList
static int numInstances=0;

- (instancetype)initWithV:(int)v Next:(id)next{
    _v = v;
    _next = next;
    _instanceId = numInstances;
    LinkedList* __weak weakSelf = self;
    _printBlock = ^void(void) { [weakSelf print]; };
    numInstances++;
    return self;
}
- (instancetype) clone {
  return [[LinkedList alloc] initWithV:self.v Next:[self.next clone]];
}

- (void) print {
  [self.next print];
}

- (LinkedList*) reverseWithPrev:(LinkedList*) prev{
  LinkedList* newHead = (self.next == nil) ? self : [self.next reverseWithPrev:self];
  self.next = prev;
  return newHead;
}

- (int) size {
  return 1 + [self.next size];
}
- (int) sum {
  return self.v + [self.next sum];
}
- (double) avg {
  return (double)[self sum] / (double)[self size];
}
- (int) sumWith:(LinkedList *) __attribute__((ns_consumed)) other {
  return [self sum] + [other sum];
}
@end

int main() {
  // CHECK:  call ptr @"-[LinkedList initWithV:Next:]D_thunk"
  // CHECK:  call ptr @"-[LinkedList initWithV:Next:]D_thunk"
  LinkedList* ll = [[LinkedList alloc] initWithV:8 Next:[[LinkedList alloc] initWithV:7 Next:nil]];

  // CHECK:  call ptr @"-[LinkedList initWithV:Next:]D_thunk"
  // CHECK:  call ptr @"-[LinkedList next]D_thunk"
  // CHECK:  call void @"-[LinkedList setNext:]D_thunk"
  ll.next.next = [[LinkedList alloc] initWithV:6 Next:nil];

  // CHECK:  call void @"-[LinkedList print]D_thunk"
  [ll print];

  // CHECK: call ptr @"-[LinkedList clone]D_thunk"
  LinkedList* cloned = [ll clone];

  // CHECK: call void @"-[LinkedList print]D_thunk"
  [cloned print];

  // CHECK: call ptr @"-[LinkedList printBlock]D_thunk"
  cloned.printBlock();

  // Test ns_consumed parameter with direct method thunk.
  // CHECK: call i32 @"-[LinkedList sumWith:]D_thunk"
  int combined = [ll sumWith:[cloned clone]];

  // CHECK: call ptr @"-[LinkedList reverseWithPrev:]D_thunk"
  ll = [ll reverseWithPrev:nil];

  // CHECK: call void @"-[LinkedList print]D_thunk"
  [ll print];

  return 0;
}
