// REQUIRES: system-darwin

// RUN: mkdir -p %t

// RUN: %clang -fobjc-direct-precondition-thunk    \
// RUN:   -target arm64-apple-darwin -fobjc-arc \
// RUN:   -O2 -framework Foundation %s -o %t/thunk-linkedlist

// RUN: %t/thunk-linkedlist 8 7 6 | FileCheck %s --check-prefix=EXE
#import <Foundation/Foundation.h>

@interface LinkedList: NSObject
@property(direct, readonly, nonatomic) int v;
@property(direct, strong, nonatomic) LinkedList* next;
@property(direct, readonly, nonatomic) int instanceId;
@property(strong, nonatomic, direct) void ( ^ printBlock )( void );
@property(class) int numInstances;

// Prints instantceId before dealloc
- (void) dealloc;
- (instancetype)initWithV:(int)v Next:(id)next __attribute__((objc_direct));
- (instancetype)clone __attribute__((objc_direct));
- (void)print __attribute__((objc_direct));
- (instancetype) reverseWithPrev:(id) prev __attribute__((objc_direct));
- (int) size __attribute__((objc_direct));
- (int) sum __attribute__((objc_direct));
- (double) avg __attribute__((objc_direct));
@end

@implementation LinkedList
@dynamic numInstances;
static int numInstances=0;

- (void) dealloc {
  printf("Dealloc id: %d\n", self.instanceId);
}

- (instancetype)initWithV:(int)v Next:(id)next{
  if (self = [super init]) {
    _v = v;
    _next = next;
    _instanceId = numInstances;
    LinkedList* __weak weakSelf = self;
    _printBlock = ^void(void) { [weakSelf print]; };
    numInstances++;
    printf("Alloc id: %d, v: %d\n", self.instanceId, self.v);
  }
  return self;
}
- (instancetype) clone {
  return [[LinkedList alloc] initWithV:self.v Next:[self.next clone]];
}

- (void) print {
  printf("id: %d, v: %d\n", self.instanceId, self.v);
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
@end

int main(int argc, char** argv) { // argv = ["8", "7", "6"]
@autoreleasepool {
  // CHECK:  call ptr @"-[LinkedList initWithV:Next:]_thunk"
  // CHECK:  call ptr @"-[LinkedList initWithV:Next:]_thunk"
  LinkedList* ll = [[LinkedList alloc] initWithV:atoi(argv[1]) Next:[[LinkedList alloc] initWithV:atoi(argv[2]) Next:nil]];
  // EXE: Alloc id: 0, v: 7
  // EXE: Alloc id: 1, v: 8

  // CHECK:  call ptr @"-[LinkedList initWithV:Next:]_thunk"
  // CHECK:  call ptr @"-[LinkedList next]_thunk"
  // CHECK:  call void @"-[LinkedList setNext:]_thunk"
  ll.next.next = [[LinkedList alloc] initWithV:atoi(argv[3]) Next:nil];
  // EXE: Alloc id: 2, v: 6

  // CHECK:  call void @"-[LinkedList print]_thunk"
  [ll print];
  // EXE: id: 1, v: 8
  // EXE: id: 0, v: 7
  // EXE: id: 2, v: 6

  // CHECK: call ptr @"-[LinkedList clone]_thunk"
  LinkedList* cloned = [ll clone];
  // Because of the recursive clone, the tail is allocated first.
  // EXE: Alloc id: 3, v: 6
  // EXE: Alloc id: 4, v: 7
  // EXE: Alloc id: 5, v: 8

  // CHECK: call void @"-[LinkedList print]_thunk"
  [cloned print];
  // EXE: id: 5, v: 8
  // EXE: id: 4, v: 7
  // EXE: id: 3, v: 6

  // CHECK: call ptr @"-[LinkedList printBlock]_thunk"
  cloned.printBlock();
  // EXE: id: 5, v: 8
  // EXE: id: 4, v: 7
  // EXE: id: 3, v: 6


  // CHECK: call ptr @"-[LinkedList reverseWithPrev:]_thunk"
  ll = [ll reverseWithPrev:nil];
  // EXE: id: 2, v: 6
  // EXE: id: 0, v: 7
  // EXE: id: 1, v: 8

  // CHECK: call void @"-[LinkedList print]_thunk"
  [ll print];

  // All objects should be deallocated.
  // EXE: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NOT: Dealloc
}
  return 0;
}
