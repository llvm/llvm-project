// REQUIRES: system-darwin

// RUN: mkdir -p %t

// RUN: %clang -fobjc-export-direct-methods     \
// RUN:   -target arm64-apple-darwin -fobjc-arc \
// RUN:   -O2 -framework Foundation %s -o %t/thunk-linkedlist

// RUN: %t/thunk-linkedlist 8 7 6 | FileCheck %s --check-prefix=CHECK-EXE
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
- (void) printWithFormat:(NSString*)format, ... NS_FORMAT_FUNCTION(1, 2) __attribute__((objc_direct));
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

- (void) printWithFormat:(NSString*)format, ...{
  [self print];
  NSString *description;
  if ([format length] > 0) {
      va_list args;
      va_start(args, format);
      description = [[NSString alloc] initWithFormat:(id)format arguments:args];
      va_end(args);
  }
  printf("%s", description.UTF8String);
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
  // CHECK-EXE: Alloc id: 0, v: 7
  // CHECK-EXE: Alloc id: 1, v: 8
  LinkedList* ll = [[LinkedList alloc] initWithV:atoi(argv[1]) Next:[[LinkedList alloc] initWithV:atoi(argv[2]) Next:nil]];
  // CHECK-EXE: Alloc id: 2, v: 6
  ll.next.next = [[LinkedList alloc] initWithV:atoi(argv[3]) Next:nil];
  // CHECK-EXE: id: 1, v: 8
  // CHECK-EXE: id: 0, v: 7
  // CHECK-EXE: id: 2, v: 6
  [ll print];

  // Because of the recursive clone, the tail is allocated first.
  // CHECK-EXE: Alloc id: 3, v: 6
  // CHECK-EXE: Alloc id: 4, v: 7
  // CHECK-EXE: Alloc id: 5, v: 8
  LinkedList* cloned = [ll clone];

  // CHECK-EXE: id: 5, v: 8
  // CHECK-EXE: id: 4, v: 7
  // CHECK-EXE: id: 3, v: 6
  [cloned print];

  // CHECK-EXE: id: 5, v: 8
  // CHECK-EXE: id: 4, v: 7
  // CHECK-EXE: id: 3, v: 6
  cloned.printBlock();

  // CHECK-EXE: id: 5, v: 8
  // CHECK-EXE: id: 4, v: 7
  // CHECK-EXE: id: 3, v: 6
  // CHECK-EXE: Hello world, I'm cloned, I have 3 elements
  [cloned printWithFormat:@"Hello world, I'm cloned, I have %d elements\n", [cloned size]];

  ll = [ll reverseWithPrev:nil];
  // CHECK-EXE: id: 2, v: 6
  // CHECK-EXE: id: 0, v: 7
  // CHECK-EXE: id: 1, v: 8
  [ll print];

  // All objects should be deallocated.
  // CHECK-EXE: Dealloc
  // CHECK-EXE: Dealloc
  // CHECK-EXE: Dealloc
  // CHECK-EXE: Dealloc
  // CHECK-EXE: Dealloc
  // CHECK-EXE: Dealloc
}
  return 0;
}
