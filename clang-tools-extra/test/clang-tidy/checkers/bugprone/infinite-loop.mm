// RUN: %check_clang_tidy %s bugprone-infinite-loop %t -- -- -fblocks
// RUN: %check_clang_tidy %s bugprone-infinite-loop %t -- -- -fblocks -fobjc-arc

typedef __typeof(sizeof(int)) NSUInteger;

@interface NSArray
+(instancetype)alloc;
-(instancetype)init;
@property(readonly) NSUInteger count;
-(void)addObject: (id)anObject;
@end

@interface I
-(void) instanceMethod;
+(void) classMethod;
+(instancetype)alloc;
-(instancetype)init;
@end

void plainCFunction() {
  int i = 0;
  int j = 0;
  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    j++;
  }
}

@implementation I
- (void)instanceMethod {
  int i = 0;
  int j = 0;
  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    j++;
  }
}

+ (void)classMethod {
  int i = 0;
  int j = 0;
  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    j++;
  }
}

+ (void)recursiveMethod {
  static int i = 0;

  i++;
  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    [I classMethod];
  }

  id x = [[I alloc] init];

  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    [x instanceMethod];
  }
  while (i < 10) {
    // no warning, there is a recursive call that can mutate the static local variable
    [I recursiveMethod];
  }
}
@end

void testArrayCount() {
  NSArray *arr = [[NSArray alloc] init];
  NSUInteger max_count = 10;
  while ([arr count] < max_count) {
    // No warning. Array count is updated on every iteration.
    [arr addObject: [[I alloc] init]];
  }
}

void testArrayCountWithConstant() {
  NSArray *arr = [[NSArray alloc] init];
  while ([arr count] < 10) {
    // No warning. Array count is updated on every iteration.
    [arr addObject: [[I alloc] init]];
  }
}

void testArrayCountProperty() {
  NSArray *arr = [[NSArray alloc] init];
  NSUInteger max_count = 10;
  while (arr.count < max_count) {
    // No warning. Array count is updated on every iteration.
    [arr addObject: [[I alloc] init]];
  }
}

void testArrayCountPropertyWithConstant() {
  NSArray *arr = [[NSArray alloc] init];
  while (arr.count < 10) {
    // No warning. Array count is updated on every iteration.
    [arr addObject: [[I alloc] init]];
  }
}

@interface MyArray {
  @public NSUInteger _count;
}
+(instancetype)alloc;
-(instancetype)init;
-(void)addObject: (id)anObject;

-(void)populate;
@end

@implementation MyArray
-(void)populate {
  NSUInteger max_count = 10;
  while (_count < max_count) {
    // No warning. Array count is updated on every iteration.
    [self addObject: [[I alloc] init]];
  }
}

-(void)populateWithConstant {
  while (_count < 10) {
    // No warning. Array count is updated on every iteration.
    [self addObject: [[I alloc] init]];
  }
}
@end

void testArrayCountIvar() {
  MyArray *arr = [[MyArray alloc] init];
  NSUInteger max_count = 10;
  while (arr->_count < max_count) {
    // No warning. Array count is updated on every iteration.
    [arr addObject: [[I alloc] init]];
  }
}

void testArrayCountIvarWithConstant() {
  MyArray *arr = [[MyArray alloc] init];
  while (arr->_count < 10) {
    // No warning. Array count is updated on every iteration.
    [arr addObject: [[I alloc] init]];
  }
}
