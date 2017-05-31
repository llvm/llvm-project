// RUN: %clang_cc1 -triple i386-apple-darwin8 -analyze -analyzer-checker=core,alpha.core -analyzer-constraints=range -analyzer-store=region -Wno-objc-root-class %s > %t.1 2>&1
// RUN: FileCheck -input-file=%t.1 -check-prefix=CHECK-darwin8 %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -analyze -analyzer-checker=core,alpha.core -analyzer-constraints=range -analyzer-store=region -Wno-objc-root-class %s > %t.2 2>&1
// RUN: FileCheck -input-file=%t.2 -check-prefix=CHECK-darwin9 %s
// RUN: %clang_cc1 -triple thumbv6-apple-ios4.0 -analyze -analyzer-checker=core,alpha.core -analyzer-constraints=range -analyzer-store=region -Wno-objc-root-class %s > %t.3 2>&1
// RUN: FileCheck -input-file=%t.3 -check-prefix=CHECK-darwin9 %s

@interface MyClass {}
- (void *)voidPtrM;
- (int)intM;
- (long long)longlongM;
- (unsigned long long)unsignedLongLongM;
- (double)doubleM;
- (long double)longDoubleM;
- (void)voidM;
@end
@implementation MyClass
- (void *)voidPtrM { return (void *)0; }
- (int)intM { return 0; }
- (long long)longlongM { return 0; }
- (unsigned long long)unsignedLongLongM { return 0; }
- (double)doubleM { return 0.0; }
- (long double)longDoubleM { return 0.0; }
- (void)voidM {}
@end

void createFoo() {
  MyClass *obj = 0;  
  
  void *v = [obj voidPtrM]; // no-warning
  int i = [obj intM]; // no-warning
}

void createFoo2() {
  MyClass *obj = 0;  
  
  long double ld = [obj longDoubleM];
}

void createFoo3() {
  MyClass *obj;
  obj = 0;  
  
  long long ll = [obj longlongM];
}

void createFoo4() {
  MyClass *obj = 0;  
  
  double d = [obj doubleM];
}

void createFoo5() {
  MyClass *obj = (id)@"";  
  
  double d = [obj doubleM]; // no-warning
}

void createFoo6() {
  MyClass *obj;
  obj = 0;  
  
  unsigned long long ull = [obj unsignedLongLongM];
}

void handleNilPruneLoop(MyClass *obj) {
  if (!!obj)
    return;
  
  // Test if [obj intM] evaluates to 0, thus pruning the entire loop.
  for (int i = 0; i < [obj intM]; i++) {
    long long j = [obj longlongM];
  }
  
  long long j = [obj longlongM];
}

int handleVoidInComma() {
  MyClass *obj = 0;
  return [obj voidM], 0;
}

int marker(void) { // control reaches end of non-void function
}

// CHECK-darwin8: warning: The receiver of message 'longDoubleM' is nil and returns a value of type 'long double' that will be garbage
// CHECK-darwin8: warning: The receiver of message 'longlongM' is nil and returns a value of type 'long long' that will be garbage
// CHECK-darwin8: warning: The receiver of message 'doubleM' is nil and returns a value of type 'double' that will be garbage
// CHECK-darwin8: warning: The receiver of message 'unsignedLongLongM' is nil and returns a value of type 'unsigned long long' that will be garbage
// CHECK-darwin8: warning: The receiver of message 'longlongM' is nil and returns a value of type 'long long' that will be garbage

// CHECK-darwin9-NOT: warning: The receiver of message 'longlongM' is nil and returns a value of type 'long long' that will be garbage
// CHECK-darwin9-NOT: warning: The receiver of message 'unsignedLongLongM' is nil and returns a value of type 'unsigned long long' that will be garbage
// CHECK-darwin9-NOT: warning: The receiver of message 'doubleM' is nil and returns a value of type 'double' that will be garbage
// CHECK-darwin9-NOT: warning: The receiver of message 'longlongM' is nil and returns a value of type 'long long' that will be garbage
// CHECK-darwin9-NOT: warning: The receiver of message 'longDoubleM' is nil and returns a value of type 'long double' that will be garbage
// CHECK-darwin9: 1 warning generated

