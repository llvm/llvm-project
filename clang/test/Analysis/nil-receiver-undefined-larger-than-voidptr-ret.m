// RUN: %clang_analyze_cc1 -triple i386-apple-darwin8 -analyzer-checker=core,alpha.core -analyzer-store=region -Wno-objc-root-class %s > %t.1 2>&1
// RUN: FileCheck -input-file=%t.1 -check-prefix=CHECK-darwin8 %s
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin9 -analyzer-checker=core,alpha.core -analyzer-store=region -Wno-objc-root-class %s > %t.2 2>&1
// RUN: FileCheck -input-file=%t.2 -check-prefix=CHECK-darwin9 %s
// RUN: %clang_analyze_cc1 -triple thumbv6-apple-ios4.0 -analyzer-checker=core,alpha.core -analyzer-store=region -Wno-objc-root-class %s > %t.3 2>&1
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

void createFoo(void) {
  MyClass *obj = 0;  
  
  void *v = [obj voidPtrM]; // no-warning
  int i = [obj intM]; // no-warning
}

void createFoo2(void) {
  MyClass *obj = 0;  
  
  long double ld = [obj longDoubleM];
}

void createFoo3(void) {
  MyClass *obj;
  obj = 0;  
  
  long long ll = [obj longlongM];
}

void createFoo4(void) {
  MyClass *obj = 0;  
  
  double d = [obj doubleM];
}

void createFoo5(void) {
  MyClass *obj = (id)@"";  
  
  double d = [obj doubleM]; // no-warning
}

void createFoo6(void) {
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

int handleVoidInComma(void) {
  MyClass *obj = 0;
  return [obj voidM], 0;
}

int marker(void) { // non-void function does not return a value
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

