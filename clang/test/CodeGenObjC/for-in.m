// RUN: %clang_cc1 %s -verify -o /dev/null
// RUN: %clang_cc1 %s -triple x86_64-apple-darwin -emit-llvm -fsanitize=objc-cast -o - | FileCheck %s

void p(const char*, ...);

@interface NSArray
+(NSArray*) arrayWithObjects: (id) first, ...;
-(unsigned) count;
@end
@interface NSString
-(const char*) cString;
@end

#define S(n) @#n
#define L1(n) S(n+0),S(n+1)
#define L2(n) L1(n+0),L1(n+2)
#define L3(n) L2(n+0),L2(n+4)
#define L4(n) L3(n+0),L3(n+8)
#define L5(n) L4(n+0),L4(n+16)
#define L6(n) L5(n+0),L5(n+32)

// CHECK-LABEL: define{{.*}} void @t0
void t0() {
  NSArray *array = [NSArray arrayWithObjects: L1(0), (void*)0];

  p("array.length: %d\n", [array count]);
  unsigned index = 0;
  for (NSString *i in array) {	// expected-warning {{collection expression type 'NSArray *' may not respond}}

    // CHECK:      [[expectedCls:%.*]] = load %struct._class_t*, {{.*}}, !nosanitize
    // CHECK-NEXT: [[kindOfClassSel:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES{{.*}}, !nosanitize
    // CHECK-NEXT: [[expectedClsI8:%.*]] = bitcast %struct._class_t* [[expectedCls]] to i8*, !nosanitize
    // CHECK-NEXT: [[isCls:%.*]] = call zeroext i1 bitcast {{.*}}@objc_msgSend to i1 (i8*, i8*, {{.*}})(i8* [[theItem:%.*]], i8* [[kindOfClassSel]], i8* [[expectedClsI8]]), !nosanitize
    // CHECK: br i1 [[isCls]]

    // CHECK: ptrtoint i8* [[theItem]] to i64, !nosanitize
    // CHECK-NEXT: call void @__ubsan_handle_invalid_objc_cast
    // CHECK-NEXT: unreachable, !nosanitize

    // CHECK: bitcast i8* [[theItem]]

    p("element %d: %s\n", index++, [i cString]);
  }
}

void t1() {
  NSArray *array = [NSArray arrayWithObjects: L6(0), (void*)0];

  p("array.length: %d\n", [array count]);
  unsigned index = 0;
  for (NSString *i in array) {	// expected-warning {{collection expression type 'NSArray *' may not respond}}
    index++;
    if (index == 10)
      continue;
    p("element %d: %s\n", index, [i cString]);
    if (index == 55)
      break;
  }
}

// rdar://problem/9027663
void t2(NSArray *array) {
  for (NSArray *array in array) { // expected-warning {{collection expression type 'NSArray *' may not respond}}
  }
}
