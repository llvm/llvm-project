// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=objc-cast -fsanitize-trap=objc-cast -emit-llvm %s -o - | FileCheck %s

@interface NSFastEnumerationState
@end

#define NSUInteger unsigned int

@interface NSArray
+(NSArray*) arrayWithObjects: (id) first, ...;
- (NSUInteger) countByEnumeratingWithState:(NSFastEnumerationState *) state 
                                   objects:(id[]) buffer 
                                     count:(NSUInteger) len;
-(unsigned) count;
@end
@interface NSString
-(const char*) cString;
@end

void receive_NSString(NSString*);

void t0(void) {
  NSArray *array = [NSArray arrayWithObjects: @"0", @"1", (void*)0];
  for (NSString *i in array) {
    receive_NSString(i);
  }
}

// CHECK-LABEL: @t0
// CHECK: call void @llvm.ubsantrap(i8 9) {{.*}}!dbg [[LOC:![0-9]+]]
// CHECK: [[LOC]] = !DILocation(line: 0, scope: [[MSG:![0-9]+]], {{.+}})
// CHECK: [[MSG]] = distinct !DISubprogram(name: "__clang_trap_msg$Undefined Behavior Sanitizer$Invalid Objective-C cast"
