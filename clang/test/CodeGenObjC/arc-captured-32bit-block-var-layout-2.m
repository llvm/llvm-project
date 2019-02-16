// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple i386-apple-darwin -print-ivar-layout -emit-llvm -o /dev/null %s > %t-32.layout
// RUN: FileCheck --input-file=%t-32.layout %s
// rdar://12184410
// rdar://12752901

@class NSString;
extern void NSLog(NSString *format, ...);
extern int printf(const char *, ...);

int main() {
  NSString *strong;
  unsigned long long eightByte = 0x8001800181818181ull;
  // Test1
  // CHECK: Inline block variable layout: 0x0100, BL_STRONG:1, BL_OPERATOR:0
  void (^block1)() = ^{ printf("%#llx", eightByte); NSLog(@"%@", strong); };

  // Test2
  int i = 1;
  // CHECK: Inline block variable layout: 0x0100, BL_STRONG:1, BL_OPERATOR:0
  void (^block2)() = ^{ printf("%#llx, %d", eightByte, i); NSLog(@"%@", strong); };

  //  Test3
  char ch = 'a';
  // CHECK: Inline block variable layout: 0x0100, BL_STRONG:1, BL_OPERATOR:0
  void (^block3)() = ^{ printf("%c %#llx", ch, eightByte); NSLog(@"%@", strong); };

  // Test4
  unsigned long fourByte = 0x8001ul;
  // CHECK: Inline block variable layout: 0x0100, BL_STRONG:1, BL_OPERATOR:0
  void (^block4)() = ^{ printf("%c %#lx", ch, fourByte); NSLog(@"%@", strong); };

  // Test5
  // CHECK: Inline block variable layout: 0x0100, BL_STRONG:1, BL_OPERATOR:0
  void (^block5)() = ^{ NSLog(@"%@", strong); printf("%c %#llx", ch, eightByte); };

  // Test6
  // CHECK: Block variable layout: BL_OPERATOR:0
  void (^block6)() = ^{ printf("%#llx", eightByte); };
}

/**
struct __block_literal_generic { // 32bytes (64bit) and 20 bytes (32bit).
0  void *__isa;
4  int __flags;
8  int __reserved;
12  void (*__invoke)(void *);
16  struct __block_descriptor *__descriptor;
};
*/
