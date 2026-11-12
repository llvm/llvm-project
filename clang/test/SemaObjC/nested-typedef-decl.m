// RUN: %clang_cc1 -x objective-c -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface Bar {
  struct _A *_hardlinkList;
}
@end
@implementation Bar
typedef struct _A {
  int dev;
  int inode;
} A;

- (void) idx:(int)idx ino:(int)ino dev:(int)dev
{
  _hardlinkList[idx].inode = ino;
  _hardlinkList[idx].dev = dev;
}
@end

