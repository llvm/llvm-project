// RUN: %clang_cc1 -emit-llvm %s -o /dev/null

@class NSImage;
void bork() {
  NSImage *nsimage;
  [nsimage release];
}
