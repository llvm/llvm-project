// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-store=region -verify %s

typedef struct CGColorSpace *CGColorSpaceRef;
extern CGColorSpaceRef CGColorSpaceCreateDeviceRGB(void);
extern CGColorSpaceRef CGColorSpaceRetain(CGColorSpaceRef space);
extern void CGColorSpaceRelease(CGColorSpaceRef space);

void f() {
  CGColorSpaceRef X = CGColorSpaceCreateDeviceRGB(); // expected-warning{{leak}}
  CGColorSpaceRetain(X);
}

void fb() {
  CGColorSpaceRef X = CGColorSpaceCreateDeviceRGB();
  CGColorSpaceRetain(X);
  CGColorSpaceRelease(X);
  CGColorSpaceRelease(X);  // no-warning
}
