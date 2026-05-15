// RUN: %clang_cc1 %s -verify=noarc -fsyntax-only -fblocks
// RUN: %clang_cc1 %s -verify=arc -fsyntax-only -fblocks -fobjc-arc
// RUN: %clang_cc1 %s -verify=arc -fsyntax-only -fblocks -fobjc-arc -x objective-c++
// noarc-no-diagnostics

@class NSString;

void c1(id *a);

void t1(void)
{
  NSString *s __attribute((cleanup(c1)));
}

// Cleanup function with __autoreleasing parameter should be accepted under ARC
// via the writeback conversion (*__strong * -> *__autoreleasing *).
void c2(NSString *__autoreleasing *obj) { (void)obj; }

void t2(void) {
  NSString *x __attribute__((cleanup(c2)));
}

typedef void (^DeferBlock)(void);
void c3(DeferBlock __autoreleasing *block) {
  if (block && *block) (*block)();
}

void t3(void) {
  DeferBlock x __attribute__((cleanup(c3))) = ^{};
}

// Writeback conversion doesn't apply to __unsafe_unretained parameters;
// only __autoreleasing is valid.
void c4(NSString *__unsafe_unretained *obj) { (void)obj; }

void t4(void) {
  NSString *x __attribute__((cleanup(c4))); // arc-error {{'cleanup' function 'c4' parameter has type 'NSString *__unsafe_unretained *' which is incompatible with type 'NSString *__strong *'}}
}
