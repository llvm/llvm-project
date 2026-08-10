// RUN: not %clang_cc1 -fobjc-runtime-has-weak -fobjc-arc -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s --strict-whitespace -check-prefixes=CHECK,ARC
// RUN: not %clang_cc1 -fobjc-runtime-has-weak -fobjc-gc -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s --strict-whitespace -check-prefixes=CHECK,GC

// CHECK:      error: no matching function
// CHECK:      :{[[@LINE+1]]:15-[[@LINE+1]]:28}: note: {{.*}}: 1st argument
void powerful(__strong id &);
void lifetime_gcattr_mismatch() {
  static __weak id weak_id;
  powerful(weak_id);
}

// CHECK:      error: no matching function
// ARC:        :{[[@LINE+2]]:11-[[@LINE+2]]:21}: note: {{.*}}: cannot implicitly convert
// GC:         :{[[@LINE+1]]:11-[[@LINE+1]]:21}: note: {{.*}}: no known conversion
void func(char *uiui);

__attribute__((objc_root_class))
@interface Interface
- (void)something;
@end

@implementation Interface
- (void)something{
    func(self);
}
@end
