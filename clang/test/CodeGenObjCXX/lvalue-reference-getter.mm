// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// rdar://10153365

static int gint;
struct SetSection { 
      int & at(int __n) { return gint; }
      const int& at(int __n) const { return gint; }
};

static SetSection gSetSection;

@interface SetShow
- (SetSection&)sections;
@end

@implementation SetShow
- (SetSection&) sections {
//  [self sections].at(100);
    self.sections.at(100);
   return gSetSection;
}
@end

// CHECK: [[SELF:%.*]] = alloca ptr, align
// CHECK: [[T0:%.*]] = load {{.*}}, ptr [[SELF]], align
// CHECK: [[T1:%.*]] = load {{.*}}, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK: [[C:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @objc_msgSend
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN10SetSection2atEi(ptr {{[^,]*}} [[C]]
