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

// CHECK: [[SELF:%.*]] = alloca [[T6:%.*]]*, align
// CHECK: [[T0:%.*]] = load {{.*}}* [[SELF]], align
// CHECK: [[T1:%.*]] = load {{.*}}* @"\01L_OBJC_SELECTOR_REFERENCES_"
// CHECK: [[C:%.*]] = call %struct.SetSection* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: call i32* @_ZN10SetSection2atEi(%struct.SetSection* [[C]]
