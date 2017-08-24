// RUN: %clang_cc1 -emit-llvm -triple i686-pc-windows-msvc19.0.24213 -gcodeview -debug-info-kind=limited -std=c++14 %s -o - | FileCheck %s
// PR33997.
struct already_AddRefed {
  ~already_AddRefed();
};
struct RefPtr {
  operator int *();
};
struct ServoCssRulesStrong {
  already_AddRefed Consume();
};
struct GroupRule {
  GroupRule(already_AddRefed);
};
class ConditionRule : GroupRule {
  using GroupRule::GroupRule;
};
class CSSMediaRule : ConditionRule {
  using ConditionRule::ConditionRule;
};
class CSSMozDocumentRule : ConditionRule {
  using ConditionRule::ConditionRule;
};
class ServoDocumentRule : CSSMozDocumentRule {
  ServoDocumentRule(RefPtr);
};
class ServoMediaRule : CSSMediaRule {
  ServoMediaRule(RefPtr);
};
ServoCssRulesStrong Servo_MediaRule_GetRules(int *);
ServoCssRulesStrong Servo_DocumentRule_GetRules(int *);
ServoDocumentRule::ServoDocumentRule(RefPtr aRawRule)
    : CSSMozDocumentRule(Servo_DocumentRule_GetRules(aRawRule).Consume()) {}

ServoMediaRule::ServoMediaRule(RefPtr aRawRule)
    : CSSMediaRule(Servo_MediaRule_GetRules(aRawRule).Consume()) {}

// CHECK: define{{.*}}ServoMediaRule
// CHECK-NOT: {{ ret }}
// CHECK: store %class.ConditionRule* %
// CHECK-SAME: %class.ConditionRule** %
// CHECK-SAME: !dbg ![[INL:[0-9]+]]

// CHECK: ![[INL]] = !DILocation(line: 16, scope: ![[SP:[0-9]+]], inlinedAt:
// CHECK: ![[SP]] = distinct !DISubprogram(name: "GroupRule", {{.*}}isDefinition: true
