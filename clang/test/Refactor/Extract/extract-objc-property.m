@interface HasProperty

@property (strong) HasProperty *item;

- (HasProperty *)implicitProp;

- (void)setImplicitSetter:(HasProperty *)value;

@end

@implementation HasProperty

- (void)test {
// property-name-begin: +2:8
// property-begin: +1:3
  self.item;
// property-end: -1:12
// property-name-end: -2:12
// CHECK: "static HasProperty *extracted(HasProperty *object) {\nreturn object.item;\n}\n\n"

// implicit-name-begin: +2:8
// implicit-begin: +1:3
  self.implicitProp;
// implicit-end: -1:20
// implicit-name-end: -2:20
// CHECK: "static HasProperty *extracted(HasProperty *object) {\nreturn object.implicitProp;\n}\n\n"
}

// RUN: clang-refactor-test perform -action extract -selected=property -selected=implicit %s -fobjc-arc | FileCheck %s
// RUN: clang-refactor-test perform -action extract -selected=property-name -selected=implicit-name %s -fobjc-arc | FileCheck %s

- (void)prohibitSetterExtraction {
// setter-pref-begin: +2:8
// setter-begin: +1:3
  self.item = 0;
// setter-end: -1:12
// setter-pref-end: -2:12
// implicit-setter-pref-begin: +2:8
// implicit-setter-begin: +1:3
  self.implicitSetter = 0;
// implicit-setter-end: -1:22
// implicit-setter-pref-end: -2:22
}
// CHECK-SETTER: Failed to initiate the refactoring action (property setter can't be extracted)!
// RUN: not clang-refactor-test initiate -action extract -selected=setter -selected=setter-pref -selected=implicit-setter -selected=implicit-setter-pref %s -fobjc-arc 2>&1 | FileCheck --check-prefix=CHECK-SETTER %s

@end

@interface HasIntProp
@property (readwrite) int item;
@end

// AVOID-CRASH: "static void extracted(HasIntProp *f) {\nf.item = !f.item;\n}\n\n"
// avoid-extraction-crash-begin: +1:42
void avoidExtractionCrash(HasIntProp *f) {
  f.item = !f.item;
// avoid-extraction-crash-end: -1:5
}

// RUN: clang-refactor-test perform -action extract -selected=avoid-extraction-crash %s -fobjc-arc | FileCheck --check-prefix=AVOID-CRASH %s
