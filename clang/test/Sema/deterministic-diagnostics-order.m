// RUN: %clang_cc1 %s -fsyntax-only -Wobjc-property-implementation -Watomic-property-with-user-defined-accessor -Wunused 2> %t.err
// RUN: FileCheck -input-file=%t.err %s

@interface I
@end

@interface I(cat)
@property id prop1;
@property id prop2;
@property id prop3;
@end

@implementation I(cat)
@end

// CHECK: warning: property 'prop1' requires method
// CHECK: warning: property 'prop2' requires method
// CHECK: warning: property 'prop3' requires method

@interface I2
@property int prop1;
@property int prop2;
@property int prop3;
@end

@implementation I2
@synthesize prop1, prop2, prop3;
-(int) prop1 { return 0; }
-(int) prop2 { return 0; }
-(int) prop3 { return 0; }
@end

// CHECK: warning: writable atomic property 'prop1'
// CHECK: warning: writable atomic property 'prop2'
// CHECK: warning: writable atomic property 'prop3'

void test_unused() {
  // Add enough variables to exceed the small storage of Scope::DeclSetTy.
  int v1;
  int v2;
  int v3;
  int v4;
  int v5;
  int v6;
  int v7;
  int v8;
  int v9;
  int v10;
  int v11;
  int v12;
  int v13;
  int v14;
  int v15;
  int v16;
  int v17;
  int v18;
  int v19;
  int v20;
  int v21;
  int v22;
  int v23;
  int v24;
  int v25;
  int v26;
  int v27;
  int v28;
  int v29;
  int v30;
  int v31;
  int v32;
  int v33;
  int v34;
  int v35;
  int v36;
  int v37;
  int v38;
}

// CHECK: warning: unused variable 'v1'
// CHECK: warning: unused variable 'v2'
// CHECK: warning: unused variable 'v3'
// CHECK: warning: unused variable 'v4'
// CHECK: warning: unused variable 'v5'
// CHECK: warning: unused variable 'v6'
// CHECK: warning: unused variable 'v7'
// CHECK: warning: unused variable 'v8'
// CHECK: warning: unused variable 'v9'
// CHECK: warning: unused variable 'v10'
// CHECK: warning: unused variable 'v11'
// CHECK: warning: unused variable 'v12'
// CHECK: warning: unused variable 'v13'
// CHECK: warning: unused variable 'v14'
// CHECK: warning: unused variable 'v15'
// CHECK: warning: unused variable 'v16'
// CHECK: warning: unused variable 'v17'
// CHECK: warning: unused variable 'v18'
// CHECK: warning: unused variable 'v19'
// CHECK: warning: unused variable 'v20'
// CHECK: warning: unused variable 'v21'
// CHECK: warning: unused variable 'v22'
// CHECK: warning: unused variable 'v23'
// CHECK: warning: unused variable 'v24'
// CHECK: warning: unused variable 'v25'
// CHECK: warning: unused variable 'v26'
// CHECK: warning: unused variable 'v27'
// CHECK: warning: unused variable 'v28'
// CHECK: warning: unused variable 'v29'
// CHECK: warning: unused variable 'v30'
// CHECK: warning: unused variable 'v31'
// CHECK: warning: unused variable 'v32'
// CHECK: warning: unused variable 'v33'
// CHECK: warning: unused variable 'v34'
// CHECK: warning: unused variable 'v35'
// CHECK: warning: unused variable 'v36'
// CHECK: warning: unused variable 'v37'
// CHECK: warning: unused variable 'v38'
