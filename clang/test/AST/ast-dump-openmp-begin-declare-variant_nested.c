// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 1;
}

#pragma omp begin declare variant match(user = {condition(1)}, device = {kind(cpu)}, implementation = {vendor(amd)})
#pragma omp begin declare variant match(device = {kind(cpu)}, implementation = {vendor(amd, pgi), extension(match_any)})
#pragma omp begin declare variant match(device = {kind(any)}, implementation = {dynamic_allocators})
int also_after(void) {
  return 0;
}
int also_before(void) {
  return 0;
}
#pragma omp end declare variant
#pragma omp end declare variant
#pragma omp end declare variant

int also_after(void) {
  return 2;
}

int test(void) {
  // Should return 0.
  return also_after() + also_before();
}

#pragma omp begin declare variant match(device = {isa("sse")})
#pragma omp declare variant(test) match(device = {isa(sse)})
int equivalent_isa_trait(void);
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {isa("sse")})
#pragma omp declare variant(test) match(device = {isa("sse2")})
int non_equivalent_isa_trait(void);
#pragma omp end declare variant

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(1)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:15:1> 'int ({{.*}})' {{.*}}Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:12:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(1)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:14:1> line:12:1 also_after[device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(...)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:14:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:13:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_6]] <line:15:1, line:17:1> line:15:1 also_before[device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(...)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <col:23, line:17:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:16:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] prev [[ADDR_7]] <line:22:1, line:24:1> line:22:5 used also_after 'int ({{.*}})'
// CHECK-NEXT: | |-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:22, line:24:1>
// CHECK-NEXT: | | `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:23:3, col:10>
// CHECK-NEXT: | |   `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(1)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:12:1> 'int ({{.*}})' {{.*}}Function [[ADDR_10]] 'also_after[device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:26:1, line:29:1> line:26:5 referenced test 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_23:0x[a-z0-9]*]] <col:16, line:29:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_24:0x[a-z0-9]*]] <line:28:3, col:37>
// CHECK-NEXT: |     `-BinaryOperator [[ADDR_25:0x[a-z0-9]*]] <col:10, col:37> 'int' '+'
// CHECK-NEXT: |       |-PseudoObjectExpr [[ADDR_26:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT: |       | |-CallExpr [[ADDR_27:0x[a-z0-9]*]] <col:10, col:21> 'int'
// CHECK-NEXT: |       | | `-ImplicitCastExpr [[ADDR_28:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |       | |   `-DeclRefExpr [[ADDR_29:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CHECK-NEXT: |       | `-CallExpr [[ADDR_30:0x[a-z0-9]*]] <line:12:1, line:28:21> 'int'
// CHECK-NEXT: |       |   `-ImplicitCastExpr [[ADDR_31:0x[a-z0-9]*]] <line:12:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |       |     `-DeclRefExpr [[ADDR_9]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_10]] 'also_after[device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT: |       `-PseudoObjectExpr [[ADDR_32:0x[a-z0-9]*]] <line:28:25, col:37> 'int'
// CHECK-NEXT: |         |-CallExpr [[ADDR_33:0x[a-z0-9]*]] <col:25, col:37> 'int'
// CHECK-NEXT: |         | `-ImplicitCastExpr [[ADDR_34:0x[a-z0-9]*]] <col:25> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |         |   `-DeclRefExpr [[ADDR_35:0x[a-z0-9]*]] <col:25> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CHECK-NEXT: |         `-CallExpr [[ADDR_36:0x[a-z0-9]*]] <line:15:1, line:28:37> 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr [[ADDR_37:0x[a-z0-9]*]] <line:15:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |             `-DeclRefExpr [[ADDR_5]] <col:1> 'int ({{.*}})' {{.*}}Function [[ADDR_6]] 'also_before[device={kind(any, cpu)}, implementation={dynamic_allocators, vendor(amd, pgi), extension(match_any)}, user={condition(...)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_38:0x[a-z0-9]*]] <line:33:1, col:30> col:5 equivalent_isa_trait 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_39:0x[a-z0-9]*]] <line:32:1, col:61> Implicit device={isa(sse)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_40:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_22]] 'test' 'int ({{.*}})' non_odr_use_unevaluated
// CHECK-NEXT: `-FunctionDecl [[ADDR_41:0x[a-z0-9]*]] <line:38:1, col:34> col:5 non_equivalent_isa_trait 'int ({{.*}})'
// CHECK-NEXT:   `-OMPDeclareVariantAttr [[ADDR_42:0x[a-z0-9]*]] <line:37:1, col:64> Implicit device={isa(sse2, sse)}
// CHECK-NEXT:     `-DeclRefExpr [[ADDR_43:0x[a-z0-9]*]] <col:29> 'int ({{.*}})' {{.*}}Function [[ADDR_22]] 'test' 'int ({{.*}})' non_odr_use_unevaluated
