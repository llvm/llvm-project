// RUN: %clang_cc1 -fsyntax-only -Wno-pragma-clang-attribute -verify %s
// RUN: not %clang_cc1 -ast-dump -ast-dump-filter test %s | FileCheck %s

// Check for contradictions in rules for attribute without a strict subject set:

#pragma clang attribute push (__attribute__((annotate("subRuleContradictions"))), apply_to = any(variable, variable(is_parameter), function(is_member), variable(is_global)))
// expected-error@-1 {{redundant attribute subject matcher sub-rule 'variable(is_parameter)'; 'variable' already matches those declarations}}
// expected-error@-2 {{redundant attribute subject matcher sub-rule 'variable(is_global)'; 'variable' already matches those declarations}}

// Ensure that we've recovered from the error:
int testRecoverSubRuleContradiction = 0;
// CHECK-LABEL: VarDecl{{.*}} testRecoverSubRuleContradiction
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: AnnotateAttr{{.*}} "subRuleContradictions"

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((annotate("negatedSubRuleContradictions2"))), apply_to = any(variable(unless(is_parameter)), variable(is_thread_local), function, variable(is_global)))
// expected-error@-1 {{negated attribute subject matcher sub-rule 'variable(unless(is_parameter))' contradicts sub-rule 'variable(is_global)'}}
// We have just one error, don't error on 'variable(is_global)'

// Ensure that we've recovered from the error:
int testRecoverNegatedContradiction = 0;
// CHECK-LABEL: VarDecl{{.*}} testRecoverNegatedContradiction
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: AnnotateAttr{{.*}} "negatedSubRuleContradictions2"

void testRecoverNegatedContradictionFunc(void);
// CHECK-LABEL: FunctionDecl{{.*}} testRecoverNegatedContradictionFunc
// CHECK-NEXT: AnnotateAttr{{.*}} "negatedSubRuleContradictions2"

#pragma clang attribute pop

// Verify the strict subject set verification.

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function))

int testRecoverStrictnessVar = 0;
// CHECK-LABEL: VarDecl{{.*}} testRecoverStrictnessVar
// CHECK-NEXT: IntegerLiteral
// CHECK-NOT: AbiTagAttr

void testRecoverStrictnessFunc(void);
// CHECK-LABEL: FunctionDecl{{.*}} testRecoverStrictnessFunc
// CHECK-NEXT: AbiTagAttr

struct testRecoverStrictnessStruct { };
// CHECK-LABEL: RecordDecl{{.*}} testRecoverStrictnessStruct
// CHECK-NOT: AbiTagAttr

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function, record(unless(is_union)), variable, enum))
// expected-error@-1 {{attribute 'abi_tag' cannot be applied to 'enum'}}

int testRecoverExtraVar = 0;
// CHECK-LABEL: VarDecl{{.*}} testRecoverExtraVar
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: AbiTagAttr

void testRecoverExtraFunc(void);
// CHECK-LABEL: FunctionDecl{{.*}} testRecoverExtraFunc
// CHECK-NEXT: AbiTagAttr

struct testRecoverExtraStruct { };
// CHECK-LABEL: RecordDecl{{.*}} testRecoverExtraStruct
// CHECK-NEXT: AbiTagAttr

enum testNoEnumAbiTag { CaseCase };
// CHECK-LABEL: EnumDecl{{.*}} testNoEnumAbiTag
// CHECK-NO: AbiTagAttr

#pragma clang attribute pop

// Verify the non-strict subject set verification.

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function))

int testSubset1Var;
// CHECK-LABEL: VarDecl{{.*}} testSubset1Var
// CHECK-NOT: AbiTagAttr

void testSubset1Func(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubset1Func
// CHECK-NEXT: AbiTagAttr

struct testSubset1Struct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubset1Struct
// CHECK-NOT: AbiTagAttr

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = variable)

int testSubset2Var;
// CHECK-LABEL: VarDecl{{.*}} testSubset2Var
// CHECK-NEXT: AbiTagAttr

void testSubset2Func(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubset2Func
// CHECK-NOT: AbiTagAttr

struct testSubset2Struct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubset2Struct
// CHECK-NOT: AbiTagAttr

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union))))

int testSubset3Var;
// CHECK-LABEL: VarDecl{{.*}} testSubset3Var
// CHECK-NOT: AbiTagAttr

void testSubset3Func(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubset3Func
// CHECK-NOT: AbiTagAttr

struct testSubset3Struct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubset3Struct
// CHECK-NEXT: AbiTagAttr

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(function, variable))

int testSubset4Var;
// CHECK-LABEL: VarDecl{{.*}} testSubset4Var
// CHECK-NEXT: AbiTagAttr

void testSubset4Func(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubset4Func
// CHECK-NEXT: AbiTagAttr

struct testSubset4Struct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubset4Struct
// CHECK-NOT: AbiTagAttr

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(variable, record(unless(is_union))))

int testSubset5Var;
// CHECK-LABEL: VarDecl{{.*}} testSubset5Var
// CHECK-NEXT: AbiTagAttr

void testSubset5Func(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubset5Func
// CHECK-NOT: AbiTagAttr

struct testSubset5Struct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubset5Struct
// CHECK-NEXT: AbiTagAttr

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union)), function))

int testSubset6Var;
// CHECK-LABEL: VarDecl{{.*}} testSubset6Var
// CHECK-NOT: AbiTagAttr

void testSubset6Func(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubset6Func
// CHECK-NEXT: AbiTagAttr

struct testSubset6Struct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubset6Struct
// CHECK-NEXT: AbiTagAttr

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union)), function, variable))

int testSubset7Var;
// CHECK-LABEL: VarDecl{{.*}} testSubset7Var
// CHECK-NEXT: AbiTagAttr

void testSubset7Func(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubset7Func
// CHECK-NEXT: AbiTagAttr

struct testSubset7Struct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubset7Struct
// CHECK-NEXT: AbiTagAttr

#pragma clang attribute pop


#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = any(record(unless(is_union)), function, variable, enum, enum_constant))
// expected-error@-1 {{attribute 'abi_tag' cannot be applied to 'enum_constant', and 'enum'}}

int testSubsetRecoverVar;
// CHECK-LABEL: VarDecl{{.*}} testSubsetRecoverVar
// CHECK-NEXT: AbiTagAttr

void testSubsetRecoverFunc(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubsetRecoverFunc
// CHECK-NEXT: AbiTagAttr

struct testSubsetRecoverStruct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubsetRecoverStruct
// CHECK-NEXT: AbiTagAttr

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to = enum)
// expected-error@-1 {{attribute 'abi_tag' cannot be applied to 'enum'}}

int testSubsetNoVar;
// CHECK-LABEL: VarDecl{{.*}} testSubsetNoVar
// CHECK-NOT: AbiTagAttr

void testSubsetNoFunc(void);
// CHECK-LABEL: FunctionDecl{{.*}} testSubsetNoFunc
// CHECK-NOT: AbiTagAttr

struct testSubsetNoStruct { };
// CHECK-LABEL: RecordDecl{{.*}} testSubsetNoStruct
// CHECK-NOT: AbiTagAttr

#pragma clang attribute pop
