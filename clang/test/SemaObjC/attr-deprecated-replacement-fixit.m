// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck --implicit-check-not fix-it: %s
// RUN: cp %s %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -DIGNORE_UNSUCCESSFUL_RENAMES -fixit -x objective-c %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -DIGNORE_UNSUCCESSFUL_RENAMES -Werror -x objective-c %t

#if !__has_feature(attribute_deprecated_with_replacement)
#error "Missing __has_feature"
#endif

#if !__has_feature(attribute_availability_with_replacement)
#error "Missing __has_feature"
#endif

#define DEPRECATED(replacement) __attribute__((deprecated("message", replacement)))

@protocol SuccessfulMultiParameterRenames
// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)multi:(int)param1 parameter:(int)param2 replacement:(int)param3 DEPRECATED("multi_new:parameter_new:replace_new_ment:");
- (void)multi_new:(int)param1 parameter_new:(int)param2 replace_new_ment:(int)param3;

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)varArgs:(int)params, ... DEPRECATED("renameVarArgs:");
- (void)renameVarArgs:(int)params, ...;

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)leadingMinus:(int)param DEPRECATED("-leadingMinusRenamed:");
- (void)leadingMinusRenamed:(int)param;

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)leadingPlus:(int)param DEPRECATED("+leadingPlusRenamed:");
- (void)leadingPlusRenamed:(int)param;

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)sourceEmptyName:(int)param1 :(int)param2 DEPRECATED("renameEmptyName:toNonEmpty:");
- (void)renameEmptyName:(int)param1 toNonEmpty:(int)param2;

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)target:(int)param1 willBecomeEmpty:(int)param2 emptyName:(int)param3 DEPRECATED("target::emptyName:");
- (void)target:(int)param1 :(int)param2 emptyName:(int)param3;

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)extra:(int)param1 whiteSpace:(int)param2 DEPRECATED("renameExtra:whiteSpace:");
- (void)renameExtra:(int)param1 whiteSpace:(int)param2;

// Test renaming that was producing valid code earlier is still producing valid
// code. The difference is that now we detect different number of parameters.
//
// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)singleArgumentRegression:(int)param DEPRECATED("renameSingleArgument");
- (void)renameSingleArgument:(int)param;
@end

void successfulRenames(id<SuccessfulMultiParameterRenames> object) {
  [object multi:0 parameter:1 replacement:2]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:16}:"multi_new"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:19-[[@LINE-2]]:28}:"parameter_new"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:31-[[@LINE-3]]:42}:"replace_new_ment"

  [object varArgs:1, 2, 3, 0]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:18}:"renameVarArgs"

  [object leadingMinus:0]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:23}:"leadingMinusRenamed"

  [object leadingPlus:0]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:22}:"leadingPlusRenamed"

  [object sourceEmptyName:0 :1]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:26}:"renameEmptyName"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:29-[[@LINE-2]]:29}:"toNonEmpty"

  [object target:0 willBecomeEmpty:1 emptyName:2]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:17}:"target"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:20-[[@LINE-2]]:35}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:38-[[@LINE-3]]:47}:"emptyName"

  [object extra: 0    whiteSpace:   1]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:16}:"renameExtra"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:23-[[@LINE-2]]:33}:"whiteSpace"

  [object singleArgumentRegression:0]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:35}:"renameSingleArgument"
}

#ifndef IGNORE_UNSUCCESSFUL_RENAMES
@protocol UnsuccessfulMultiParameterRenames
// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)differentNumberOfParameters:(int)param DEPRECATED("rename:hasMoreParameters:");

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)differentNumber:(int)param1 ofParameters:(int)param2 DEPRECATED("renameHasLessParameters:");

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)methodLike:(int)param1 replacement:(int)param2 DEPRECATED("noColon:atTheEnd");

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)freeFormText DEPRECATED("Use something else");

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)freeFormTextReplacementStartsAsMethod DEPRECATED("-Use something different");

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)replacementHasSingleSkipCharacter DEPRECATED("-");

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)replacementHasInvalid:(int)param1 slotName:(int)param2 DEPRECATED("renameWith:1nonIdentifier:");
@end

void unsuccessfulRenames(id<UnsuccessfulMultiParameterRenames> object) {
  [object differentNumberOfParameters:0]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:38}:"rename:hasMoreParameters:"

  [object differentNumber:0 ofParameters:1]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:26}:"renameHasLessParameters:"

  [object methodLike:0 replacement:1]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:21}:"noColon:atTheEnd"

  [object freeFormText]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:23}:"Use something else"

  [object freeFormTextReplacementStartsAsMethod]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:48}:"-Use something different"

  [object replacementHasSingleSkipCharacter]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:44}:"-"

  [object replacementHasInvalid:0 slotName:1]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:32}:"renameWith:1nonIdentifier:"
}
#endif // IGNORE_UNSUCCESSFUL_RENAMES

// Make sure classes are treated the same way as protocols.
__attribute__((objc_root_class))
@interface Interface
// expected-note@+1 {{has been explicitly marked deprecated here}}
+ (void)classMethod:(int)param1 replacement:(int)param2 DEPRECATED("renameClassMethod:replace_new_ment:");
+ (void)renameClassMethod:(int)param1 replace_new_ment:(int)param2;

// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)multi:(int)param1 parameter:(int)param2 replacement:(int)param3 DEPRECATED("multi_new:parameter_new:replace_new_ment:");
- (void)multi_new:(int)param1 parameter_new:(int)param2 replace_new_ment:(int)param3;
@end

@implementation Interface
+ (void)classMethod:(int)param1 replacement:(int)param2 {}
+ (void)renameClassMethod:(int)param1 replace_new_ment:(int)param2 {}

- (void)multi:(int)param1 parameter:(int)param2 replacement:(int)param3 {}
- (void)multi_new:(int)param1 parameter_new:(int)param2 replace_new_ment:(int)param3 {}

- (void)usage {
  [Interface classMethod:0 replacement:1]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:14-[[@LINE-1]]:25}:"renameClassMethod"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:28-[[@LINE-2]]:39}:"replace_new_ment"

  [self multi:0 parameter:1 replacement:2]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:14}:"multi_new"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:26}:"parameter_new"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:29-[[@LINE-3]]:40}:"replace_new_ment"
}
@end

// Make sure availability attribute is handled the same way as deprecation attribute.
@protocol AvailabilityAttributeRenames
// expected-note@+1 {{has been explicitly marked deprecated here}}
- (void)multi:(int)param1 parameter:(int)param2 replacement:(int)param3 __attribute__((availability(macosx,deprecated=9.0,replacement="multi_new:parameter_new:replace_new_ment:")));
- (void)multi_new:(int)param1 parameter_new:(int)param2 replace_new_ment:(int)param3;
@end

void availabilityAttributeRenames(id<AvailabilityAttributeRenames> object) {
  [object multi:0 parameter:1 replacement:2]; // expected-warning {{is deprecated}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:16}:"multi_new"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:19-[[@LINE-2]]:28}:"parameter_new"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:31-[[@LINE-3]]:42}:"replace_new_ment"
}
