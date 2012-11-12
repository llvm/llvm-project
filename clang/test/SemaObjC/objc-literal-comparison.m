// RUN: %clang_cc1 -fsyntax-only -Wno-everything -Wobjc-literal-compare "-Dnil=((id)0)" -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-everything -Wobjc-literal-compare "-Dnil=(id)0" -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-everything -Wobjc-literal-compare "-Dnil=0" -verify %s

// RUN: %clang_cc1 -fsyntax-only -Wno-everything -Wobjc-literal-compare -fobjc-arc "-Dnil=((id)0)" -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-everything -Wobjc-literal-compare -fobjc-arc "-Dnil=(id)0" -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-everything -Wobjc-literal-compare -fobjc-arc "-Dnil=0" -verify %s

// (test the warning flag as well)

typedef signed char BOOL;

@interface BaseObject
+ (instancetype)new;
@end

@interface NSObject : BaseObject
- (BOOL)isEqual:(id)other;
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(BOOL)value;
@end

@interface NSArray : NSObject
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
@end

@interface NSDictionary : NSObject
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
@end

@interface NSString : NSObject
@end

void testComparisonsWithFixits(id obj) {
  if (obj == @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (obj != @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (@"" == obj) return; // expected-warning{{direct comparison of a string literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (@"" == @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}

  if (@[] == obj) return; // expected-warning{{direct comparison of an array literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (@{} == obj) return; // expected-warning{{direct comparison of a dictionary literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (@12 == obj) return; // expected-warning{{direct comparison of a numeric literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (@1.0 == obj) return; // expected-warning{{direct comparison of a numeric literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (@__objc_yes == obj) return; // expected-warning{{direct comparison of a numeric literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (@(1+1) == obj) return; // expected-warning{{direct comparison of a boxed expression has undefined behavior}} expected-note{{use 'isEqual:' instead}}
}


@interface BadEqualReturnString : NSString
- (void)isEqual:(id)other;
@end

@interface BadEqualArgString : NSString
- (BOOL)isEqual:(int)other;
@end


void testComparisonsWithoutFixits() {
  if ([BaseObject new] == @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}}

  if ([BadEqualReturnString new] == @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}}
  if ([BadEqualArgString new] == @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}}

  if (@"" < @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}}
  if (@"" > @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}}
  if (@"" <= @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}}
  if (@"" >= @"") return; // expected-warning{{direct comparison of a string literal has undefined behavior}}
}


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-string-compare"

void testWarningFlags(id obj) {
  if (obj == @"") return; // no-warning
  if (@"" == obj) return; // no-warning

  if (obj == @1) return; // expected-warning{{direct comparison of a numeric literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
  if (@1 == obj) return; // expected-warning{{direct comparison of a numeric literal has undefined behavior}} expected-note{{use 'isEqual:' instead}}
}

#pragma clang diagnostic pop


void testNilComparison() {
  // Don't warn when comparing to nil in a macro.
#define RETURN_IF_NIL(x) if (x == nil || nil == x) return
  RETURN_IF_NIL(@"");
  RETURN_IF_NIL(@1);
  RETURN_IF_NIL(@1.0);
  RETURN_IF_NIL(@[]);
  RETURN_IF_NIL(@{});
  RETURN_IF_NIL(@__objc_yes);
  RETURN_IF_NIL(@(1+1));
}

