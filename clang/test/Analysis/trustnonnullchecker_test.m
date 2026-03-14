// Temporarily disabling the test, it failes the "system is over-constrained" (part of expensive checks)
// assertion in *non* optimized builds.
// REQUIRES: rdar44992170
// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core,nullability,apiModeling,debug.ExprInspection -verify %s

#include "Inputs/system-header-simulator-for-nullability.h"

void clang_analyzer_warnIfReached(void);

NSString* _Nonnull trust_nonnull_framework_annotation(void) {
  NSString* out = [NSString generateString];
  if (out) {}
  return out; // no-warning
}

NSString* _Nonnull trust_instancemsg_annotation(NSString* _Nonnull param) {
  NSString* out = [param stringByAppendingString:@"string"];
  if (out) {}
  return out; // no-warning
}

NSString* _Nonnull distrust_instancemsg_noannotation(NSString* param) {
  if (param) {}
  NSString* out = [param stringByAppendingString:@"string"];
  if (out) {}
  return out; // expected-warning{{}}
}

NSString* _Nonnull trust_analyzer_knowledge(NSString* param) {
  if (!param)
    return @"";
  NSString* out = [param stringByAppendingString:@"string"];
  if (out) {}
  return out; // no-warning
}

NSString* _Nonnull trust_assume_nonnull_macro(void) {
  NSString* out = [NSString generateImplicitlyNonnullString];
  if (out) {}
  return out; // no-warning
}

NSString* _Nonnull distrust_without_annotation(void) {
  NSString* out = [NSString generatePossiblyNullString];
  if (out) {}
  return out; // expected-warning{{}}
}

NSString* _Nonnull nonnull_please_trust_me(void);

NSString* _Nonnull distrust_local_nonnull_annotation(void) {
  NSString* out = nonnull_please_trust_me();
  if (out) {}
  return out; // expected-warning{{}}
}

NSString* _Nonnull trust_c_function(void) {
  NSString* out = getString();
  if (out) {};
  return out; // no-warning
}

NSString* _Nonnull distrust_unannoted_function(void) {
  NSString* out = getPossiblyNullString();
  if (out) {};
  return out; // expected-warning{{}}
}

NSString * _Nonnull distrustProtocol(id<MyProtocol> o) {
  NSString* out = [o getString];
  if (out) {};
  return out; // expected-warning{{}}
}

// If the return value is non-nil, the index is non-nil.
NSString *_Nonnull retImpliesIndex(NSString *s,
                                   NSDictionary *dic) {
  id obj = dic[s];
  if (s) {}
  if (obj)
    return s; // no-warning
  return @"foo";
}

NSString *_Nonnull retImpliesIndexOtherMethod(NSString *s,
                                   NSDictionary *dic) {
  id obj = [dic objectForKey:s];
  if (s) {}
  if (obj)
    return s; // no-warning
  return @"foo";
}

NSString *_Nonnull retImpliesIndexOnRHS(NSString *s,
                                        NSDictionary *dic) {
  id obj = dic[s];
  if (s) {}
  if (nil != obj)
    return s; // no-warning
  return @"foo";
}

NSString *_Nonnull retImpliesIndexReverseCheck(NSString *s,
                                               NSDictionary *dic) {
  id obj = dic[s];
  if (s) {}
  if (!obj)
    return @"foo";
  return s; // no-warning
}

NSString *_Nonnull retImpliesIndexReverseCheckOnRHS(NSString *s,
                                                    NSDictionary *dic) {
  id obj = dic[s];
  if (s) {}
  if (nil == obj)
    return @"foo";
  return s; // no-warning
}

NSString *_Nonnull retImpliesIndexWrongBranch(NSString *s,
                                              NSDictionary *dic) {
  id obj = dic[s];
  if (s) {}
  if (!obj)
    return s; // expected-warning{{}}
  return @"foo";
}

NSString *_Nonnull retImpliesIndexWrongBranchOnRHS(NSString *s,
                                                   NSDictionary *dic) {
  id obj = dic[s];
  if (s) {}
  if (nil == obj)
    return s; // expected-warning{{}}
  return @"foo";
}

// The return value could still be nil for a non-nil index.
NSDictionary *_Nonnull indexDoesNotImplyRet(NSString *s,
                                            NSDictionary *dic) {
  id obj = dic[s];
  if (obj) {}
  if (s)
    return obj; // expected-warning{{}}
  return [[NSDictionary alloc] init];
}

// The return value could still be nil for a non-nil index.
NSDictionary *_Nonnull notIndexImpliesNotRet(NSString *s,
                                             NSDictionary *dic) {
  id obj = dic[s];
  if (!s) {
    if (obj != nil) {
      clang_analyzer_warnIfReached(); // no-warning
    }
  }
  return [[NSDictionary alloc] init];
}

NSString *_Nonnull checkAssumeOnMutableDictionary(NSMutableDictionary *d,
                                                  NSString *k,
                                                  NSString *val) {
  d[k] = val;
  if (k) {}
  return k; // no-warning
}

NSString *_Nonnull checkAssumeOnMutableDictionaryOtherMethod(NSMutableDictionary *d,
                                                             NSString *k,
                                                             NSString *val) {
  [d setObject:val forKey:k];
  if (k) {}
  return k; // no-warning
}

// Check that we don't crash when the added assumption is enough
// to make the state unfeasible.
@class DummyClass;
@interface DictionarySubclass : NSDictionary {
  DummyClass *g;
  DictionarySubclass *d;
}
@end
@implementation DictionarySubclass
- (id) objectForKey:(id)e {
  if (e) {}
  return d;
}
- (void) coder {
  for (id e in g) {
    id f = [self objectForKey:e];
    if (f)
      (void)e;
  }
}
@end
