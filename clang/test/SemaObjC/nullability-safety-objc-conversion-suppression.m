// The legacy type-based nullable->nonnull conversion warning
// (-Wnullable-to-nonnull-conversion) is suppressed inside any method or block:
// the conversion's own annotated source and destination opt the enclosing decl
// into the flow analysis, which reports it instead. For ObjC methods and blocks
// getCurFunctionDecl() is null, so the suppression has to consult the current
// method/block context, or the method/block would get BOTH warnings.
//
// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -fnullability-safety -Wnullable-to-nonnull-conversion -Wno-objc-root-class -Wno-unused-value %s -verify

extern void takesNonnull(int *_Nonnull p);
extern int *_Nullable getNullable(void);

@interface Gate
@end

@implementation Gate

// Annotated -> opted in -> flow analyzes the method. The flow warning covers
// the conversion; the legacy -Wnullable-to-nonnull-conversion warning must NOT
// also fire (that double-warning is what this gate prevents).
- (void)annotated:(int *_Nullable)q {
  takesNonnull(q); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

// Unannotated signature, but the body uses annotated functions, so the method
// is flow-analyzed and only the flow warning fires.
- (void)unannotated {
  takesNonnull(getNullable()); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
}

@end

// Same gate for blocks: an annotated block opts in, so only the flow warning
// fires; the legacy one is suppressed.
void annotated_block(void) {
  void (^b)(int *_Nullable) = ^(int *_Nullable q) {
    takesNonnull(q); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
  };
  b(0);
}

// Unannotated block: opted in by its body, so again only the flow warning.
void unannotated_block(void) {
  void (^b)(void) = ^{
    takesNonnull(getNullable()); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{add a null check}}
  };
  b();
}
