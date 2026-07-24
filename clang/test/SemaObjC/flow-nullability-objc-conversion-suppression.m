// The legacy type-based nullable->nonnull conversion warning
// (-Wnullable-to-nonnull-conversion) must be suppressed only when the flow
// analysis actually covers the enclosing decl. For ObjC methods and blocks
// getCurFunctionDecl() is null, so the suppression has to consult the current
// method/block context — otherwise an opted-in ObjC method/block under an
// unspecified default gets BOTH the legacy warning AND the flow warning.
//
// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -fflow-sensitive-nullability -Wnullable-to-nonnull-conversion -Wno-objc-root-class -Wno-unused-value %s -verify

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

// Unannotated -> NOT opted in under the unspecified default -> the method is
// never flow-analyzed, so the legacy warning must still fire (nobody else warns
// about this conversion). The nullable source is external so the method itself
// stays annotation-free.
- (void)unannotated {
  takesNonnull(getNullable()); // expected-warning {{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
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

// Unannotated block (no nullability in its signature) -> not opted in -> legacy
// warning still fires.
void unannotated_block(void) {
  void (^b)(void) = ^{
    takesNonnull(getNullable()); // expected-warning {{implicit conversion from nullable pointer 'int * _Nullable' to non-nullable pointer type 'int * _Nonnull'}}
  };
  b();
}
