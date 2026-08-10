// Test is line- and column-sensitive. Run lines are below

/// Foo docs
struct Foo {
    /// Bar docs
    int bar;
};

/// Base docs
@interface Base
/// Base property docs
@property struct Foo baseProperty;

/// Base method docs
- (void)baseMethodWithArg:(int)arg;
@end

/// Protocol docs
@protocol Protocol
/// Protocol property docs
@property struct Foo protocolProperty;
@end

/// Derived docs
@interface Derived: Base
/// Derived method docs
- (void)derivedMethodWithValue:(id<Protocol>)value;
@end

@implementation Derived
- (void)derivedMethodWithValue:(id<Protocol>)value {
    int a = 5;
}
/// Impl only docs
- (void)implOnlyMethod { }
@end

// RUN: c-index-test -single-symbol-sgf-at=%s:4:9 local %s | FileCheck -check-prefix=CHECK-FOO %s
// CHECK-FOO: "parentContexts":[{"kind":"objective-c.struct","name":"Foo","usr":"c:@S@Foo"}]
// CHECK-FOO: "relatedSymbols":[]
// CHECK-FOO: "relationships":[]
// CHECK-FOO: "text":"Foo docs"
// CHECK-FOO: "kind":{"displayName":"Structure","identifier":"objective-c.struct"}
// CHECK-FOO: "title":"Foo"

// RUN: c-index-test -single-symbol-sgf-at=%s:6:9 local %s | FileCheck -check-prefix=CHECK-BAR %s
// CHECK-BAR: "parentContexts":[{"kind":"objective-c.struct","name":"Foo","usr":"c:@S@Foo"},{"kind":"objective-c.property","name":"bar","usr":"c:@S@Foo@FI@bar"}]
// CHECK-BAR: "relatedSymbols":[]
// CHECK-BAR: "relationships":[{"kind":"memberOf","source":"c:@S@Foo@FI@bar","target":"c:@S@Foo"
// CHECK-BAR: "text":"Bar docs"
// CHECK-BAR: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-BAR: "title":"bar"

// RUN: c-index-test -single-symbol-sgf-at=%s:10:11 local %s | FileCheck -check-prefix=CHECK-BASE %s
// CHECK-BASE: "parentContexts":[{"kind":"objective-c.class","name":"Base","usr":"c:objc(cs)Base"}]
// CHECK-BASE: "relatedSymbols":[]
// CHECK-BASE: "relationships":[]
// CHECK-BASE: "text":"Base docs"
// CHECK-BASE: "kind":{"displayName":"Class","identifier":"objective-c.class"}
// CHECK-BASE: "title":"Base"

// RUN: c-index-test -single-symbol-sgf-at=%s:12:25 local %s | FileCheck -check-prefix=CHECK-BASE-PROP %s
// CHECK-BASE-PROP: "parentContexts":[{"kind":"objective-c.class","name":"Base","usr":"c:objc(cs)Base"},{"kind":"objective-c.property","name":"baseProperty","usr":"c:objc(cs)Base(py)baseProperty"}]
// CHECK-BASE-PROP: "relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-BASE-PROP: "isSystem":false
// CHECK-BASE-PROP: "usr":"c:@S@Foo"}]
// CHECK-BASE-PROP: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Base(py)baseProperty","target":"c:objc(cs)Base"
// CHECK-BASE-PROP: "text":"Base property docs"
// CHECK-BASE-PROP: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-BASE-PROP: "title":"baseProperty"

// RUN: c-index-test -single-symbol-sgf-at=%s:15:9 local %s | FileCheck -check-prefix=CHECK-BASE-METHOD %s
// CHECK-BASE-METHOD: "parentContexts":[{"kind":"objective-c.class","name":"Base","usr":"c:objc(cs)Base"},{"kind":"objective-c.method","name":"baseMethodWithArg:","usr":"c:objc(cs)Base(im)baseMethodWithArg:"}]
// CHECK-BASE-METHOD: "relatedSymbols":[]
// CHECK-BASE-METHOD: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Base(im)baseMethodWithArg:","target":"c:objc(cs)Base"
// CHECK-BASE-METHOD: "text":"Base method docs"
// CHECK-BASE-METHOD: "kind":{"displayName":"Instance Method","identifier":"objective-c.method"}
// CHECK-BASE-METHOD: "title":"baseMethodWithArg:"

// RUN: c-index-test -single-symbol-sgf-at=%s:19:11 local %s | FileCheck -check-prefix=CHECK-PROTOCOL %s
// CHECK-PROTOCOL: "parentContexts":[{"kind":"objective-c.protocol","name":"Protocol","usr":"c:objc(pl)Protocol"}]
// CHECK-PROTOCOL: "relatedSymbols":[]
// CHECK-PROTOCOL: "relationships":[]
// CHECK-PROTOCOL: "text":"Protocol docs"
// CHECK-PROTOCOL: "kind":{"displayName":"Protocol","identifier":"objective-c.protocol"}
// CHECK-PROTOCOL: "title":"Protocol"

// RUN: c-index-test -single-symbol-sgf-at=%s:21:27 local %s | FileCheck -check-prefix=CHECK-PROTOCOL-PROP %s
// CHECK-PROTOCOL-PROP: "parentContexts":[{"kind":"objective-c.protocol","name":"Protocol","usr":"c:objc(pl)Protocol"},{"kind":"objective-c.property","name":"protocolProperty","usr":"c:objc(pl)Protocol(py)protocolProperty"}]
// CHECK-PROTOCOL-PROP: "relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-PROTOCOL-PROP: "isSystem":false
// CHECK-PROTOCOL-PROP: "usr":"c:@S@Foo"}]
// CHECK-PROTOCOL-PROP: "relationships":[{"kind":"memberOf","source":"c:objc(pl)Protocol(py)protocolProperty","target":"c:objc(pl)Protocol"
// CHECK-PROTOCOL-PROP: "text":"Protocol property docs"
// CHECK-PROTOCOL-PROP: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-PROTOCOL-PROP: "title":"protocolProperty"

// RUN: c-index-test -single-symbol-sgf-at=%s:25:15 local %s | FileCheck -check-prefix=CHECK-DERIVED %s
// CHECK-DERIVED: "parentContexts":[{"kind":"objective-c.class","name":"Derived","usr":"c:objc(cs)Derived"}]
// CHECK-DERIVED: "relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-DERIVED: "isSystem":false
// CHECK-DERIVED: "usr":"c:objc(cs)Base"}]
// CHECK-DERIVED: "relationships":[{"kind":"inheritsFrom","source":"c:objc(cs)Derived","target":"c:objc(cs)Base"
// CHECK-DERIVED: "text":"Derived docs"
// CHECK-DERIVED: "kind":{"displayName":"Class","identifier":"objective-c.class"}
// CHECK-DERIVED: "title":"Derived"

// RUN: c-index-test -single-symbol-sgf-at=%s:27:11 local %s | FileCheck -check-prefix=CHECK-DERIVED-METHOD %s
// CHECK-DERIVED-METHOD: "parentContexts":[{"kind":"objective-c.class","name":"Derived","usr":"c:objc(cs)Derived"},{"kind":"objective-c.method","name":"derivedMethodWithValue:","usr":"c:objc(cs)Derived(im)derivedMethodWithValue:"}]
// CHECK-DERIVED-METHOD: "relatedSymbols":[]
// CHECK-DERIVED-METHOD: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Derived(im)derivedMethodWithValue:","target":"c:objc(cs)Derived"
// CHECK-DERIVED-METHOD: "text":"Derived method docs"
// CHECK-DERIVED-METHOD: "kind":{"displayName":"Instance Method","identifier":"objective-c.method"}
// CHECK-DERIVED-METHOD: "title":"derivedMethodWithValue:"

// RUN: c-index-test -single-symbol-sgf-at=%s:31:11 local %s | FileCheck -check-prefix=CHECK-DERIVED-METHOD-IMPL %s
// CHECK-DERIVED-METHOD-IMPL: "parentContexts":[{"kind":"objective-c.class","name":"Derived","usr":"c:objc(cs)Derived"},{"kind":"objective-c.method","name":"derivedMethodWithValue:","usr":"c:objc(cs)Derived(im)derivedMethodWithValue:"}]
// CHECK-DERIVED-METHOD-IMPL: "relatedSymbols":[]
// CHECK-DERIVED-METHOD-IMPL: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Derived(im)derivedMethodWithValue:","target":"c:objc(cs)Derived"
// CHECK-DERIVED-METHOD-IMPL: "text":"Derived method docs"
// CHECK-DERIVED-METHOD-IMPL: "kind":{"displayName":"Instance Method","identifier":"objective-c.method"}
// CHECK-DERIVED-METHOD-IMPL: "title":"derivedMethodWithValue:"

// RUN: c-index-test -single-symbol-sgf-at=%s:35:11 local %s | FileCheck -check-prefix=CHECK-IMPL-ONLY %s
// CHECK-IMPL-ONLY: "relatedSymbols":[]
// CHECK-IMPL-ONLY: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Derived(im)implOnlyMethod","target":"c:objc(cs)Derived"
// CHECK-IMPL-ONLY: "text":"Impl only docs"
// CHECK-IMPL-ONLY: "kind":{"displayName":"Instance Method","identifier":"objective-c.method"}
// CHECK-IMPL-ONLY: "title":"implOnlyMethod"
