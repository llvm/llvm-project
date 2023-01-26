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


// Checking for Foo
// RUN: c-index-test "-single-symbol-sgf-for=c:@S@Foo" %s | FileCheck -check-prefix=CHECK-FOO %s
// CHECK-FOO: "parentContexts":[]
// CHECK-FOO-SAME: "relatedSymbols":[]
// CHECK-FOO-SAME: "relationships":[]
// CHECK-FOO-SAME: "text":"Foo docs"
// CHECK-FOO-SAME: "kind":{"displayName":"Structure","identifier":"objective-c.struct"}
// CHECK-FOO-SAME: "title":"Foo"


// Checking for bar
// RUN: c-index-test "-single-symbol-sgf-for=c:@S@Foo@FI@bar" %s | FileCheck -check-prefix=CHECK-BAR %s
// CHECK-BAR: "parentContexts":[{"kind":"objective-c.struct","name":"Foo","usr":"c:@S@Foo"}]
// CHECK-BAR-SAME: "relatedSymbols":[]
// CHECK-BAR-SAME: "relationships":[{"kind":"memberOf","source":"c:@S@Foo@FI@bar","target":"c:@S@Foo"
// CHECK-BAR-SAME: "text":"Bar docs"
// CHECK-BAR-SAME: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-BAR-SAME: "title":"bar"

// Checking for Base
// RUN: c-index-test "-single-symbol-sgf-for=c:objc(cs)Base" %s | FileCheck -check-prefix=CHECK-BASE %s
// CHECK-BASE: "parentContexts":[]
// CHECK-BASE-SAME: "relatedSymbols":[]
// CHECK-BASE-SAME: "relationships":[]
// CHECK-BASE-SAME: "text":"Base docs"
// CHECK-BASE-SAME: "kind":{"displayName":"Class","identifier":"objective-c.class"}
// CHECK-BASE-SAME: "title":"Base"

// Checking for baseProperty
// RUN: c-index-test "-single-symbol-sgf-for=c:objc(cs)Base(py)baseProperty" %s | FileCheck -check-prefix=CHECK-BASEPROP %s
// CHECK-BASEPROP: "parentContexts":[{"kind":"objective-c.class","name":"Base","usr":"c:objc(cs)Base"}]
// CHECK-BASEPROP-SAME:"relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-BASEPROP-SAME: "isSystem":false
// CHECK-BASEPROP-SAME: "usr":"c:@S@Foo"}]
// CHECK-BASEPROP-SAME: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Base(py)baseProperty","target":"c:objc(cs)Base"
// CHECK-BASEPROP-SAME: "text":"Base property docs"
// CHECK-BASEPROP-SAME: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-BASEPROP-SAME: "title":"baseProperty"

// Checking for baseMethodWithArg
// RUN: c-index-test "-single-symbol-sgf-for=c:objc(cs)Base(im)baseMethodWithArg:" %s | FileCheck -check-prefix=CHECK-BASEMETHOD %s
// CHECK-BASEMETHOD: "parentContexts":[{"kind":"objective-c.class","name":"Base","usr":"c:objc(cs)Base"}]
// CHECK-BASEMETHOD-SAME:"relatedSymbols":[]
// CHECK-BASEMETHOD-SAME: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Base(im)baseMethodWithArg:","target":"c:objc(cs)Base"
// CHECK-BASEMETHOD-SAME: "text":"Base method docs"
// CHECK-BASEMETHOD-SAME: "kind":{"displayName":"Instance Method","identifier":"objective-c.method"}
// CHECK-BASEMETHOD-SAME: "title":"baseMethodWithArg:"

// Checking for Protocol
// RUN: c-index-test "-single-symbol-sgf-for=c:objc(pl)Protocol" %s | FileCheck -check-prefix=CHECK-PROT %s
// CHECK-PROT: "parentContexts":[]
// CHECK-PROT-SAME: "relatedSymbols":[]
// CHECK-PROT-SAME: "relationships":[]
// CHECK-PROT-SAME: "text":"Protocol docs"
// CHECK-PROT-SAME: "kind":{"displayName":"Protocol","identifier":"objective-c.protocol"}
// CHECK-PROT-SAME: "title":"Protocol"

// Checking for protocolProperty
// RUN: c-index-test "-single-symbol-sgf-for=c:objc(pl)Protocol(py)protocolProperty" %s | FileCheck -check-prefix=CHECK-PROTPROP %s
// CHECK-PROTPROP: "parentContexts":[{"kind":"objective-c.protocol","name":"Protocol","usr":"c:objc(pl)Protocol"}]
// CHECK-PROTPROP-SAME:"relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-PROTPROP-SAME: "isSystem":false
// CHECK-PROTPROP-SAME: "usr":"c:@S@Foo"}]
// CHECK-PROTPROP-SAME: "relationships":[{"kind":"memberOf","source":"c:objc(pl)Protocol(py)protocolProperty","target":"c:objc(pl)Protocol"
// CHECK-PROTPROP-SAME: "text":"Protocol property docs"
// CHECK-PROTPROP-SAME: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-PROTPROP-SAME: "title":"protocolProperty"

// Checking for Derived
// RUN: c-index-test "-single-symbol-sgf-for=c:objc(cs)Derived" %s | FileCheck -check-prefix=CHECK-DERIVED %s
// CHECK-DERIVED: "parentContexts":[]
// CHECK-DERIVED-SAME:"relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-DERIVED-SAME: "isSystem":false
// CHECK-DERIVED-SAME: "usr":"c:objc(cs)Base"}]
// CHECK-DERIVED-SAME: "relationships":[{"kind":"inheritsFrom","source":"c:objc(cs)Derived","target":"c:objc(cs)Base"
// CHECK-DERIVED-SAME: "text":"Derived docs"
// CHECK-DERIVED-SAME: "kind":{"displayName":"Class","identifier":"objective-c.class"}
// CHECK-DERIVED-SAME: "title":"Derived"

// Checking for derivedMethodWithValue
// RUN: c-index-test "-single-symbol-sgf-for=c:objc(cs)Derived(im)derivedMethodWithValue:" %s | FileCheck -check-prefix=CHECK-DERIVEDMETHOD %s
// CHECK-DERIVEDMETHOD: "parentContexts":[{"kind":"objective-c.class","name":"Derived","usr":"c:objc(cs)Derived"}]
// CHECK-DERIVEDMETHOD-SAME:"relatedSymbols":[]
// CHECK-DERIVEDMETHOD-SAME: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Derived(im)derivedMethodWithValue:","target":"c:objc(cs)Derived"
// CHECK-DERIVEDMETHOD-SAME: "text":"Derived method docs"
// CHECK-DERIVEDMETHOD-SAME: "kind":{"displayName":"Instance Method","identifier":"objective-c.method"}
// CHECK-DERIVEDMETHOD-SAME: "title":"derivedMethodWithValue:"
