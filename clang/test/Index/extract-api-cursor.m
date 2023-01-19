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

/// This won't show up in docs because we can't serialize it
@interface Derived ()
/// Derived method in category docs, won't show up either.
- (void)derivedMethodInCategory;
@end

// RUN: c-index-test -single-symbol-sgfs local %s | FileCheck %s

// Checking for Foo
// CHECK: "parentContexts":[]
// CHECK-SAME: "relatedSymbols":[]
// CHECK-SAME: "relationships":[]
// CHECK-SAME: "text":"Foo docs"
// CHECK-SAME: "kind":{"displayName":"Structure","identifier":"objective-c.struct"}
// CHECK-SAME: "title":"Foo"

// Checking for bar
// CHECK-NEXT: "parentContexts":[{"kind":"objective-c.struct","name":"Foo","usr":"c:@S@Foo"}]
// CHECK-SAME: "relatedSymbols":[]
// CHECK-SAME: "relationships":[{"kind":"memberOf","source":"c:@S@Foo@FI@bar","target":"c:@S@Foo"
// CHECK-SAME: "text":"Bar docs"
// CHECK-SAME: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-SAME: "title":"bar"

// Checking for Base
// CHECK-NEXT: "parentContexts":[]
// CHECK-SAME: "relatedSymbols":[]
// CHECK-SAME: "relationships":[]
// CHECK-SAME: "text":"Base docs"
// CHECK-SAME: "kind":{"displayName":"Class","identifier":"objective-c.class"}
// CHECK-SAME: "title":"Base"

// Checking for baseProperty
// CHECK-NEXT: "parentContexts":[{"kind":"objective-c.class","name":"Base","usr":"c:objc(cs)Base"}]
// CHECK-SAME: "relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-SAME: "isSystem":false
// CHECK-SAME: "usr":"c:@S@Foo"}]
// CHECK-SAME: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Base(py)baseProperty","target":"c:objc(cs)Base"
// CHECK-SAME: "text":"Base property docs"
// CHECK-SAME: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-SAME: "title":"baseProperty"

// Checking for baseMethodWithArg
// CHECK-NEXT: "parentContexts":[{"kind":"objective-c.class","name":"Base","usr":"c:objc(cs)Base"}]
// CHECK-SAME: "relatedSymbols":[]
// CHECK-SAME: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Base(im)baseMethodWithArg:","target":"c:objc(cs)Base"
// CHECK-SAME: "text":"Base method docs"
// CHECK-SAME: "kind":{"displayName":"Instance Method","identifier":"objective-c.method"}
// CHECK-SAME: "title":"baseMethodWithArg:"

// Checking for Protocol
// CHECK-NEXT: "parentContexts":[]
// CHECK-SAME: "relatedSymbols":[]
// CHECK-SAME: "relationships":[]
// CHECK-SAME: "text":"Protocol docs"
// CHECK-SAME: "kind":{"displayName":"Protocol","identifier":"objective-c.protocol"}
// CHECK-SAME: "title":"Protocol"

// Checking for protocolProperty
// CHECK-NEXT: "parentContexts":[{"kind":"objective-c.protocol","name":"Protocol","usr":"c:objc(pl)Protocol"}]
// CHECK-SAME: "relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-SAME: "isSystem":false
// CHECK-SAME: "usr":"c:@S@Foo"}]
// CHECK-SAME: "relationships":[{"kind":"memberOf","source":"c:objc(pl)Protocol(py)protocolProperty","target":"c:objc(pl)Protocol"
// CHECK-SAME: "text":"Protocol property docs"
// CHECK-SAME: "kind":{"displayName":"Instance Property","identifier":"objective-c.property"}
// CHECK-SAME: "title":"protocolProperty"

// Checking for Derived
// CHECK-NEXT: "parentContexts":[]
// CHECK-SAME: "relatedSymbols":[{"accessLevel":"public","declarationLanguage":"objective-c"
// CHECK-SAME: "isSystem":false
// CHECK-SAME: "usr":"c:objc(cs)Base"}]
// CHECK-SAME: "relationships":[{"kind":"inheritsFrom","source":"c:objc(cs)Derived","target":"c:objc(cs)Base"
// CHECK-SAME: "text":"Derived docs"
// CHECK-SAME: "kind":{"displayName":"Class","identifier":"objective-c.class"}
// CHECK-SAME: "title":"Derived"

// Checking for derivedMethodWithValue
// CHECK-NEXT: "parentContexts":[{"kind":"objective-c.class","name":"Derived","usr":"c:objc(cs)Derived"}]
// CHECK-SAME: "relatedSymbols":[]
// CHECK-SAME: "relationships":[{"kind":"memberOf","source":"c:objc(cs)Derived(im)derivedMethodWithValue:","target":"c:objc(cs)Derived"
// CHECK-SAME: "text":"Derived method docs"
// CHECK-SAME: "kind":{"displayName":"Instance Method","identifier":"objective-c.method"}
// CHECK-SAME: "title":"derivedMethodWithValue:"

// CHECK-NOT: This won't show up in docs because we can't serialize it
// CHECK-NOT: Derived method in category docs, won't show up either.
