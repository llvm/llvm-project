// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

// # Flexible array member.
// ## Instance variables only in interface.
@interface LastIvar {
  char flexible[];
}
@end

@interface NotLastIvar {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

// ## Instance variables in implementation.
@interface LastIvarInImpl
@end
@implementation LastIvarInImpl {
  char flexible[]; // expected-warning {{field 'flexible' with variable sized type 'char[]' is not visible to subclasses and can conflict with their instance variables}}
}
@end

@interface NotLastIvarInImpl
@end
@implementation NotLastIvarInImpl {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
  // expected-warning@-1 {{field 'flexible' with variable sized type 'char[]' is not visible to subclasses and can conflict with their instance variables}}
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

@implementation NotLastIvarInImplWithoutInterface { // expected-warning {{cannot find interface declaration for 'NotLastIvarInImplWithoutInterface'}}
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
  // expected-warning@-1 {{field 'flexible' with variable sized type 'char[]' is not visible to subclasses and can conflict with their instance variables}}
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

@interface LastIvarInClass_OtherIvarInImpl {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
}
@end
@implementation LastIvarInClass_OtherIvarInImpl {
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

// ## Non-instance variables in implementation.
@interface LastIvarInClass_UnrelatedVarInImpl {
  char flexible[];
}
@end
@implementation LastIvarInClass_UnrelatedVarInImpl
int nonIvar;
@end

// ## Instance variables in class extension.
@interface LastIvarInExtension
@end
@interface LastIvarInExtension() {
  char flexible[]; // expected-warning {{field 'flexible' with variable sized type 'char[]' is not visible to subclasses and can conflict with their instance variables}}
}
@end

@interface NotLastIvarInExtension
@end
@interface NotLastIvarInExtension() {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
  // expected-warning@-1 {{field 'flexible' with variable sized type 'char[]' is not visible to subclasses and can conflict with their instance variables}}
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

@interface LastIvarInClass_OtherIvarInExtension {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
}
@end
@interface LastIvarInClass_OtherIvarInExtension() {
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

@interface LastIvarInExtension_OtherIvarInExtension
@end
@interface LastIvarInExtension_OtherIvarInExtension() {
  int last; // expected-note {{next instance variable declaration is here}}
}
@end
@interface LastIvarInExtension_OtherIvarInExtension()
// Extension without ivars to test we see through such extensions.
@end
@interface LastIvarInExtension_OtherIvarInExtension() {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
  // expected-warning@-1 {{field 'flexible' with variable sized type 'char[]' is not visible to subclasses and can conflict with their instance variables}}
}
@end

@interface LastIvarInExtension_OtherIvarInImpl
@end
@interface LastIvarInExtension_OtherIvarInImpl() {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
  // expected-warning@-1 {{field 'flexible' with variable sized type 'char[]' is not visible to subclasses and can conflict with their instance variables}}
}
@end
@implementation LastIvarInExtension_OtherIvarInImpl {
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

// ## Instance variables in named categories.
@interface IvarInNamedCategory
@end
@interface IvarInNamedCategory(Category) {
  char flexible[]; // expected-error {{instance variables may not be placed in categories}}
}
@end

// ## Synthesized instance variable.
@interface LastIvarAndProperty {
  char _flexible[];
}
@property char flexible[]; // expected-error {{property cannot have array or function type 'char[]'}}
@end

// ## Synthesize other instance variables.
@interface LastIvar_ExplicitlyNamedPropertyBackingIvarPreceding {
  int _elementsCount;
  char flexible[];
}
@property int count;
@end
@implementation LastIvar_ExplicitlyNamedPropertyBackingIvarPreceding
@synthesize count = _elementsCount;
@end

@interface LastIvar_ImplicitlyNamedPropertyBackingIvarPreceding {
  int count;
  char flexible[];
}
@property int count;
@end
@implementation LastIvar_ImplicitlyNamedPropertyBackingIvarPreceding
@synthesize count;
@end

@interface NotLastIvar_ExplicitlyNamedPropertyBackingIvarLast {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
}
@property int count;
@end
@implementation NotLastIvar_ExplicitlyNamedPropertyBackingIvarLast
@synthesize count = _elementsCount; // expected-note {{next synthesized instance variable is here}}
@end

@interface NotLastIvar_ImplicitlyNamedPropertyBackingIvarLast {
  char flexible[]; // expected-error {{flexible array member 'flexible' with type 'char[]' is not at the end of class}}
}
@property int count; // expected-note {{next synthesized instance variable is here}}
@end
@implementation NotLastIvar_ImplicitlyNamedPropertyBackingIvarLast
// Test auto-synthesize.
//@synthesize count;
@end


// # Variable sized types.
struct Packet {
  unsigned int size;
  char data[];
};

// ## Instance variables only in interface.
@interface LastStructIvar {
  struct Packet flexible;
}
@end

@interface NotLastStructIvar {
  struct Packet flexible; // expected-error {{field 'flexible' with variable sized type 'struct Packet' is not at the end of class}}
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

// ## Instance variables in implementation.
@interface LastStructIvarInImpl
@end
@implementation LastStructIvarInImpl {
  struct Packet flexible; // expected-warning {{field 'flexible' with variable sized type 'struct Packet' is not visible to subclasses and can conflict with their instance variables}}
}
@end

@interface NotLastStructIvarInImpl
@end
@implementation NotLastStructIvarInImpl {
  struct Packet flexible; // expected-error {{field 'flexible' with variable sized type 'struct Packet' is not at the end of class}}
  // expected-warning@-1 {{field 'flexible' with variable sized type 'struct Packet' is not visible to subclasses and can conflict with their instance variables}}
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

@interface LastStructIvarInClass_OtherIvarInImpl {
  struct Packet flexible; // expected-error {{field 'flexible' with variable sized type 'struct Packet' is not at the end of class}}
}
@end
@implementation LastStructIvarInClass_OtherIvarInImpl {
  int last; // expected-note {{next instance variable declaration is here}}
}
@end

// ## Synthesized instance variable.
@interface LastSynthesizeStructIvar
@property int first;
@property struct Packet flexible; // expected-error {{synthesized property with variable size type 'struct Packet' requires an existing instance variable}}
@end
@implementation LastSynthesizeStructIvar
@end

@interface NotLastSynthesizeStructIvar
@property struct Packet flexible; // expected-error {{synthesized property with variable size type 'struct Packet' requires an existing instance variable}}
@property int last;
@end
@implementation NotLastSynthesizeStructIvar
@end

@interface LastStructIvarWithExistingIvarAndSynthesizedProperty {
  struct Packet _flexible;
}
@property struct Packet flexible;
@end
@implementation LastStructIvarWithExistingIvarAndSynthesizedProperty
@end


// # Subclasses.
@interface FlexibleArrayMemberBase {
  char flexible[]; // expected-note6 {{'flexible' declared here}}
}
@end

@interface NoIvarAdditions : FlexibleArrayMemberBase
@end
@implementation NoIvarAdditions
@end

@interface AddedIvarInInterface : FlexibleArrayMemberBase {
  int last; // expected-warning {{field 'last' can overwrite instance variable 'flexible' with variable sized type 'char[]' in superclass 'FlexibleArrayMemberBase'}}
}
@end

@interface AddedIvarInImplementation : FlexibleArrayMemberBase
@end
@implementation AddedIvarInImplementation {
  int last; // expected-warning {{field 'last' can overwrite instance variable 'flexible' with variable sized type 'char[]' in superclass 'FlexibleArrayMemberBase'}}
}
@end

@interface AddedIvarInExtension : FlexibleArrayMemberBase
@end
@interface AddedIvarInExtension() {
  int last; // expected-warning {{field 'last' can overwrite instance variable 'flexible' with variable sized type 'char[]' in superclass 'FlexibleArrayMemberBase'}}
}
@end

@interface SynthesizedIvar : FlexibleArrayMemberBase
@property int count;
@end
@implementation SynthesizedIvar
@synthesize count; // expected-warning {{field 'count' can overwrite instance variable 'flexible' with variable sized type 'char[]' in superclass 'FlexibleArrayMemberBase'}}
@end

@interface WarnInSubclassOnlyOnce : FlexibleArrayMemberBase {
  int last; // expected-warning {{field 'last' can overwrite instance variable 'flexible' with variable sized type 'char[]' in superclass 'FlexibleArrayMemberBase'}}
}
@end
@interface WarnInSubclassOnlyOnce() {
  int laster;
}
@end
@implementation WarnInSubclassOnlyOnce {
  int lastest;
}
@end

@interface AddedIvarInSubSubClass : NoIvarAdditions {
  int last; // expected-warning {{field 'last' can overwrite instance variable 'flexible' with variable sized type 'char[]' in superclass 'FlexibleArrayMemberBase'}}
}
@end
