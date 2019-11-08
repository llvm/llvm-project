// RUN: %clang_cc1 -fsyntax-only -verify -Wselector-type-mismatch %s

@protocol Proto
- (void)protoMethod;      // expected-note {{previous declaration is here}}
+ (void)classProtoMethod; // expected-note {{previous declaration is here}}
@end

@protocol ProtoDirectFail
- (void)protoMethod __attribute__((objc_direct));      // expected-error {{'objc_direct' attribute cannot be applied to methods declared in an Objective-C protocol}}
+ (void)classProtoMethod __attribute__((objc_direct)); // expected-error {{'objc_direct' attribute cannot be applied to methods declared in an Objective-C protocol}}
@end

__attribute__((objc_root_class))
@interface Root
- (void)rootRegular;                                  // expected-note {{previous declaration is here}}
+ (void)classRootRegular;                             // expected-note {{previous declaration is here}}
- (void)rootDirect __attribute__((objc_direct));      // expected-note {{previous declaration is here}};
+ (void)classRootDirect __attribute__((objc_direct)); // expected-note {{previous declaration is here}};
- (void)otherRootDirect __attribute__((objc_direct)); // expected-note {{direct method 'otherRootDirect' declared here}}
+ (void)otherClassRootDirect __attribute__((objc_direct)); // expected-note {{direct method 'otherClassRootDirect' declared here}}
- (void)notDirectInIface;                             // expected-note {{previous declaration is here}}
+ (void)classNotDirectInIface;                        // expected-note {{previous declaration is here}}
@end

__attribute__((objc_direct_members))
@interface Root ()
- (void)rootExtensionDirect;      // expected-note {{previous declaration is here}}
+ (void)classRootExtensionDirect; // expected-note {{previous declaration is here}}
@end

__attribute__((objc_direct_members))
@interface Root(Direct)
- (void)rootCategoryDirect;      // expected-note {{previous declaration is here}}
+ (void)classRootCategoryDirect; // expected-note {{previous declaration is here}}
@end

@interface Root ()
- (void)rootExtensionRegular;                                   // expected-note {{previous declaration is here}}
+ (void)classRootExtensionRegular;                              // expected-note {{previous declaration is here}}
- (void)rootExtensionDirect2 __attribute__((objc_direct));      // expected-note {{previous declaration is here}}
+ (void)classRootExtensionDirect2 __attribute__((objc_direct)); // expected-note {{previous declaration is here}}
@end

@interface Root (Direct2)
- (void)rootCategoryRegular;                                   // expected-note {{previous declaration is here}}
+ (void)classRootCategoryRegular;                              // expected-note {{previous declaration is here}}
- (void)rootCategoryDirect2 __attribute__((objc_direct));      // expected-note {{previous declaration is here}}
+ (void)classRootCategoryDirect2 __attribute__((objc_direct)); // expected-note {{previous declaration is here}}
@end

__attribute__((objc_root_class, objc_direct_members)) // expected-error {{'objc_direct_members' attribute only applies to Objective-C implementation declarations and Objective-C containers}}
@interface SubDirectFail : Root
- (instancetype)init;
@end

@interface Sub : Root <Proto>
/* invalid overrides with directs */
- (void)rootRegular __attribute__((objc_direct));               // expected-error {{methods that override superclass methods cannot be direct}}
+ (void)classRootRegular __attribute__((objc_direct));          // expected-error {{methods that override superclass methods cannot be direct}}
- (void)protoMethod __attribute__((objc_direct));               // expected-error {{methods that implement protocol requirements cannot be direct}}
+ (void)classProtoMethod __attribute__((objc_direct));          // expected-error {{methods that implement protocol requirements cannot be direct}}
- (void)rootExtensionRegular __attribute__((objc_direct));      // expected-error {{methods that override superclass methods cannot be direct}}
+ (void)classRootExtensionRegular __attribute__((objc_direct)); // expected-error {{methods that override superclass methods cannot be direct}}
- (void)rootCategoryRegular __attribute__((objc_direct));       // expected-error {{methods that override superclass methods cannot be direct}}
+ (void)classRootCategoryRegular __attribute__((objc_direct));  // expected-error {{methods that override superclass methods cannot be direct}}

/* invalid overrides of directs */
- (void)rootDirect;                // expected-error {{cannot override a method that is declared direct by a superclass}}
+ (void)classRootDirect;           // expected-error {{cannot override a method that is declared direct by a superclass}}
- (void)rootExtensionDirect;       // expected-error {{cannot override a method that is declared direct by a superclass}}
+ (void)classRootExtensionDirect;  // expected-error {{cannot override a method that is declared direct by a superclass}}
- (void)rootExtensionDirect2;      // expected-error {{cannot override a method that is declared direct by a superclass}}
+ (void)classRootExtensionDirect2; // expected-error {{cannot override a method that is declared direct by a superclass}}
- (void)rootCategoryDirect;        // expected-error {{cannot override a method that is declared direct by a superclass}}
+ (void)classRootCategoryDirect;   // expected-error {{cannot override a method that is declared direct by a superclass}}
- (void)rootCategoryDirect2;       // expected-error {{cannot override a method that is declared direct by a superclass}}
+ (void)classRootCategoryDirect2;  // expected-error {{cannot override a method that is declared direct by a superclass}}
@end

__attribute__((objc_direct_members))
@implementation Root
- (void)rootRegular {
}
+ (void)classRootRegular {
}
- (void)rootDirect {
}
+ (void)classRootDirect {
}
- (void)otherRootDirect {
}
+ (void)otherClassRootDirect {
}
- (void)rootExtensionDirect {
}
+ (void)classRootExtensionDirect {
}
- (void)rootExtensionRegular {
}
+ (void)classRootExtensionRegular {
}
- (void)rootExtensionDirect2 {
}
+ (void)classRootExtensionDirect2 {
}
- (void)notDirectInIface __attribute__((objc_direct)) // expected-error {{direct method implementation was previously declared not direct}}
{
}
+ (void)classNotDirectInIface __attribute__((objc_direct)) // expected-error {{direct method implementation was previously declared not direct}}
{
}
- (void)direct1 { // expected-note {{direct method 'direct1' declared here}}
}
- (void)direct2 { // expected-note {{direct method 'direct2' declared here}}
}
@end

@interface Foo : Root
- (id)directMismatch1; // expected-note {{using}}
- (id)directMismatch2; // expected-note {{method 'directMismatch2' declared here}}
@end

@interface Bar : Root
- (void)directMismatch1 __attribute__((objc_direct)); // expected-note {{also found}}
- (void)directMismatch2 __attribute__((objc_direct)); // expected-note {{method 'directMismatch2' declared here}}
@end

@interface ValidSub : Root
@end

@implementation ValidSub
- (void)someValidSubMethod {
  [super otherRootDirect]; // expected-error {{messaging super with a direct method}}
}
@end

extern void callMethod(id obj, Class cls);
extern void useSel(SEL sel);

void callMethod(id obj, Class cls) {
  [Root otherClassRootDirect];
  [cls otherClassRootDirect]; // expected-error {{messaging a Class with a method that is possibly direct}}
  [obj direct1];              // expected-error {{messaging unqualified id with a method that is possibly direct}}
  [(Root *)obj direct1];
  [obj directMismatch1];              // expected-warning {{multiple methods named 'directMismatch1' found}}
  useSel(@selector(direct2));         // expected-error {{@selector expression formed with direct selector 'direct2'}}
  useSel(@selector(directMismatch2)); // expected-warning {{several methods with selector 'directMismatch2' of mismatched types are found for the @selector expression}}
}
