// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class -x objective-c++ %s

// Test decls inside Objective-C entities are considered to be duplicates of same-name decls outside of these entities.

@protocol SomeProtocol
struct InProtocol {}; // expected-note {{previous definition is here}}
- (union MethodReturnType { int x; float y; })returningMethod; // expected-note {{previous definition is here}}
#ifdef __cplusplus
// expected-error@-2 {{'MethodReturnType' cannot be defined in a parameter type}}
#endif
@end

@interface Container {
  struct InInterfaceCurliesWithField {} field; // expected-note {{previous definition is here}}
  union InInterfaceCurlies { int x; float y; }; // expected-note {{previous definition is here}}
}
enum InInterface { kX = 0, }; // expected-note {{previous definition is here}}
#ifdef __cplusplus
enum class InInterfaceScoped { kXScoped = 0, }; // expected-note {{previous definition is here}}
#endif
@end

@interface Container(Category)
union InCategory { int x; float y; }; // expected-note {{previous definition is here}}
@end

@interface Container() {
  enum InExtensionCurliesWithField: int { kY = 1, } extensionField; // expected-note {{previous definition is here}}
  struct InExtensionCurlies {}; // expected-note {{previous definition is here}}
}
union InExtension { int x; float y; }; // expected-note {{previous definition is here}}
@end

@implementation Container {
  union InImplementationCurliesWithField { int x; float y; } implField; // expected-note {{previous definition is here}}
  enum InImplementationCurlies { kZ = 2, }; // expected-note {{previous definition is here}}
}
struct InImplementation {}; // expected-note {{previous definition is here}}
@end

@implementation Container(Category)
enum InCategoryImplementation { kW = 3, }; // expected-note {{previous definition is here}}
@end


struct InProtocol { int a; }; // expected-error {{redefinition of 'InProtocol'}}
union MethodReturnType { int a; long b; }; // expected-error {{redefinition of 'MethodReturnType'}}

struct InInterfaceCurliesWithField { int a; }; // expected-error {{redefinition of 'InInterfaceCurliesWithField'}}
union InInterfaceCurlies { int a; long b; }; // expected-error {{redefinition of 'InInterfaceCurlies'}}
enum InInterface { kA = 10, }; // expected-error {{redefinition of 'InInterface'}}
#ifdef __cplusplus
enum class InInterfaceScoped { kAScoped = 10, }; // expected-error {{redefinition of 'InInterfaceScoped'}}
#endif

union InCategory { int a; long b; }; // expected-error {{redefinition of 'InCategory'}}

enum InExtensionCurliesWithField: int { kB = 11, }; // expected-error {{redefinition of 'InExtensionCurliesWithField'}}
struct InExtensionCurlies { int a; }; // expected-error {{redefinition of 'InExtensionCurlies'}}
union InExtension { int a; long b; }; // expected-error {{redefinition of 'InExtension'}}

union InImplementationCurliesWithField { int a; long b; }; // expected-error {{redefinition of 'InImplementationCurliesWithField'}}
enum InImplementationCurlies { kC = 12, }; // expected-error {{redefinition of 'InImplementationCurlies'}}
struct InImplementation { int a; }; // expected-error {{redefinition of 'InImplementation'}}

enum InCategoryImplementation { kD = 13, }; // expected-error {{redefinition of 'InCategoryImplementation'}}
