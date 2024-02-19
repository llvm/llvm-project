// RUN: %clang_cc1 -verify -fsyntax-only %s

typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor:,CGColor))) CGColor *CGColorRefOk;
typedef struct __attribute__((objc_bridge_related(NSColor,,CGColor))) CGColor *CGColorRef1Ok;
typedef struct __attribute__((objc_bridge_related(NSColor,,))) CGColor *CGColorRef2Ok;
typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor:,))) CGColor *CGColorRef3Ok;

typedef struct __attribute__((objc_bridge_related(,colorWithCGColor:,CGColor))) CGColor *CGColorRef1NotOk; // expected-error {{expected a related ObjectiveC class name, e.g., 'NSColor'}}
typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor,CGColor))) CGColor *CGColorRef2NotOk; // expected-error {{expected a class method selector with single argument, e.g., 'colorWithCGColor:'}}
typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor::,CGColor))) CGColor *CGColorRef3NotOk; // expected-error {{expected a class method selector with single argument, e.g., 'colorWithCGColor:'}}
typedef struct __attribute__((objc_bridge_related(12,colorWithCGColor:,CGColor))) CGColor *CGColorRef4NotOk; // expected-error {{expected a related ObjectiveC class name, e.g., 'NSColor'}}
typedef struct __attribute__((objc_bridge_related(NSColor,+:,CGColor))) CGColor *CGColorRef5NotOk; // expected-error {{expected ','}}
typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor:,+))) CGColor *CGColorRef6NotOk; // expected-error {{expected ')'}}

