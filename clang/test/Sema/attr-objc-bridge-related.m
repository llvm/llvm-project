// RUN: %clang_cc1 -verify -fsyntax-only %s

struct [[clang::objc_bridge_related(NSParagraphStyle,,)]] TestBridgedRef;

struct [[clang::objc_bridge_related(NSColor,colorWithCGColor:,CGColor)]] CGColorRefOk;
struct [[clang::objc_bridge_related(,colorWithCGColor:,CGColor)]] CGColorRef1NotOk; // expected-error {{expected a related ObjectiveC class name, e.g., 'NSColor'}}
struct [[clang::objc_bridge_related(NSColor,colorWithCGColor::,CGColor)]] CGColorRef3NotOk; // expected-error {{expected a class method selector with single argument, e.g., 'colorWithCGColor:'}}
