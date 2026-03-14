// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

@interface NSObj
@end

@interface NSChild : NSObj
@end

static_assert(__is_base_of(NSObj, NSChild), "");
static_assert(!__is_base_of(NSChild, NSObj), "");

static_assert(__is_base_of(NSObj, NSObj), "");

static_assert(!__is_base_of(NSObj *, NSChild *), "");
static_assert(!__is_base_of(NSChild *, NSObj *), "");

static_assert(__is_base_of(const volatile NSObj, NSChild), "");
static_assert(__is_base_of(NSObj, const volatile NSChild), "");

@class NSForward; // expected-note{{forward declaration of class}}

static_assert(!__is_base_of(NSForward, NSObj), "");
static_assert(!__is_base_of(NSObj, NSForward), ""); // expected-error{{incomplete type 'NSForward'}}

static_assert(!__is_base_of(id, NSObj), "");
