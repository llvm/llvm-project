// RUN: %clang_cc1  -fsyntax-only -fblocks -verify -Wno-objc-root-class %s
// rdar://10414277

@protocol P
void p_foo(void) {} // expected-warning {{function definition inside an Objective-C container is deprecated}}
@end

@interface I
void foo(void) {} // expected-warning {{function definition inside an Objective-C container is deprecated}}
inline void v_foo(void) {} // expected-warning {{function definition inside an Objective-C container is deprecated}}
static int s_foo(void) {return 0; } // expected-warning {{function definition inside an Objective-C container is deprecated}}
static inline int si_val(void) { return 1; } // expected-warning {{function definition inside an Objective-C container is deprecated}}
@end

@interface I(CAT)
void cat_foo(void) {} // expected-warning {{function definition inside an Objective-C container is deprecated}}
@end

@implementation I
inline void v_imp_foo(void) {} 
@end

@implementation I(CAT)
void cat_imp_foo(void) {} 
@end

// rdar://16859666
@interface PrototypeState

@property (strong, readwrite) id moin1; // expected-note {{property declared here}}

static inline void prototype_observe_moin1(void (^callback)(id)) { // expected-warning {{function definition inside an Objective-C container is deprecated}}
        (void)^(PrototypeState *prototypeState){
            callback(prototypeState.moin1); // expected-error {{use of Objective-C property in function nested in Objective-C container not supported, move function outside its container}}
        };
}
@end
