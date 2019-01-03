
@protocol foo;
@class itf;

// Expressions
typedef typeof(@"foo" "bar") objc_string;
typedef typeof(@encode(int)) objc_encode;
typedef typeof(@protocol(foo)) objc_protocol;
typedef typeof(@selector(noArgs)) objc_selector_noArgs;
typedef typeof(@selector(oneArg:)) objc_selector_oneArg;
typedef typeof(@selector(foo:bar:)) objc_selector_twoArg;


// Types.
typedef typeof(id<foo>) objc_id_protocol_ty;

typedef typeof(itf*) objc_interface_ty;
typedef typeof(itf<foo>*) objc_qual_interface_ty;

@interface PP
@property (assign) id prop;
@end

static inline id getPseudoObject(PP *p) {
    return p.prop;
}
