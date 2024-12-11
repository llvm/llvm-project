#pragma clang system_header

#include <ptrcheck.h>

struct foo {
    int *__counted_by(count) p;
    int count;
};

struct bar {
    int *__ended_by(end) p;
    int *end;
};

static inline struct foo funcInSDK1(int *p, int count) {
    //strict-error@+1{{initializing 'int *__single __counted_by(count)' (aka 'int *__single') with an expression of incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    struct foo f = { p, count };
    return f;
}

static inline struct foo funcInSDK2(int *p, int count) {
    //strict-error@+1{{initializing 'int *__single __counted_by(count)' (aka 'int *__single') with an expression of incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    return (struct foo){ p, count };
}

static inline struct foo funcInSDK3(int *p, int count) {
    struct foo f;
    //strict-error@+1{{assigning to 'int *__single __counted_by(count)' (aka 'int *__single') from incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    f.p = p; // This results in a RecoveryExpr, so later analysis cannot see the assignment.
    //strict-error@+1{{assignment to 'f.count' requires corresponding assignment to 'int *__single __counted_by(count)' (aka 'int *__single') 'f.p'; add self assignment 'f.p = f.p' if the value has not changed}}
    f.count = count;
    return f;
}


static inline struct foo funcInSDK4(int *p, int count) {
    struct foo f;
    //strict-error@+1{{assigning to 'int *__single __counted_by(count)' (aka 'int *__single') from incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    f.p = p;
    return f;
}

static inline struct foo funcInSDK5(int *p, int count) {
    struct foo f;
    //strict-error@+1{{assignment to 'f.count' requires corresponding assignment to 'int *__single __counted_by(count)' (aka 'int *__single') 'f.p'; add self assignment 'f.p = f.p' if the value has not changed}}
    f.count = count;
    return f;
}

static inline struct bar funcInSDK6(int *p, int *end) {
    //strict-error@+2{{initializing 'int *__single __ended_by(end)' (aka 'int *__single') with an expression of incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    //strict-error@+1{{initializing 'int *__single /* __started_by(p) */ ' (aka 'int *__single') with an expression of incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    struct bar b = { p, end };
    return b;
}

static inline struct bar funcInSDK7(int *p, int *end) {
    //strict-error@+2{{initializing 'int *__single __ended_by(end)' (aka 'int *__single') with an expression of incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    //strict-error@+1{{initializing 'int *__single /* __started_by(p) */ ' (aka 'int *__single') with an expression of incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    return (struct bar){ p, end };
}

static inline struct bar funcInSDK8(int *p, int *end) {
    struct bar b;
    //strict-error@+1{{assigning to 'int *__single __ended_by(end)' (aka 'int *__single') from incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    b.p = p;
    //strict-error@+1{{assigning to 'int *__single /* __started_by(p) */ ' (aka 'int *__single') from incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    b.end = end;
    return b;
}

static inline struct bar funcInSDK9(struct bar in) {
    struct bar b;
    //strict-error@+1{{assignment to 'int *__single __ended_by(end)' (aka 'int *__single') 'b.p' requires corresponding assignment to 'b.end'; add self assignment 'b.end = b.end' if the value has not changed}}
    b.p = in.p;
    return b;
}

static inline struct bar funcInSDK10(struct bar in) {
    struct bar b;
    //strict-error@+1{{assignment to 'int *__single __ended_by(end)' (aka 'int *__single') 'b.end' requires corresponding assignment to 'b.p'; add self assignment 'b.p = b.p' if the value has not changed}}
    b.end = in.end;
    return b;
}

static inline struct foo funcInSDK11(struct foo in) {
    struct foo f;
    //strict-error@+1{{assignment to 'int *__single __counted_by(count)' (aka 'int *__single') 'f.p' requires corresponding assignment to 'f.count'; add self assignment 'f.count = f.count' if the value has not changed}}
    f.p = in.p;
    return f;
}

static inline struct foo funcInSDK12(struct foo in) {
    struct foo f;
    //strict-error@+1{{assignment to 'f.count' requires corresponding assignment to 'int *__single __counted_by(count)' (aka 'int *__single') 'f.p'; add self assignment 'f.p = f.p' if the value has not changed}}
    f.count = in.count;
    return f;
}

static tmp() {
}
