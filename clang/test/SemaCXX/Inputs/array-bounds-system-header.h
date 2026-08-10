// "System header" for testing that -Warray-bounds is properly suppressed in
// certain cases.

#define BAD_MACRO_1 \
    int i[3]; \
    i[3] = 5
#define BAD_MACRO_2(_b, _i) \
    (_b)[(_i)] = 5
#define QUESTIONABLE_MACRO(_a) \
    sizeof(_a) > 3 ? (_a)[3] = 5 : 5
#define NOP(x) (x)
