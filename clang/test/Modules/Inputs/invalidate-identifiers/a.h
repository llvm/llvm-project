// Ensure that loading 'i' introduces enough identifiers to cause the
// identifier table to be reallocated.
#define TYPEDEFS(x) typedef x##0 x; typedef x##1 x;
#define DEPTH_0(x) DEPTH_1(x##0) DEPTH_1(x##1) TYPEDEFS(x)
#define DEPTH_1(x) DEPTH_2(x##0) DEPTH_2(x##1) TYPEDEFS(x)
#define DEPTH_2(x) DEPTH_3(x##0) DEPTH_3(x##1) TYPEDEFS(x)
#define DEPTH_3(x) DEPTH_4(x##0) DEPTH_4(x##1) TYPEDEFS(x)
#define DEPTH_4(x) DEPTH_5(x##0) DEPTH_5(x##1) TYPEDEFS(x)
#define DEPTH_5(x) DEPTH_6(x##0) DEPTH_6(x##1) TYPEDEFS(x)
#define DEPTH_6(x) DEPTH_7(x##0) DEPTH_7(x##1) TYPEDEFS(x)
#define DEPTH_7(x) DEPTH_8(x##0) DEPTH_8(x##1) TYPEDEFS(x)
#define DEPTH_8(x) DEPTH_9(x##0) DEPTH_9(x##1) TYPEDEFS(x)
#define DEPTH_9(x) DEPTH_A(x##0) DEPTH_A(x##1) TYPEDEFS(x)
#define DEPTH_A(x) DEPTH_B(x##0) DEPTH_B(x##1) TYPEDEFS(x)
#define DEPTH_B(x) typedef int x;
DEPTH_0(i)
extern i v;
