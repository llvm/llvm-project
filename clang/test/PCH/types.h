/* Used with the types.c test */

// TYPE_EXT_QUAL
typedef __attribute__((address_space(1))) int ASInt;

// TYPE_COMPLEX
typedef _Complex float Cfloat;

// TYPE_ATOMIC
typedef _Atomic(int) AtomicInt;

// TYPE_POINTER
typedef int * int_ptr;

// TYPE_BLOCK_POINTER
typedef int (^Block)(int, float);

// TYPE_CONSTANT_ARRAY
typedef int five_ints[5];

// TYPE_INCOMPLETE_ARRAY
typedef float float_array[];

// TYPE_VARIABLE_ARRAY in stmts.[ch]

// TYPE_VECTOR
typedef float float4 __attribute__((vector_size(16)));

// TYPE_EXT_VECTOR
typedef float ext_float4 __attribute__((ext_vector_type(4)));

// TYPE_FUNCTION_NO_PROTO
typedef int noproto();

// TYPE_FUNCTION_PROTO
typedef float proto(float, float, ...);

// TYPE_TYPEDEF
typedef int_ptr * int_ptr_ptr;

// TYPE_TYPEOF_EXPR
typedef typeof(17) typeof_17;

// TYPE_TYPEOF
typedef typeof(int_ptr *) int_ptr_ptr2;

struct S2;
struct S2 {};
enum E;
enum E { myenum };
