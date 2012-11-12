// Header for PCH test exprs.c

// DeclRefExpr
int i = 17;
enum Enum { Enumerator = 18 };
typedef typeof(i) int_decl_ref;
typedef typeof(Enumerator) enum_decl_ref;

// IntegerLiteral
typedef typeof(17) integer_literal;
typedef typeof(17l) long_literal;

// FloatingLiteral and ParenExpr
typedef typeof((42.5)) floating_literal;

// ImaginaryLiteral
typedef typeof(17.0i) imaginary_literal;

// StringLiteral
const char *hello = "Hello" "PCH" "World";

// CharacterLiteral
typedef typeof('a') char_literal;

// UnaryOperator
typedef typeof(-Enumerator) negate_enum;

// OffsetOfExpr
struct X {
  int member;
};
struct Y {
  struct X array[5];
};
struct Z {
  struct Y y;
};
typedef typeof(__builtin_offsetof(struct Z, y.array[1 + 2].member)) 
  offsetof_type;

// UnaryExprOrTypeTraitExpr
typedef typeof(sizeof(int)) typeof_sizeof;
typedef typeof(sizeof(Enumerator)) typeof_sizeof2;

// ArraySubscriptExpr
extern double values[];
typedef typeof(values[2]) array_subscript;

// CallExpr
double dplus(double x, double y);
double d0, d1;
typedef typeof((&dplus)(d0, d1)) call_returning_double;

// MemberExpr
struct S {
  double x;
};
typedef typeof(((struct S*)0)->x) member_ref_double;

// BinaryOperator
typedef typeof(i + Enumerator) add_result;

// CompoundAssignOperator
typedef typeof(i += Enumerator) addeq_result;

// ConditionalOperator
typedef typeof(i? : d0) conditional_operator;

// CStyleCastExpr
typedef typeof((void *)0) void_ptr;

// CompoundLiteral
typedef typeof((struct S){.x = 3.5}) compound_literal;

typedef typeof(i + sizeof(int[i + Enumerator])) add_result_with_typeinfo;

// ExtVectorElementExpr
typedef __attribute__(( ext_vector_type(2) )) double double2;
extern double2 vec2, vec2b;
typedef typeof(vec2.x) ext_vector_element;

// InitListExpr
double double_array[3] = { 1.0, 2.0 };

// DesignatedInitExpr
struct {
  int x;
  float y;
} designated_inits[3] = { [0].y = 17,
                          [2].x = 12.3, // expected-warning {{implicit conversion from 'double' to 'int' changes value from 12.3 to 12}}
                          3.5 };

// TypesCompatibleExpr
typedef typeof(__builtin_types_compatible_p(float, double)) types_compatible;

// ChooseExpr
typedef typeof(__builtin_choose_expr(17 > 19, d0, 1)) choose_expr;

// GNUNullExpr FIXME: needs C++
// typedef typeof(__null) null_type;

// ShuffleVectorExpr
typedef typeof(__builtin_shufflevector(vec2, vec2b, 2, 1)) shuffle_expr;

// GenericSelectionExpr
typedef typeof(_Generic(i, char*: 0, int: 0., default: hello))
  generic_selection_expr;
