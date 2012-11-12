// Test this without pch.
// RUN: %clang_cc1 -fblocks -include %S/exprs.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -fblocks -o %t %S/exprs.h
// RUN: %clang_cc1 -fblocks -include-pch %t -fsyntax-only -verify %s -DWITH_PCH

#ifdef WITH_PCH
// expected-no-diagnostics
#endif

__SIZE_TYPE__ size_type_value;
int integer;
long long_integer;
double floating;
_Complex double floating_complex;

// DeclRefExpr
int_decl_ref *int_ptr1 = &integer;
enum_decl_ref *enum_ptr1 = &integer;

// IntegerLiteral
integer_literal *int_ptr2 = &integer;
long_literal *long_ptr1 = &long_integer;

// FloatingLiteral + ParenExpr
floating_literal *double_ptr = &floating;

// ImaginaryLiteral
imaginary_literal *cdouble_ptr = &floating_complex;

// StringLiteral
const char* printHello() {
  return hello;
}

// CharacterLiteral
char_literal *int_ptr3 = &integer;

// UnaryOperator
negate_enum *int_ptr4 = &integer;

// OffsetOfExpr
offsetof_type *offsetof_ptr = &size_type_value;

// UnaryExprOrTypeTraitExpr
typeof(sizeof(float)) size_t_value;
typeof_sizeof *size_t_ptr = &size_t_value;
typeof_sizeof2 *size_t_ptr2 = &size_t_value;

// ArraySubscriptExpr
array_subscript *double_ptr1_5 = &floating;

// CallExpr
call_returning_double *double_ptr2 = &floating;

// MemberExpr
member_ref_double *double_ptr3 = &floating;

// BinaryOperator
add_result *int_ptr5 = &integer;

// CompoundAssignOperator
addeq_result *int_ptr6 = &integer;

add_result_with_typeinfo *int_typeinfo_ptr6;

// ConditionalOperator
conditional_operator *double_ptr4 = &floating;

// CStyleCastExpr
void_ptr vp1 = &integer;

// CompoundLiteral
struct S s;
compound_literal *sptr = &s;

// ExtVectorElementExpr
ext_vector_element *double_ptr5 = &floating;

// InitListExpr
double get_from_double_array(unsigned Idx) { return double_array[Idx]; }

/// DesignatedInitExpr
float get_from_designated(unsigned Idx) {
  return designated_inits[2].y;
}

// TypesCompatibleExpr
types_compatible *int_ptr7 = &integer;

// ChooseExpr
choose_expr *int_ptr8 = &integer;

// GNUNullExpr FIXME: needs C++
//null_type null = __null;

// ShuffleVectorExpr
shuffle_expr *vec_ptr = &vec2;

// GenericSelectionExpr
generic_selection_expr *double_ptr6 = &floating;
