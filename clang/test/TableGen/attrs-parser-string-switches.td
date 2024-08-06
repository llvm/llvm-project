// RUN: clang-tblgen -gen-clang-attr-parser-string-switches -I%p/../../include %s -o - 2>&1 | FileCheck %s

// Tests that the tablegen can support attributes with the same spellings but
// different argument types.

include "clang/Basic/Attr.td"

// Test attributeParsedArgsUnevaluated : different ParseArgumentsAsUnevaluated
def TestUnEvalOne : InheritableAttr {
  let Spellings = [Clang<"test_uneval">];
  let Args = [ExprArgument<"Count">];
  let Subjects = SubjectList<[Function]>;
  let ParseArgumentsAsUnevaluated = 1;
  let Documentation = [Undocumented];
}

def TestUnEvalTwo : InheritableAttr {
  let Spellings = [Pragma<"", "test_uneval">];
  let Args = [ExprArgument<"Count">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_ARG_CONTEXT_LIST)
// CHECK-NOT: .Case("Pragma::test_uneval", true)
// CHECK: .Case("GNU::test_uneval", true)
// CHECK-NOT: .Case("Pragma::test_uneval", true)
// CHECK: .Case("CXX11::clang::test_uneval", true)
// CHECK-NOT: .Case("Pragma::test_uneval", true)
// CHECK: .Case("C23::clang::test_uneval", true)
// CHECK-NOT: .Case("Pragma::test_uneval", true)
// CHECK: #endif // CLANG_ATTR_ARG_CONTEXT_LIST

// Test attributeHasIdentifierArg: Same spelling, one with and one without
// an IdentifierArg.
def TestIdentOne : Attr {
  let Spellings = [Clang<"test_ident">];
  let Args = [EnumArgument<"Option", "OptionType", /*is_string=*/false,
              ["optA", "optB"], ["OPTA", "OPTB"]>];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def TestIdentTwo : StmtAttr {
  let Spellings = [Pragma<"", "test_ident">];
  let Args = [UnsignedArgument<"val", /*opt*/1>];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_IDENTIFIER_ARG_LIST)
// CHECK-NOT: .Case("Pragma::test_ident", true)
// CHECK: .Case("GNU::test_ident", true)
// CHECK-NOT: .Case("Pragma::test_ident", true)
// CHECK: .Case("CXX11::clang::test_ident", true)
// CHECK-NOT: .Case("Pragma::test_ident", true)
// CHECK: .Case("C23::clang::test_ident", true)
// CHECK-NOT: .Case("Pragma::test_ident", true)
// CHECK: #endif // CLANG_ATTR_IDENTIFIER_ARG_LIST

// Test attributeStringLiteralListArg : Same spelling, some with a
// StringArgument, some without, some in different locations.
def TestStringOne : DeclOrTypeAttr {
  let Spellings = [Clang<"test_string">];
  let Args = [StringArgument<"strarg">];
  let Subjects = SubjectList<[Function, TypedefName, ParmVar]>;
  let Documentation = [AcquireHandleDocs];
}

def TestStringTwo : InheritableAttr {
  let Spellings = [Pragma<"", "test_string">];
  let Args = [UnsignedArgument<"unsarg">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// In a different position
def TestStringThree : Attr {
  let Spellings = [Declspec<"test_string">];
  let Args = [UnsignedArgument<"uarg">, StringArgument<"strarg">];
  let Subjects = SubjectList<[Function, TypedefName, ParmVar]>;
  let Documentation = [AcquireHandleDocs];
}

// CHECK: #if defined(CLANG_ATTR_STRING_LITERAL_ARG_LIST)
// CHECK-NOT: .Case("Pragma::test_string"
// CHECK: .Case("GNU::test_string", 1)
// CHECK: .Case("CXX11::clang::test_string", 1)
// CHECK: .Case("C23::clang::test_string", 1)
// CHECK-NOT: .Case("Pragma::test_string"
// CHECK: .Case("Declspec::test_string", 2)
// CHECK-NOT: .Case("Pragma::test_string"
// CHECK: #endif // CLANG_ATTR_STRING_LITERAL_ARG_LIST

// Test attributeHasVariadicIdentifierArg : One with VariadicIdentifierArgument
// and one without.
def TestVariadicIdentOne : InheritableAttr {
  let Spellings = [Clang<"test_var_ident">];
  let Args = [VariadicIdentifierArgument<"iargs">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def TestVariadicIdentTwo : InheritableAttr {
  let Spellings = [Pragma<"", "test_var_ident">];
  let Args = [UnsignedArgument<"Hint">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_VARIADIC_IDENTIFIER_ARG_LIST)
// CHECK-NOT: .Case("Pragma::"test_var_ident", true)
// CHECK: .Case("GNU::test_var_ident", true)
// CHECK-NOT: .Case("Pragma::test_var_ident", true)
// CHECK: .Case("CXX11::clang::test_var_ident", true)
// CHECK-NOT: .Case("Pragma::test_var_ident", true)
// CHECK: .Case("C23::clang::test_var_ident", true)
// CHECK-NOT: .Case("Pragma::test_var_ident", true)
// CHECK: #endif // CLANG_ATTR_VARIADIC_IDENTIFIER_ARG_LIST

// Test attributeTreatsKeywordThisAsIdentifier : Same spelling, one with and
// one without VariadicParamOrParamIdxArgument.
def TestVarOrIdxOne : InheritableAttr {
  let Spellings = [Clang<"test_var_idx">];
  let Args = [VariadicParamOrParamIdxArgument<"arg">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def TestVarOrIdxTwo : InheritableAttr {
  let Spellings = [Pragma<"", "test_var_idx">];
  let Args = [UnsignedArgument<"Hint">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_THIS_ISA_IDENTIFIER_ARG_LIST)
// CHECK-NOT: .Case("Pragma::test_var_idx", true)
// CHECK: .Case("GNU::test_var_idx", true)
// CHECK-NOT: .Case("Pragma::test_var_idx", true)
// CHECK: .Case("CXX11::clang::test_var_idx", true)
// CHECK-NOT: .Case("Pragma::test_var_idx", true)
// CHECK: .Case("C23::clang::test_var_idx", true)
// CHECK-NOT: .Case("Pragma::test_var_idx", true)
// CHECK: #endif // CLANG_ATTR_THIS_ISA_IDENTIFIER_ARG_LIST

// Test attributeAcceptsExprPack : One with, one without.
def TestExprPackOne : InheritableAttr {
  let Spellings = [Clang<"test_expr_pack">];
  let Args = [StringArgument<"str">, VariadicExprArgument<"args">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let AcceptsExprPack = 1;
  let Documentation = [Undocumented];
}

def TestExprPackTwo : InheritableAttr {
  let Spellings = [Pragma<"", "test_expr_pack">];
  let Args = [StringArgument<"str">, VariadicExprArgument<"args">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_ACCEPTS_EXPR_PACK)
// CHECK-NOT: .Case("Pragma::test_expr_pack", true)
// CHECK: .Case("GNU::test_expr_pack", true)
// CHECK-NOT: .Case("Pragma::test_expr_pack", true)
// CHECK: .Case("CXX11::clang::test_expr_pack", true)
// CHECK-NOT: .Case("Pragma::test_expr_pack", true)
// CHECK: .Case("C23::clang::test_expr_pack", true)
// CHECK-NOT: .Case("Pragma::test_expr_pack", true)
// CHECK: #endif // CLANG_ATTR_ACCEPTS_EXPR_PACK


// Test attributeIsTypeArgAttr : Same spelling, one with TypeArgument and one
// without.
def TestTypeOne : InheritableAttr {
  let Spellings = [Clang<"test_type">];
  let Args = [TypeArgument<"Hint">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def TestTypeTwo : InheritableAttr {
  let Spellings = [Pragma<"", "test_type">];
  let Args = [UnsignedArgument<"Hint">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_TYPE_ARG_LIST)
// CHECK-NOT: .Case("Pragma::test_type", true)
// CHECK: .Case("GNU::test_type", true)
// CHECK-NOT: .Case("Pragma::test_type", true)
// CHECK: .Case("CXX11::clang::test_type", true)
// CHECK-NOT: .Case("Pragma::test_type", true)
// CHECK: .Case("C23::clang::test_type", true)
// CHECK-NOT: .Case("Pragma::test_type", true)
// CHECK: #endif // CLANG_ATTR_TYPE_ARG_LIST

// Test attributeHasStrictIdentifierArgs and
// attributeHasStrictIdentifierArgAtIndex, one used StrictEnumParameters, the
// other does not.
def TestStrictEnumOne : InheritableAttr {
  let Spellings = [Clang<"strict_enum">];
  let StrictEnumParameters = 1;
  let Args = [EnumArgument<"One", "OneType", /*is_string=*/true,
                ["a", "b", "c", "d"],
                ["A", "B", "C", "D"]>,
              IntArgument<"Other", 1>,
              EnumArgument<"Two", "TwoType", /*is_string=*/true,
                ["e", "f", "g", "h"],
                ["E", "F", "G", "H"]>];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def TestStrictEnumTwo : InheritableAttr {
  let Spellings = [Pragma<"", "strict_enum">];
  let Args = [VariadicExprArgument<"Args">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_STRICT_IDENTIFIER_ARG_AT_INDEX_LIST)
// CHECK-NOT: .Case("Pragma::strict_enum", 5ull)
// CHECK: .Case("GNU::strict_enum", 5ull)
// CHECK-NOT: .Case("Pragma::strict_enum", 5ull)
// CHECK: .Case("CXX11::clang::strict_enum", 5ull)
// CHECK-NOT: .Case("Pragma::strict_enum", 5ull)
// CHECK: .Case("C23::clang::strict_enum", 5ull)
// CHECK-NOT: .Case("Pragma::strict_enum", 5ull)
// CHECK: #endif // CLANG_ATTR_STRICT_IDENTIFIER_ARG_AT_INDEX_LIST
