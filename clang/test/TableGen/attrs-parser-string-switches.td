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
// CHECK: .Case("test_uneval", (Syntax==AttributeCommonInfo::AS_GNU && !ScopeName) ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_CXX11 && ScopeName && ScopeName->getName()=="clang") ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_C23 && ScopeName && ScopeName->getName()=="clang") ? 1 : 0)
// CHECK: #endif // CLANG_ATTR_ARG_CONTEXT_LIST

// Test attributeHasIdentifierArg: Same spelling, one with and one without
// an IdentifierArg.
def TestIdentOne : Attr {
  let Spellings = [GNU<"test_ident">];
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

// Checks that the simple value is produced if only one attribute with a
// spelling.
def TestOnlyIdent : Attr {
  let Spellings = [GNU<"test_only_ident">];
  let Args = [EnumArgument<"Option", "OptionType", /*is_string=*/false,
              ["optA", "optB"], ["OPTA", "OPTB"]>];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_IDENTIFIER_ARG_LIST)
// CHECK: .Case("test_ident", (Syntax==AttributeCommonInfo::AS_GNU && !ScopeName)
// CHECK: .Case("test_only_ident", true)
// CHECK: .Case("test_targspec",
// CHECK-SAME: (T.getArch() == llvm::Triple::arm)) ? 1 : 0
// CHECK: #endif // CLANG_ATTR_IDENTIFIER_ARG_LIST

// Test attributeStringLiteralListArg : Same spelling, some with a
// StringArgument, some without, some in different locations.
def TestStringOne : DeclOrTypeAttr {
  let Spellings = [Clang<"test_string">];
  let Args = [UnsignedArgument<"unsarg">];
  let Subjects = SubjectList<[Function, TypedefName, ParmVar]>;
  let Documentation = [AcquireHandleDocs];
}

def TestStringTwo : InheritableAttr {
  let Spellings = [Pragma<"", "test_string">];
  let Args = [StringArgument<"strarg">];
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
// CHECK: .Case("test_string", (Syntax==AttributeCommonInfo::AS_Declspec && !ScopeName) ? 2 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_Pragma && !ScopeName) ? 1 : 0)
// CHECK: .Case("test_targspec",
// CHECK-SAME: (T.getArch() == llvm::Triple::arm)) ? 4294967294 :
// CHECK-SAME: (T.getArch() == llvm::Triple::ppc)) ? 1 :
// CHECK: #endif // CLANG_ATTR_STRING_LITERAL_ARG_LIST

// Test attributeHasVariadicIdentifierArg : One with VariadicIdentifierArgument
// and one without.
def TestVariadicIdentOne : InheritableAttr {
  let Spellings = [Clang<"test_var_ident">];
  let Args = [UnsignedArgument<"Hint">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def TestVariadicIdentTwo : InheritableAttr {
  let Spellings = [Pragma<"", "test_var_ident">];
  let Args = [VariadicIdentifierArgument<"iargs">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// CHECK: #if defined(CLANG_ATTR_VARIADIC_IDENTIFIER_ARG_LIST)
// CHECK: .Case("test_var_ident", (Syntax==AttributeCommonInfo::AS_Pragma && !ScopeName))
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
// CHECK: .Case("test_var_idx", (Syntax==AttributeCommonInfo::AS_GNU && !ScopeName) ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_CXX11 && ScopeName && ScopeName->getName()=="clang") ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_C23 && ScopeName && ScopeName->getName()=="clang") ? 1 : 0)
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
// CHECK: .Case("test_expr_pack", (Syntax==AttributeCommonInfo::AS_GNU && !ScopeName) ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_CXX11 && ScopeName && ScopeName->getName()=="clang") ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_C23 && ScopeName && ScopeName->getName()=="clang") ? 1 : 0)
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
// CHECK: .Case("test_type", (Syntax==AttributeCommonInfo::AS_GNU && !ScopeName) ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_CXX11 && ScopeName && ScopeName->getName()=="clang") ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_C23 && ScopeName && ScopeName->getName()=="clang") ? 1 : 0)
// CHECK: #endif // CLANG_ATTR_TYPE_ARG_LIST

// Test attributeHasStrictIdentifierArgs, one used StrictEnumParameters, the
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

// CHECK: #if defined(CLANG_ATTR_STRICT_IDENTIFIER_ARG_LIST)
// CHECK: .Case("strict_enum", (Syntax==AttributeCommonInfo::AS_GNU && !ScopeName) ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_CXX11 && ScopeName && ScopeName->getName()=="clang") ? 1 :
// CHECK-SAME: (Syntax==AttributeCommonInfo::AS_C23 && ScopeName && ScopeName->getName()=="clang") ? 1 : 0)
// CHECK: #endif // CLANG_ATTR_STRICT_IDENTIFIER_ARG_LIST

// Test that TargetSpecific attributes work as expected.

def TargSpecX86 : InheritableAttr, TargetSpecificAttr<TargetArch<["x86"]>> {
  let Spellings = [GCC<"test_targspec">];
  let Subjects = SubjectList<[Function]>;
  let Args = [UnsignedArgument<"ua">, DefaultIntArgument<"o", 0>];
  let ParseKind = "TargSpec";
  let Documentation = [Undocumented];
}
def TargSpecPPC : InheritableAttr, TargetSpecificAttr<TargetArch<["ppc"]>> {
  let Spellings = [GCC<"test_targspec">];
  let Subjects = SubjectList<[Function]>;
  let Args = [StringArgument<"str">, VariadicExprArgument<"args">];
  let ParseKind = "TargSpec";
  let Documentation = [Undocumented];
}
def TargSpecARM : InheritableAttr, TargetSpecificAttr<TargetArch<["arm"]>> {
  let Spellings = [GCC<"test_targspec">];
  let Subjects = SubjectList<[Function]>;
  let Args = [IdentifierArgument<"id">, VariadicStringArgument<"sargs">];
  let ParseKind = "TargSpec";
  let Documentation = [Undocumented];
}
