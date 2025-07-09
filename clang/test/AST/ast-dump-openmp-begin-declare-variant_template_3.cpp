// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s | FileCheck %s
// expected-no-diagnostics
// PR47655
template <typename T> struct S {
 S(int, T *) {}
};

template <typename T>
int also_before(T s) {
 return 0;
}

#pragma omp begin declare variant match(implementation = {extension(allow_templates)})
template <typename T>
int also_before(S<T> s) {
 // Ensure there is no error because this is never instantiated.
 double t;
 S<T> q(1, &t);
 return 1;
}
template <typename T>
int special(S<T> s) {
 T t;
 S<T> q(0, &t);
 return 0;
}
template <typename T>
int also_after(S<T> s) {
 // Ensure there is no error because this is never instantiated.
 double t;
 S<T> q(2.0, &t);
 return 2;
}
#pragma omp end declare variant

template <typename T>
int also_after(T s) {
 return 0;
}

int test() {
 // Should return 0.
 return also_before(0) + also_after(0) + also_before(0.) + also_after(0.) + special(S<int>(0, 0));
}

//CHECK: |-ClassTemplateDecl {{.*}} S
//CHECK: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
//CHECK: | |-CXXRecordDecl {{.*}} struct S definition
//CHECK: | | |-DefinitionData empty standard_layout trivially_copyable has_user_declared_ctor can_const_default_init
//CHECK: | | | |-DefaultConstructor defaulted_is_constexpr
//CHECK: | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
//CHECK: | | | |-MoveConstructor exists simple trivial needs_implicit
//CHECK: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
//CHECK: | | | |-MoveAssignment exists simple trivial needs_implicit
//CHECK: | | | `-Destructor simple irrelevant trivial needs_implicit
//CHECK: | | |-CXXRecordDecl {{.*}} implicit referenced struct S
//CHECK: | | `-CXXConstructorDecl {{.*}} S<T> 'void (int, T *)' implicit-inline
//CHECK: | | |-ParmVarDecl {{.*}} 'int'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | |-ParmVarDecl {{.*}} 'T *'
//CHECK: | | | `-typeDetails: PointerType {{.*}} 'T *' dependent
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | `-CompoundStmt {{.*}} 
//CHECK: | |-ClassTemplateSpecializationDecl {{.*}} struct S definition implicit_instantiation
//CHECK: | | |-DefinitionData pass_in_registers empty standard_layout trivially_copyable has_user_declared_ctor can_const_default_init
//CHECK: | | | |-DefaultConstructor defaulted_is_constexpr
//CHECK: | | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
//CHECK: | | | |-MoveConstructor exists simple trivial
//CHECK: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
//CHECK: | | | |-MoveAssignment exists simple trivial needs_implicit
//CHECK: | | | `-Destructor simple irrelevant trivial
//CHECK: | | |-TemplateArgument type 'int'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | |-CXXRecordDecl {{.*}} implicit struct S
//CHECK: | | |-CXXConstructorDecl {{.*}} used S 'void (int, int *)' implicit_instantiation implicit-inline instantiated_from {{.*}}
//CHECK: | | | |-ParmVarDecl {{.*}} 'int'
//CHECK: | | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | | |-ParmVarDecl {{.*}} 'int *'
//CHECK: | | | | `-typeDetails: PointerType {{.*}} 'int *'
//CHECK: | | | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
//CHECK: | | | | |-ClassTemplateSpecialization {{.*}} 'S'
//CHECK: | | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | | `-CompoundStmt {{.*}} 
//CHECK: | | |-CXXConstructorDecl {{.*}} implicit constexpr S 'void (const S<int> &)' inline default trivial noexcept-unevaluated {{.*}}
//CHECK: | | | `-ParmVarDecl {{.*}} 'const S<int> &'
//CHECK: | | | `-typeDetails: LValueReferenceType {{.*}} 'const S<int> &'
//CHECK: | | | `-qualTypeDetail: QualType {{.*}} 'const S<int>' const
//CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'S<int>' sugar
//CHECK: | | | `-typeDetails: RecordType {{.*}} 'S<int>'
//CHECK: | | | `-ClassTemplateSpecialization {{.*}} 'S'
//CHECK: | | |-CXXConstructorDecl {{.*}} implicit constexpr S 'void (S<int> &&)' inline default trivial noexcept-unevaluated {{.*}}
//CHECK: | | | `-ParmVarDecl {{.*}} 'S<int> &&'
//CHECK: | | | `-typeDetails: RValueReferenceType {{.*}} 'S<int> &&'
//CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'S<int>' sugar
//CHECK: | | | `-typeDetails: RecordType {{.*}} 'S<int>'
//CHECK: | | | `-ClassTemplateSpecialization {{.*}} 'S'
//CHECK: | | `-CXXDestructorDecl {{.*}} implicit referenced ~S 'void () noexcept' inline default trivial
//CHECK: | `-ClassTemplateSpecializationDecl {{.*}} struct S
//CHECK: | `-TemplateArgument type 'double'
//CHECK: | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: |-FunctionTemplateDecl {{.*}} also_before
//CHECK: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
//CHECK: | |-FunctionDecl {{.*}} also_before 'int (T)'
//CHECK: | | |-ParmVarDecl {{.*}} s 'T'
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | |-CompoundStmt {{.*}} 
//CHECK: | | | `-ReturnStmt {{.*}} 
//CHECK: | | | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: | | `-attrDetails: OMPDeclareVariantAttr {{.*}} <<invalid sloc>> Implicit implementation={extension(allow_templates)}
//CHECK: | | `-DeclRefExpr {{.*}} 'int (S<T>)' Function {{.*}} 'also_before[implementation={extension(allow_templates)}]' 'int (S<T>)'
//CHECK: | |-FunctionDecl {{.*}} used also_before 'int (int)' implicit_instantiation
//CHECK: | | |-TemplateArgument type 'int'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | |-ParmVarDecl {{.*}} s 'int'
//CHECK: | | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
//CHECK: | | | |-FunctionTemplate {{.*}} 'also_before'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | `-CompoundStmt {{.*}} 
//CHECK: | | `-ReturnStmt {{.*}} 
//CHECK: | | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: | `-FunctionDecl {{.*}} used also_before 'int (double)' implicit_instantiation
//CHECK: | |-TemplateArgument type 'double'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: | |-ParmVarDecl {{.*}} s 'double'
//CHECK: | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'double' sugar typename depth 0 index 0 T
//CHECK: | | |-FunctionTemplate {{.*}} 'also_before'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: | `-CompoundStmt {{.*}} 
//CHECK: | `-ReturnStmt {{.*}} 
//CHECK: | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: |-FunctionTemplateDecl {{.*}} also_before[implementation={extension(allow_templates)}]
//CHECK: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
//CHECK: | |-FunctionDecl {{.*}} referenced also_before[implementation={extension(allow_templates)}] 'int (S<T>)'
//CHECK: | | |-ParmVarDecl {{.*}} s 'S<T>'
//CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'S<T>' sugar dependent
//CHECK: | | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<T>' dependent
//CHECK: | | | |-name: 'S' qualified
//CHECK: | | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | | `-TemplateArgument type 'T':'type-parameter-0-0'
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | `-CompoundStmt {{.*}} 
//CHECK: | | |-DeclStmt {{.*}} 
//CHECK: | | | `-VarDecl {{.*}} referenced t 'double'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: | | |-DeclStmt {{.*}} 
//CHECK: | | | `-VarDecl {{.*}} q 'S<T>' callinit
//CHECK: | | | |-ParenListExpr {{.*}} 'NULL TYPE'
//CHECK: | | | | |-IntegerLiteral {{.*}} 'int' 1
//CHECK: | | | | `-UnaryOperator {{.*}} 'double *' prefix '&' cannot overflow
//CHECK: | | | | `-DeclRefExpr {{.*}} 'double' lvalue Var {{.*}} 't' 'double'
//CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'S<T>' sugar dependent
//CHECK: | | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<T>' dependent
//CHECK: | | | |-name: 'S' qualified
//CHECK: | | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | | `-TemplateArgument type 'T':'type-parameter-0-0'
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | `-ReturnStmt {{.*}} 
//CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
//CHECK: | |-FunctionDecl {{.*}} also_before[implementation={extension(allow_templates)}] 'int (S<int>)' implicit_instantiation
//CHECK: | | |-TemplateArgument type 'int'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | `-ParmVarDecl {{.*}} s 'S<int>'
//CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'S<int>' sugar
//CHECK: | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<int>' sugar
//CHECK: | | |-name: 'S' qualified
//CHECK: | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | |-TemplateArgument type 'int'
//CHECK: | | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
//CHECK: | | | |-FunctionTemplate {{.*}} 'also_before[implementation={extension(allow_templates)}]'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | `-typeDetails: RecordType {{.*}} 'S<int>'
//CHECK: | | `-ClassTemplateSpecialization {{.*}} 'S'
//CHECK: | `-FunctionDecl {{.*}} also_before[implementation={extension(allow_templates)}] 'int (S<double>)' implicit_instantiation
//CHECK: | |-TemplateArgument type 'double'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: | `-ParmVarDecl {{.*}} s 'S<double>'
//CHECK: | `-typeDetails: ElaboratedType {{.*}} 'S<double>' sugar
//CHECK: | `-typeDetails: TemplateSpecializationType {{.*}} 'S<double>' sugar
//CHECK: | |-name: 'S' qualified
//CHECK: | | `-ClassTemplateDecl {{.*}} S
//CHECK: | |-TemplateArgument type 'double'
//CHECK: | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'double' sugar typename depth 0 index 0 T
//CHECK: | | |-FunctionTemplate {{.*}} 'also_before[implementation={extension(allow_templates)}]'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: | `-typeDetails: RecordType {{.*}} 'S<double>'
//CHECK: | `-ClassTemplateSpecialization {{.*}} 'S'
//CHECK: |-FunctionTemplateDecl {{.*}} implicit special
//CHECK: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
//CHECK: | |-FunctionDecl {{.*}} special 'int (S<T>)'
//CHECK: | | |-ParmVarDecl {{.*}} s 'S<T>'
//CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'S<T>' sugar dependent
//CHECK: | | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<T>' dependent
//CHECK: | | | |-name: 'S' qualified
//CHECK: | | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | | `-TemplateArgument type 'T':'type-parameter-0-0'
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | `-attrDetails: OMPDeclareVariantAttr {{.*}} <<invalid sloc>> Implicit implementation={extension(allow_templates)}
//CHECK: | | `-DeclRefExpr {{.*}} 'int (S<T>)' Function {{.*}} 'special[implementation={extension(allow_templates)}]' 'int (S<T>)'
//CHECK: | `-FunctionDecl {{.*}} used special 'int (S<int>)' implicit_instantiation
//CHECK: | |-TemplateArgument type 'int'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | |-ParmVarDecl {{.*}} s 'S<int>'
//CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'S<int>' sugar
//CHECK: | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<int>' sugar
//CHECK: | | |-name: 'S' qualified
//CHECK: | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | |-TemplateArgument type 'int'
//CHECK: | | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
//CHECK: | | | |-FunctionTemplate {{.*}} 'special'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | `-typeDetails: RecordType {{.*}} 'S<int>'
//CHECK: | | `-ClassTemplateSpecialization {{.*}} 'S'
//CHECK: | `-attrDetails: OMPDeclareVariantAttr {{.*}} <<invalid sloc>> Implicit implementation={extension(allow_templates)}
//CHECK: | `-DeclRefExpr {{.*}} 'int (S<int>)' Function {{.*}} 'special[implementation={extension(allow_templates)}]' 'int (S<int>)'
//CHECK: |-FunctionTemplateDecl {{.*}} special[implementation={extension(allow_templates)}]
//CHECK: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
//CHECK: | |-FunctionDecl {{.*}} referenced special[implementation={extension(allow_templates)}] 'int (S<T>)'
//CHECK: | | |-ParmVarDecl {{.*}} s 'S<T>'
//CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'S<T>' sugar dependent
//CHECK: | | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<T>' dependent
//CHECK: | | | |-name: 'S' qualified
//CHECK: | | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | | `-TemplateArgument type 'T':'type-parameter-0-0'
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | `-CompoundStmt {{.*}} 
//CHECK: | | |-DeclStmt {{.*}} 
//CHECK: | | | `-VarDecl {{.*}} referenced t 'T'
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | |-DeclStmt {{.*}} 
//CHECK: | | | `-VarDecl {{.*}} q 'S<T>' callinit
//CHECK: | | | |-ParenListExpr {{.*}} 'NULL TYPE'
//CHECK: | | | | |-IntegerLiteral {{.*}} 'int' 0
//CHECK: | | | | `-UnaryOperator {{.*}} '<dependent type>' prefix '&' cannot overflow
//CHECK: | | | | `-DeclRefExpr {{.*}} 'T' lvalue Var {{.*}} 't' 'T'
//CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'S<T>' sugar dependent
//CHECK: | | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<T>' dependent
//CHECK: | | | |-name: 'S' qualified
//CHECK: | | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | | `-TemplateArgument type 'T':'type-parameter-0-0'
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | `-ReturnStmt {{.*}} 
//CHECK: | | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: | `-FunctionDecl {{.*}} special[implementation={extension(allow_templates)}] 'int (S<int>)' implicit_instantiation
//CHECK: | |-TemplateArgument type 'int'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | |-ParmVarDecl {{.*}} s 'S<int>'
//CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'S<int>' sugar
//CHECK: | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<int>' sugar
//CHECK: | | |-name: 'S' qualified
//CHECK: | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | |-TemplateArgument type 'int'
//CHECK: | | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
//CHECK: | | | |-FunctionTemplate {{.*}} 'special[implementation={extension(allow_templates)}]'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | `-typeDetails: RecordType {{.*}} 'S<int>'
//CHECK: | | `-ClassTemplateSpecialization {{.*}} 'S'
//CHECK: | `-CompoundStmt {{.*}} 
//CHECK: | |-DeclStmt {{.*}} 
//CHECK: | | `-VarDecl {{.*}} used t 'int'
//CHECK: | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
//CHECK: | | |-Function {{.*}} 'special[implementation={extension(allow_templates)}]' 'int (S<int>)'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | |-DeclStmt {{.*}} 
//CHECK: | | `-VarDecl {{.*}} q 'S<int>' callinit
//CHECK: | | |-CXXConstructExpr {{.*}} 'S<int>' 'void (int, int *)'
//CHECK: | | | |-IntegerLiteral {{.*}} 'int' 0
//CHECK: | | | `-UnaryOperator {{.*}} 'int *' prefix '&' cannot overflow
//CHECK: | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 't' 'int'
//CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'S<int>' sugar
//CHECK: | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<int>' sugar
//CHECK: | | |-name: 'S' qualified
//CHECK: | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | |-TemplateArgument type 'int'
//CHECK: | | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
//CHECK: | | | |-Function {{.*}} 'special[implementation={extension(allow_templates)}]' 'int (S<int>)'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | `-typeDetails: RecordType {{.*}} 'S<int>'
//CHECK: | | `-ClassTemplateSpecialization {{.*}} 'S'
//CHECK: | `-ReturnStmt {{.*}} 
//CHECK: | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: |-FunctionTemplateDecl {{.*}} implicit also_after
//CHECK: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
//CHECK: | `-FunctionDecl {{.*}} also_after 'int (S<T>)'
//CHECK: | |-ParmVarDecl {{.*}} s 'S<T>'
//CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'S<T>' sugar dependent
//CHECK: | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<T>' dependent
//CHECK: | | |-name: 'S' qualified
//CHECK: | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | `-TemplateArgument type 'T':'type-parameter-0-0'
//CHECK: | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | `-attrDetails: OMPDeclareVariantAttr {{.*}} <<invalid sloc>> Implicit implementation={extension(allow_templates)}
//CHECK: | `-DeclRefExpr {{.*}} 'int (S<T>)' Function {{.*}} 'also_after[implementation={extension(allow_templates)}]' 'int (S<T>)'
//CHECK: |-FunctionTemplateDecl {{.*}} also_after[implementation={extension(allow_templates)}]
//CHECK: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
//CHECK: | `-FunctionDecl {{.*}} also_after[implementation={extension(allow_templates)}] 'int (S<T>)'
//CHECK: | |-ParmVarDecl {{.*}} s 'S<T>'
//CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'S<T>' sugar dependent
//CHECK: | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<T>' dependent
//CHECK: | | |-name: 'S' qualified
//CHECK: | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | `-TemplateArgument type 'T':'type-parameter-0-0'
//CHECK: | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | `-CompoundStmt {{.*}} 
//CHECK: | |-DeclStmt {{.*}} 
//CHECK: | | `-VarDecl {{.*}} referenced t 'double'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: | |-DeclStmt {{.*}} 
//CHECK: | | `-VarDecl {{.*}} q 'S<T>' callinit
//CHECK: | | |-ParenListExpr {{.*}} 'NULL TYPE'
//CHECK: | | | |-FloatingLiteral {{.*}} 'double' 2.000000e+00
//CHECK: | | | `-UnaryOperator {{.*}} 'double *' prefix '&' cannot overflow
//CHECK: | | | `-DeclRefExpr {{.*}} 'double' lvalue Var {{.*}} 't' 'double'
//CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'S<T>' sugar dependent
//CHECK: | | `-typeDetails: TemplateSpecializationType {{.*}} 'S<T>' dependent
//CHECK: | | |-name: 'S' qualified
//CHECK: | | | `-ClassTemplateDecl {{.*}} S
//CHECK: | | `-TemplateArgument type 'T':'type-parameter-0-0'
//CHECK: | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | `-ReturnStmt {{.*}} 
//CHECK: | `-IntegerLiteral {{.*}} 'int' 2
//CHECK: |-FunctionTemplateDecl {{.*}} also_after
//CHECK: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
//CHECK: | |-FunctionDecl {{.*}} also_after 'int (T)'
//CHECK: | | |-ParmVarDecl {{.*}} s 'T'
//CHECK: | | | `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
//CHECK: | | | `-TemplateTypeParm {{.*}} 'T'
//CHECK: | | `-CompoundStmt {{.*}} 
//CHECK: | | `-ReturnStmt {{.*}} 
//CHECK: | | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: | |-FunctionDecl {{.*}} used also_after 'int (int)' implicit_instantiation
//CHECK: | | |-TemplateArgument type 'int'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | |-ParmVarDecl {{.*}} s 'int'
//CHECK: | | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
//CHECK: | | | |-FunctionTemplate {{.*}} 'also_after'
//CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'int'
//CHECK: | | `-CompoundStmt {{.*}} 
//CHECK: | | `-ReturnStmt {{.*}} 
//CHECK: | | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: | `-FunctionDecl {{.*}} used also_after 'int (double)' implicit_instantiation
//CHECK: | |-TemplateArgument type 'double'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: | |-ParmVarDecl {{.*}} s 'double'
//CHECK: | | `-typeDetails: SubstTemplateTypeParmType {{.*}} 'double' sugar typename depth 0 index 0 T
//CHECK: | | |-FunctionTemplate {{.*}} 'also_after'
//CHECK: | | `-typeDetails: BuiltinType {{.*}} 'double'
//CHECK: | `-CompoundStmt {{.*}} 
//CHECK: | `-ReturnStmt {{.*}} 
//CHECK: | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: `-FunctionDecl {{.*}} test 'int ()'
//CHECK: `-CompoundStmt {{.*}} 
//CHECK: `-ReturnStmt {{.*}} 
//CHECK: `-BinaryOperator {{.*}} 'int' '+'
//CHECK: |-BinaryOperator {{.*}} 'int' '+'
//CHECK: | |-BinaryOperator {{.*}} 'int' '+'
//CHECK: | | |-BinaryOperator {{.*}} 'int' '+'
//CHECK: | | | |-CallExpr {{.*}} 'int'
//CHECK: | | | | |-ImplicitCastExpr {{.*}} 'int (*)(int)' <FunctionToPointerDecay>
//CHECK: | | | | | `-DeclRefExpr {{.*}} 'int (int)' lvalue Function {{.*}} 'also_before' 'int (int)' (FunctionTemplate {{.*}} 'also_before')
//CHECK: | | | | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: | | | `-CallExpr {{.*}} 'int'
//CHECK: | | | |-ImplicitCastExpr {{.*}} 'int (*)(int)' <FunctionToPointerDecay>
//CHECK: | | | | `-DeclRefExpr {{.*}} 'int (int)' lvalue Function {{.*}} 'also_after' 'int (int)' (FunctionTemplate {{.*}} 'also_after')
//CHECK: | | | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: | | `-CallExpr {{.*}} 'int'
//CHECK: | | |-ImplicitCastExpr {{.*}} 'int (*)(double)' <FunctionToPointerDecay>
//CHECK: | | | `-DeclRefExpr {{.*}} 'int (double)' lvalue Function {{.*}} 'also_before' 'int (double)' (FunctionTemplate {{.*}} 'also_before')
//CHECK: | | `-FloatingLiteral {{.*}} 'double' 0.000000e+00
//CHECK: | `-CallExpr {{.*}} 'int'
//CHECK: | |-ImplicitCastExpr {{.*}} 'int (*)(double)' <FunctionToPointerDecay>
//CHECK: | | `-DeclRefExpr {{.*}} 'int (double)' lvalue Function {{.*}} 'also_after' 'int (double)' (FunctionTemplate {{.*}} 'also_after')
//CHECK: | `-FloatingLiteral {{.*}} 'double' 0.000000e+00
//CHECK: `-PseudoObjectExpr {{.*}} 'int'
//CHECK: |-CallExpr {{.*}} 'int'
//CHECK: | |-ImplicitCastExpr {{.*}} 'int (*)(S<int>)' <FunctionToPointerDecay>
//CHECK: | | `-DeclRefExpr {{.*}} 'int (S<int>)' lvalue Function {{.*}} 'special' 'int (S<int>)' (FunctionTemplate {{.*}} 'special')
//CHECK: | `-CXXTemporaryObjectExpr {{.*}} 'S<int>' 'void (int, int *)'
//CHECK: | |-IntegerLiteral {{.*}} 'int' 0
//CHECK: | `-ImplicitCastExpr {{.*}} 'int *' <NullToPointer>
//CHECK: | `-IntegerLiteral {{.*}} 'int' 0
//CHECK: `-CallExpr {{.*}} 'int'
//CHECK: |-ImplicitCastExpr {{.*}} 'int (*)(S<int>)' <FunctionToPointerDecay>
//CHECK: | `-DeclRefExpr {{.*}} 'int (S<int>)' Function {{.*}} 'special[implementation={extension(allow_templates)}]' 'int (S<int>)'
//CHECK: `-CXXTemporaryObjectExpr {{.*}} 'S<int>' 'void (int, int *)'
//CHECK: |-IntegerLiteral {{.*}} 'int' 0
//CHECK: `-ImplicitCastExpr {{.*}} 'int *' <NullToPointer>
//CHECK: `-IntegerLiteral {{.*}} 'int' 0