class AttrSubject;

class Stmt<bit abstract = 0> : AttrSubject {
  bit Abstract = abstract;
}

class DStmt<Stmt base, bit abstract = 0> : Stmt<abstract> {
  Stmt Base = base;
}

// Statements
def NullStmt : Stmt;
def CompoundStmt : Stmt;
def LabelStmt : Stmt;
def AttributedStmt : Stmt;
def IfStmt : Stmt;
def SwitchStmt : Stmt;
def WhileStmt : Stmt;
def DoStmt : Stmt;
def ForStmt : Stmt;
def GotoStmt : Stmt;
def IndirectGotoStmt : Stmt;
def ContinueStmt : Stmt;
def BreakStmt : Stmt;
def ReturnStmt : Stmt;
def DeclStmt  : Stmt;
def SwitchCase : Stmt<1>;
def CaseStmt : DStmt<SwitchCase>;
def DefaultStmt : DStmt<SwitchCase>;

// Asm statements
def AsmStmt : Stmt<1>;
def GCCAsmStmt : DStmt<AsmStmt>;
def MSAsmStmt : DStmt<AsmStmt>;

// Obj-C statements
def ObjCAtTryStmt : Stmt;
def ObjCAtCatchStmt : Stmt;
def ObjCAtFinallyStmt : Stmt;
def ObjCAtThrowStmt : Stmt;
def ObjCAtSynchronizedStmt : Stmt;
def ObjCForCollectionStmt : Stmt;
def ObjCAutoreleasePoolStmt : Stmt;

// C++ statments
def CXXCatchStmt : Stmt;
def CXXTryStmt : Stmt;
def CXXForRangeStmt : Stmt;

// Expressions
def Expr : Stmt<1>;
def PredefinedExpr : DStmt<Expr>;
def DeclRefExpr : DStmt<Expr>;
def IntegerLiteral : DStmt<Expr>;
def FloatingLiteral : DStmt<Expr>;
def ImaginaryLiteral : DStmt<Expr>;
def StringLiteral : DStmt<Expr>;
def CharacterLiteral : DStmt<Expr>;
def ParenExpr : DStmt<Expr>;
def UnaryOperator : DStmt<Expr>;
def OffsetOfExpr : DStmt<Expr>;
def UnaryExprOrTypeTraitExpr : DStmt<Expr>;
def ArraySubscriptExpr : DStmt<Expr>;
def CallExpr : DStmt<Expr>;
def MemberExpr : DStmt<Expr>;
def CastExpr : DStmt<Expr, 1>;
def BinaryOperator : DStmt<Expr>;
def CompoundAssignOperator : DStmt<BinaryOperator>;
def AbstractConditionalOperator : DStmt<Expr, 1>;
def ConditionalOperator : DStmt<AbstractConditionalOperator>;
def BinaryConditionalOperator : DStmt<AbstractConditionalOperator>;
def ImplicitCastExpr : DStmt<CastExpr>;
def ExplicitCastExpr : DStmt<CastExpr, 1>;
def CStyleCastExpr : DStmt<ExplicitCastExpr>;
def CompoundLiteralExpr : DStmt<Expr>;
def ExtVectorElementExpr : DStmt<Expr>;
def InitListExpr : DStmt<Expr>;
def DesignatedInitExpr : DStmt<Expr>;
def ImplicitValueInitExpr : DStmt<Expr>;
def ParenListExpr : DStmt<Expr>;
def VAArgExpr : DStmt<Expr>;
def GenericSelectionExpr : DStmt<Expr>;
def PseudoObjectExpr : DStmt<Expr>;

// Atomic expressions
def AtomicExpr : DStmt<Expr>;

// GNU Extensions.
def AddrLabelExpr : DStmt<Expr>;
def StmtExpr : DStmt<Expr>;
def ChooseExpr : DStmt<Expr>;
def GNUNullExpr : DStmt<Expr>;

// C++ Expressions.
def CXXOperatorCallExpr : DStmt<CallExpr>;
def CXXMemberCallExpr : DStmt<CallExpr>;
def CXXNamedCastExpr : DStmt<ExplicitCastExpr, 1>;
def CXXStaticCastExpr : DStmt<CXXNamedCastExpr>;
def CXXDynamicCastExpr : DStmt<CXXNamedCastExpr>;
def CXXReinterpretCastExpr : DStmt<CXXNamedCastExpr>;
def CXXConstCastExpr : DStmt<CXXNamedCastExpr>;
def CXXFunctionalCastExpr : DStmt<ExplicitCastExpr>;
def CXXTypeidExpr : DStmt<Expr>;
def UserDefinedLiteral : DStmt<CallExpr>;
def CXXBoolLiteralExpr : DStmt<Expr>;
def CXXNullPtrLiteralExpr : DStmt<Expr>;
def CXXThisExpr : DStmt<Expr>;
def CXXThrowExpr : DStmt<Expr>;
def CXXDefaultArgExpr : DStmt<Expr>;
def CXXScalarValueInitExpr : DStmt<Expr>;
def CXXNewExpr : DStmt<Expr>;
def CXXDeleteExpr : DStmt<Expr>;
def CXXPseudoDestructorExpr : DStmt<Expr>;
def TypeTraitExpr : DStmt<Expr>;
def UnaryTypeTraitExpr : DStmt<Expr>;
def BinaryTypeTraitExpr : DStmt<Expr>;
def ArrayTypeTraitExpr : DStmt<Expr>;
def ExpressionTraitExpr : DStmt<Expr>;
def DependentScopeDeclRefExpr : DStmt<Expr>;
def CXXConstructExpr : DStmt<Expr>;
def CXXBindTemporaryExpr : DStmt<Expr>;
def ExprWithCleanups : DStmt<Expr>;
def CXXTemporaryObjectExpr : DStmt<CXXConstructExpr>;
def CXXUnresolvedConstructExpr : DStmt<Expr>;
def CXXDependentScopeMemberExpr : DStmt<Expr>;
def OverloadExpr : DStmt<Expr, 1>;
def UnresolvedLookupExpr : DStmt<OverloadExpr>;
def UnresolvedMemberExpr : DStmt<OverloadExpr>;
def CXXNoexceptExpr : DStmt<Expr>;
def PackExpansionExpr : DStmt<Expr>;
def SizeOfPackExpr : DStmt<Expr>;
def SubstNonTypeTemplateParmExpr : DStmt<Expr>;
def SubstNonTypeTemplateParmPackExpr : DStmt<Expr>;
def FunctionParmPackExpr : DStmt<Expr>;
def MaterializeTemporaryExpr : DStmt<Expr>;
def LambdaExpr : DStmt<Expr>;

// Obj-C Expressions.
def ObjCStringLiteral : DStmt<Expr>;
def ObjCBoxedExpr : DStmt<Expr>;
def ObjCArrayLiteral : DStmt<Expr>;
def ObjCDictionaryLiteral : DStmt<Expr>;
def ObjCEncodeExpr : DStmt<Expr>;
def ObjCMessageExpr : DStmt<Expr>;
def ObjCSelectorExpr : DStmt<Expr>;
def ObjCProtocolExpr : DStmt<Expr>;
def ObjCIvarRefExpr : DStmt<Expr>;
def ObjCPropertyRefExpr : DStmt<Expr>;
def ObjCIsaExpr : DStmt<Expr>;
def ObjCIndirectCopyRestoreExpr : DStmt<Expr>;
def ObjCBoolLiteralExpr : DStmt<Expr>;
def ObjCSubscriptRefExpr : DStmt<Expr>;

// Obj-C ARC Expressions.
def ObjCBridgedCastExpr : DStmt<ExplicitCastExpr>;

// CUDA Expressions.
def CUDAKernelCallExpr : DStmt<CallExpr>;

// Clang Extensions.
def ShuffleVectorExpr : DStmt<Expr>;
def BlockExpr : DStmt<Expr>;
def OpaqueValueExpr : DStmt<Expr>;

// Microsoft Extensions.
def CXXUuidofExpr : DStmt<Expr>; 
def SEHTryStmt : Stmt;
def SEHExceptStmt : Stmt;
def SEHFinallyStmt : Stmt;
def MSDependentExistsStmt : Stmt;

// OpenCL Extensions.
def AsTypeExpr : DStmt<Expr>;
