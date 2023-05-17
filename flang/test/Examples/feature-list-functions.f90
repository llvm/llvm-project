! UNSUPPORTED: system-windows
! REQUIRES: plugins, shell, examples

! RUN: %flang_fc1 -load %llvmshlibdir/flangFeatureList%pluginext \
! RUN:            -plugin feature-list %s 2>&1 | FileCheck %s

program list_features_test
    implicit none
    call test_sub(test_func(2, 3), 4)
contains
    subroutine test_sub(a, b)
        integer, intent(in) :: a, b
        print "(I0)", a + b
    end subroutine

    integer function test_func(a, b)
        integer, intent(in) :: a, b
        test_func = a * b
    end function
end program list_features_test

! CHECK: Name: 19
! CHECK-NEXT: Expr: 11
! CHECK-NEXT: DataRef: 5
! CHECK-NEXT: Designator: 5
! CHECK-NEXT: ActualArg: 4
! CHECK-NEXT: ActualArgSpec: 4
! CHECK-NEXT: EntityDecl: 4
! CHECK-NEXT: LiteralConstant: 4
! CHECK-NEXT: ActionStmt: 3
! CHECK-NEXT: Block: 3
! CHECK-NEXT: DeclarationTypeSpec: 3
! CHECK-NEXT: ExecutableConstruct: 3
! CHECK-NEXT: ExecutionPart: 3
! CHECK-NEXT: ExecutionPartConstruct: 3
! CHECK-NEXT: ImplicitPart: 3
! CHECK-NEXT: IntLiteralConstant: 3
! CHECK-NEXT: IntegerTypeSpec: 3
! CHECK-NEXT: IntrinsicTypeSpec: 3
! CHECK-NEXT: SpecificationPart: 3
! CHECK-NEXT: AttrSpec: 2
! CHECK-NEXT: Call: 2
! CHECK-NEXT: DeclarationConstruct: 2
! CHECK-NEXT: DummyArg: 2
! CHECK-NEXT: IntentSpec: 2
! CHECK-NEXT: IntentSpec::Intent: 2
! CHECK-NEXT: InternalSubprogram: 2
! CHECK-NEXT: ProcedureDesignator: 2
! CHECK-NEXT: SpecificationConstruct: 2
! CHECK-NEXT: TypeDeclarationStmt: 2
! CHECK-NEXT: AssignmentStmt: 1
! CHECK-NEXT: CallStmt: 1
! CHECK-NEXT: CharLiteralConstant: 1
! CHECK-NEXT: ContainsStmt: 1
! CHECK-NEXT: EndFunctionStmt: 1
! CHECK-NEXT: EndProgramStmt: 1
! CHECK-NEXT: EndSubroutineStmt: 1
! CHECK-NEXT: Expr::Add: 1
! CHECK-NEXT: Expr::Multiply: 1
! CHECK-NEXT: Format: 1
! CHECK-NEXT: FunctionReference: 1
! CHECK-NEXT: FunctionStmt: 1
! CHECK-NEXT: FunctionSubprogram: 1
! CHECK-NEXT: ImplicitPartStmt: 1
! CHECK-NEXT: ImplicitStmt: 1
! CHECK-NEXT: InternalSubprogramPart: 1
! CHECK-NEXT: MainProgram: 1
! CHECK-NEXT: OutputItem: 1
! CHECK-NEXT: PrefixSpec: 1
! CHECK-NEXT: PrintStmt: 1
! CHECK-NEXT: Program: 1
! CHECK-NEXT: ProgramStmt: 1
! CHECK-NEXT: ProgramUnit: 1
! CHECK-NEXT: SubroutineStmt: 1
! CHECK-NEXT: SubroutineSubprogram: 1
! CHECK-NEXT: Variable: 1
