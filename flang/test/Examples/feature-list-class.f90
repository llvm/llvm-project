! UNSUPPORTED: system-windows
! REQUIRES: plugins, shell, examples

! RUN: %flang_fc1 -load %llvmshlibdir/flangFeatureList%pluginext \
! RUN:            -plugin feature-list %s 2>&1 | FileCheck %s

module list_features_test
    implicit none

    type :: test_class_1
        integer :: a
        real :: b
    contains
        procedure :: sum => sum_test_class_1
        procedure :: set => set_values_test_class_1
    end type
contains
    real function sum_test_class_1(self)
        class(test_class_1), intent(in) :: self
        sum_test_class_1 = self%a + self%b
    end function

    subroutine set_values_test_class_1(self, a, b)
        class(test_class_1), intent(out) :: self
        integer, intent(in) :: a, b
        self%a = a
        self%b = b
    end subroutine
end module list_features_test

! CHECK: Name: 32
! CHECK-NEXT: DataRef: 11
! CHECK-NEXT: Designator: 7
! CHECK-NEXT: DeclarationTypeSpec: 6
! CHECK-NEXT: Expr: 5
! CHECK-NEXT: DeclarationConstruct: 4
! CHECK-NEXT: EntityDecl: 4
! CHECK-NEXT: IntrinsicTypeSpec: 4
! CHECK-NEXT: SpecificationConstruct: 4
! CHECK-NEXT: StructureComponent: 4
! CHECK-NEXT: ActionStmt: 3
! CHECK-NEXT: AssignmentStmt: 3
! CHECK-NEXT: AttrSpec: 3
! CHECK-NEXT: DummyArg: 3
! CHECK-NEXT: ExecutableConstruct: 3
! CHECK-NEXT: ExecutionPartConstruct: 3
! CHECK-NEXT: ImplicitPart: 3
! CHECK-NEXT: IntentSpec: 3
! CHECK-NEXT: IntentSpec::Intent: 3
! CHECK-NEXT: SpecificationPart: 3
! CHECK-NEXT: TypeDeclarationStmt: 3
! CHECK-NEXT: Variable: 3
! CHECK-NEXT: Block: 2
! CHECK-NEXT: ComponentDecl: 2
! CHECK-NEXT: ComponentDefStmt: 2
! CHECK-NEXT: ComponentOrFill: 2
! CHECK-NEXT: ContainsStmt: 2
! CHECK-NEXT: DataComponentDefStmt: 2
! CHECK-NEXT: DeclarationTypeSpec::Class: 2
! CHECK-NEXT: DerivedTypeSpec: 2
! CHECK-NEXT: ExecutionPart: 2
! CHECK-NEXT: IntegerTypeSpec: 2
! CHECK-NEXT: IntrinsicTypeSpec::Real: 2
! CHECK-NEXT: ModuleSubprogram: 2
! CHECK-NEXT: TypeBoundProcBinding: 2
! CHECK-NEXT: TypeBoundProcDecl: 2
! CHECK-NEXT: TypeBoundProcedureStmt: 2
! CHECK-NEXT: TypeBoundProcedureStmt::WithoutInterface: 2
! CHECK-NEXT: DerivedTypeDef: 1
! CHECK-NEXT: DerivedTypeStmt: 1
! CHECK-NEXT: EndFunctionStmt: 1
! CHECK-NEXT: EndModuleStmt: 1
! CHECK-NEXT: EndSubroutineStmt: 1
! CHECK-NEXT: EndTypeStmt: 1
! CHECK-NEXT: Expr::Add: 1
! CHECK-NEXT: FunctionStmt: 1
! CHECK-NEXT: FunctionSubprogram: 1
! CHECK-NEXT: ImplicitPartStmt: 1
! CHECK-NEXT: ImplicitStmt: 1
! CHECK-NEXT: Module: 1
! CHECK-NEXT: ModuleStmt: 1
! CHECK-NEXT: ModuleSubprogramPart: 1
! CHECK-NEXT: PrefixSpec: 1
! CHECK-NEXT: Program: 1
! CHECK-NEXT: ProgramUnit: 1
! CHECK-NEXT: SubroutineStmt: 1
! CHECK-NEXT: SubroutineSubprogram: 1
! CHECK-NEXT: TypeBoundProcedurePart: 1
