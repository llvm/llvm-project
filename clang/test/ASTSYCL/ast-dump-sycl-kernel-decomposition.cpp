// Tests without serialization:
// RUN: %clang_cc1 -std=c++17 -triple spirv64-unknown-unknown -fsycl-is-device \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck %s
//
// Tests with serialization:
// RUN: %clang_cc1 -std=c++17 -triple spirv64-unknown-unknown -fsycl-is-device \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -triple spirv64-unknown-unknown -fsycl-is-device \
// RUN:   -include-pch %t -ast-dump-all /dev/null \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -include-pch %t -ast-dump-all /dev/null \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck %s

// This test validates the AST body produced for functions declared with the
// sycl_kernel_entry_point attribute in case an argument of such function
// contains an object that requires decomposition.

// CHECK: TranslationUnitDecl {{.*}}

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

struct [[clang::sycl_special_kernel_parameter]] EmptySpecial {
  int data;
};

template<typename T>
struct Wrapper {
 T data;
 int *data1;
};

template <typename KernelName, typename... Ts>
auto sycl_kernel_launch(const char *, Ts...) {
    return [](auto&&... special_subobjects) { };
}


template <typename KN, typename KT>
[[clang::sycl_kernel_entry_point(KN)]] void k(KT Kernel) {
  Kernel();
}
// CHECK:      |-FunctionTemplateDecl {{.*}} k{{.*}}
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 KN
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 1 KT
// CHECK-NEXT: | |-FunctionDecl {{.*}} k 'void (KT)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} referenced Kernel 'KT'
// CHECK-NEXT: | | |-UnresolvedSYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | | `-CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: | | | |  `-DeclRefExpr {{.*}} 'KT' lvalue ParmVar {{.*}} 'Kernel' 'KT'
// CHECK-NEXT: | | | `-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (ADL) = 'sycl_kernel_launch' {{.*}}
// CHECK-NEXT: | | |   `-TemplateArgument type 'KN':'type-parameter-0-0'
// CHECK-NEXT: | | |     `-TemplateTypeParmType {{.*}} 'KN' dependent depth 0 index 0
// CHECK-NEXT: | | |       `-TemplateTypeParm {{.*}} 'KN'
// CHECK-NEXT: | |  `-SYCLKernelEntryPointAttr {{.*}} KN
// CHECK-NEXT: | |-FunctionDecl {{.*}} used k {{.*}} implicit_instantiation instantiated_from {{.*}}
// CHECK-NEXT: | | |-TemplateArgument type 'KN<0>'
// CHECK-NEXT: | | | `-RecordType {{.*}} 'KN<0>' canonical
// CHECK-NEXT: | | |   `-ClassTemplateSpecialization {{.*}} 'KN'
// CHECK-NEXT: | | |-TemplateArgument type '{{.*}}'
// CHECK-NEXT: | | | `-RecordType {{.*}} canonical
// CHECK-NEXT: | | |   `-CXXRecord {{.*}}
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} used Kernel {{.*}}
// CHECK-NEXT: | | |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | | | |   |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: | | | |   | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: | | | |   `-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: | | | |     `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: | | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | | `-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT: | | | |   `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | | | |     |-ImplicitCastExpr {{.*}} 'void (*)(EmptySpecial &) const' <FunctionToPointerDecay>
// CHECK-NEXT: | | | |     | `-DeclRefExpr {{.*}} 'void (EmptySpecial &) const' lvalue CXXMethod {{.*}} 'operator()' '{{.*}}'
// CHECK-NEXT: | | | |     |-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: | | | |     | `-MaterializeTemporaryExpr {{.*}} '{{.*}}' lvalue
// CHECK-NEXT: | | | |     |   `-CallExpr {{.*}} '{{.*}}'
// CHECK-NEXT: | | | |     |     |-ImplicitCastExpr {{.*}} '{{.*}}' <FunctionToPointerDecay>
// CHECK-NEXT: | | | |     |     | `-DeclRefExpr {{.*}} '{{.*}}' lvalue Function {{.*}} 'sycl_kernel_launch' {{.*}}
// CHECK-NEXT: | | | |     |     |-ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT: | | | |     |     | `-StringLiteral {{.*}} 'const char[14]' lvalue "_ZTS2KNILi0EE"
// CHECK-NEXT: | | | |     |     `-CXXConstructExpr {{.*}} '{{.*}}' 'void ({{.*}} &&) noexcept'
// CHECK-NEXT: | | | |     |       `-ImplicitCastExpr {{.*}} '{{.*}}' xvalue <NoOp>
// CHECK-NEXT: | | | |     |         `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: | | | |     `-MemberExpr {{.*}} 'EmptySpecial' lvalue .data {{.*}}
// CHECK-NEXT: | | | |       `-MemberExpr {{.*}} 'Wrapper<EmptySpecial>' lvalue . {{.*}}
// CHECK-NEXT: | | | |         `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: | | | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: | | |   |-ImplicitParamDecl {{.*}} implicit used Kernel {{.*}}
// CHECK-NEXT: | | |   `-CompoundStmt {{.*}}
// CHECK-NEXT: | | |     `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | | |       |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: | | |       | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: | | |       `-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: | | |         `-DeclRefExpr {{.*}} lvalue ImplicitParam {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: | | `-SYCLKernelEntryPointAttr {{.*}} struct KN<0>

// Test that a class inheriting from a sycl_special_kernel_parameter type
// is accessed via a DerivedToBase cast.
// The instantiation for KN<1> should produce a SYCLKernelCallStmt where
// the base class SpecialBase is accessed via DerivedToBase cast.
// CHECK:      | `-FunctionDecl {{.*}} used k {{.*}} implicit_instantiation instantiated_from {{.*}}
// CHECK-NEXT: |   |-TemplateArgument type 'KN<1>'
// CHECK:      |   |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: |   | |-CompoundStmt {{.*}}
// CHECK-NEXT: |   | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK:      |   | |-CompoundStmt {{.*}}
// CHECK-NEXT: |   | | `-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT: |   | |   `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   | |     |-ImplicitCastExpr {{.*}} 'void (*)(SpecialBase &) const' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |     | `-DeclRefExpr {{.*}} 'void (SpecialBase &) const' lvalue CXXMethod {{.*}} 'operator()' '{{.*}}'
// CHECK-NEXT: |   | |     |-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: |   | |     | `-MaterializeTemporaryExpr {{.*}} '{{.*}}' lvalue
// CHECK-NEXT: |   | |     |   `-CallExpr {{.*}} '{{.*}}'
// CHECK-NEXT: |   | |     |     |-ImplicitCastExpr {{.*}} '{{.*}}' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |     |     | `-DeclRefExpr {{.*}} '{{.*}}' lvalue Function {{.*}} 'sycl_kernel_launch' {{.*}}
// CHECK-NEXT: |   | |     |     |-ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |     |     | `-StringLiteral {{.*}} 'const char[14]' lvalue "_ZTS2KNILi1EE"
// CHECK-NEXT: |   | |     |     `-CXXConstructExpr {{.*}} '{{.*}}' 'void ({{.*}} &&) noexcept'
// CHECK-NEXT: |   | |     |       `-ImplicitCastExpr {{.*}} '{{.*}}' xvalue <NoOp>
// CHECK-NEXT: |   | |     |         `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: |   | |     `-ImplicitCastExpr {{.*}} 'SpecialBase' lvalue <DerivedToBase (SpecialBase)>
// CHECK-NEXT: |   | |       `-MemberExpr {{.*}} 'DerivedFromSpecial' lvalue . {{.*}}
// CHECK-NEXT: |   | |         `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: |   | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: |   |   |-ImplicitParamDecl {{.*}} implicit used Kernel {{.*}}
// CHECK-NEXT: |   |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |   |     `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: |   |       | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: |   |       `-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: |   |         `-DeclRefExpr {{.*}} lvalue ImplicitParam {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: |   `-SYCLKernelEntryPointAttr {{.*}} struct KN<1>

void case1() {
    Wrapper<EmptySpecial> KernelArg;
    k<KN<0>>([KernelArg](){});
}

struct [[clang::sycl_special_kernel_parameter]] SpecialBase {
  int data;
};

struct DerivedFromSpecial : SpecialBase {
  int extra;
};

void case2() {
    DerivedFromSpecial DFS;
    k<KN<1>>([DFS](){});
}
