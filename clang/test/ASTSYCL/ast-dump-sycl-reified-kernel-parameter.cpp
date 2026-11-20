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

// These tests validate the AST produced for functions declared with the
// sycl_kernel_entry_point attribute when one of the function parameters has a
// type declared with the sycl_reified_kernel_parameter attribute.

// CHECK: TranslationUnitDecl {{.*}}

template <typename KI, typename... Ts>
auto sycl_kernel_launch(Ts...) {
  return [](auto&&... special_subobjects) { };
}

template <typename KI, typename... Ts>
auto sycl_reify_kernel_parameters(Ts...) {
  return [](auto&&... special_subobjects) { };
}

template <typename KN, typename KT>
[[clang::sycl_kernel_entry_point(KN)]] void skep(const KT &k) {
  k();
}

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

// A type that requires reification as a kernel parameter.
struct [[clang::sycl_reified_kernel_parameter]] SRKP {};
struct SRKPDerived : SRKP {};
struct SRKPArray {
  SRKP a[2];
};

// A kernel object type with subobjects that require reification as a kernel
// parameter.
struct K : SRKP {
  void operator()() const {}
  SRKP srkp;
  SRKPDerived srkp_derived;
  SRKPArray skrp_array;
};

template void skep<KN<1>>(const K&);

// CHECK:      |-FunctionTemplateDecl {{.*}} skep{{.*}}
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 KN
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 1 KT
// CHECK-NEXT: | |-FunctionDecl {{.*}} skep 'void (const KT &)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} referenced k 'const KT &'
// CHECK-NEXT: | | |-UnresolvedSYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | | `-CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: | | | |  `-DeclRefExpr {{.*}} 'const KT' lvalue ParmVar {{.*}} 'k' 'const KT &'
// CHECK-NEXT: | | | |-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (ADL) = 'sycl_kernel_launch' {{.*}}
// CHECK-NEXT: | | | | `-TemplateArgument type '__sycl_kernel_info<KN>':'__sycl_kernel_info<type-parameter-0-0>'
// CHECK-NEXT: | | | |   `-TemplateSpecializationType {{.*}} '__sycl_kernel_info<KN>' dependent
// CHECK-NEXT: | | | |     |-name: '__sycl_kernel_info'
// CHECK-NEXT: | | | |     | `-ClassTemplateDecl {{.*}} referenced __sycl_kernel_info external-linkage
// CHECK-NEXT: | | | |     `-TemplateArgument type 'KN':'type-parameter-0-0'
// CHECK-NEXT: | | | |       `-TemplateTypeParmType {{.*}} 'KN' dependent depth 0 index 0
// CHECK-NEXT: | | | |         `-TemplateTypeParm {{.*}} 'KN'
// CHECK-NEXT: | | | `-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (ADL) = 'sycl_reify_kernel_parameters' {{.*}}
// CHECK-NEXT: | | |   `-TemplateArgument type '__sycl_kernel_info<KN>':'__sycl_kernel_info<type-parameter-0-0>'
// CHECK-NEXT: | | |     `-TemplateSpecializationType {{.*}} '__sycl_kernel_info<KN>' dependent
// CHECK-NEXT: | | |       |-name: '__sycl_kernel_info'
// CHECK-NEXT: | | |       | `-ClassTemplateDecl {{.*}} referenced __sycl_kernel_info external-linkage
// CHECK-NEXT: | | |       `-TemplateArgument type 'KN':'type-parameter-0-0'
// CHECK-NEXT: | | |         `-TemplateTypeParmType {{.*}} 'KN' dependent depth 0 index 0
// CHECK-NEXT: | | |           `-TemplateTypeParm {{.*}} 'KN'
// CHECK-NEXT: | | `-SYCLKernelEntryPointAttr {{.*}} KN
// CHECK-NEXT: | `-FunctionDecl {{.*}} skep {{.*}} explicit_instantiation_definition instantiated_from {{.*}}
// CHECK-NEXT: |   |-TemplateArgument type 'KN<1>'
// CHECK-NEXT: |   | `-RecordType {{.*}} 'KN<1>' canonical
// CHECK-NEXT: |   |   `-ClassTemplateSpecialization {{.*}} 'KN'
// CHECK-NEXT: |   |-TemplateArgument type '{{.*}}'
// CHECK-NEXT: |   | `-RecordType {{.*}} canonical
// CHECK-NEXT: |   |   `-CXXRecord {{.*}}
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} used k {{.*}}
// CHECK-NEXT: |   |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: |   | |-CompoundStmt {{.*}}
// CHECK-NEXT: |   | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |   | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: |   | |   `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'k' {{.*}}
// CHECK-NEXT: |   | |-CompoundStmt {{.*}}
// CHECK-NEXT: |   | | `-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT: |   | |   `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   | |     |-ImplicitCastExpr {{.*}} 'void (*)(SRKP &, SRKP &, SRKP &) const' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |     | `-DeclRefExpr {{.*}} 'void (SRKP &, SRKP &, SRKP &) const' lvalue CXXMethod {{.*}} 'operator()' 'void (SRKP &, SRKP &, SRKP &) const' {{.*}}
// CHECK-NEXT: |   | |     |-ImplicitCastExpr {{.*}} 'const (lambda at {{.*}})' lvalue <NoOp>
// CHECK-NEXT: |   | |     | `-MaterializeTemporaryExpr {{.*}} '(lambda at {{.*}})' lvalue
// CHECK-NEXT: |   | |     |   `-CallExpr {{.*}} '(lambda at {{.*}})'
// CHECK-NEXT: |   | |     |     |-ImplicitCastExpr {{.*}} '(lambda at {{.*}}) (*)(K)' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |     |     | `-DeclRefExpr {{.*}} '(lambda at {{.*}}) (K)' lvalue Function {{.*}} 'sycl_kernel_launch' {{.*}}
// CHECK-NEXT: |   | |     |     `-CXXConstructExpr {{.*}} 'K' 'void (const K &) noexcept'
// CHECK-NEXT: |   | |     |       `-DeclRefExpr {{.*}} 'const K' lvalue ParmVar {{.*}} 'k' 'const K &'
// CHECK-NEXT: |   | |     |-ImplicitCastExpr {{.*}} 'SRKP' lvalue <DerivedToBase (SRKP)>
// CHECK-NEXT: |   | |     | `-DeclRefExpr {{.*}} 'const K' lvalue ParmVar {{.*}} 'k' 'const K &'
// CHECK-NEXT: |   | |     |-MemberExpr {{.*}} 'SRKP' lvalue .srkp {{.*}}
// CHECK-NEXT: |   | |     | `-DeclRefExpr {{.*}} 'const K' lvalue ParmVar {{.*}} 'k' 'const K &'
// CHECK-NEXT: |   | |     `-ImplicitCastExpr {{.*}} 'SRKP' lvalue <DerivedToBase (SRKP)>
// CHECK-NEXT: |   | |       `-MemberExpr {{.*}} 'SRKPDerived' lvalue .srkp_derived {{.*}}
// CHECK-NEXT: |   | |         `-DeclRefExpr {{.*}} 'const K' lvalue ParmVar {{.*}} 'k' 'const K &'
// CHECK-NEXT: |   | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: |   |   |-ImplicitParamDecl {{.*}} implicit used k 'const K'
// CHECK-NEXT: |   |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |   |     |-CompoundStmt {{.*}}
// CHECK-NEXT: |   |     | `-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT: |   |     |   `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   |     |     |-ImplicitCastExpr {{.*}} 'void (*)(const K &, const SRKP &, const SRKPDerived &) const' <FunctionToPointerDecay>
// CHECK-NEXT: |   |     |     | `-DeclRefExpr {{.*}} 'void (const K &, const SRKP &, const SRKPDerived &) const' lvalue CXXMethod {{.*}} 'operator()' 'void (const K &, const SRKP &, const SRKPDerived &) const' {{.*}}
// CHECK-NEXT: |   |     |     |-ImplicitCastExpr {{.*}} 'const (lambda at {{.*}})' lvalue <NoOp>
// CHECK-NEXT: |   |     |     | `-MaterializeTemporaryExpr {{.*}} '(lambda at {{.*}})' lvalue
// CHECK-NEXT: |   |     |     |   `-CallExpr {{.*}} '(lambda at {{.*}})'
// CHECK-NEXT: |   |     |     |     |-ImplicitCastExpr {{.*}} '(lambda at {{.*}}) (*)(K)' <FunctionToPointerDecay>
// CHECK-NEXT: |   |     |     |     | `-DeclRefExpr {{.*}} '(lambda at {{.*}}) (K)' lvalue Function {{.*}} 'sycl_reify_kernel_parameters' {{.*}}
// CHECK-NEXT: |   |     |     |     `-CXXConstructExpr {{.*}} 'K' 'void (const K &) noexcept'
// CHECK-NEXT: |   |     |     |       `-DeclRefExpr {{.*}} 'const K' lvalue ImplicitParam {{.*}} 'k' 'const K'
// CHECK-NEXT: |   |     |     |-DeclRefExpr {{.*}} 'const K' lvalue ImplicitParam {{.*}} 'k' 'const K'
// CHECK-NEXT: |   |     |     |-MemberExpr {{.*}} 'const SRKP' lvalue .srkp {{.*}}
// CHECK-NEXT: |   |     |     | `-DeclRefExpr {{.*}} 'const K' lvalue ImplicitParam {{.*}} 'k' 'const K'
// CHECK-NEXT: |   |     |     `-MemberExpr {{.*}} 'const SRKPDerived' lvalue .srkp_derived {{.*}}
// CHECK-NEXT: |   |     |       `-DeclRefExpr {{.*}} 'const K' lvalue ImplicitParam {{.*}} 'k' 'const K'
// CHECK-NEXT: |   |     `-CompoundStmt {{.*}}
// CHECK-NEXT: |   |       `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   |         |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: |   |         | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: |   |         `-DeclRefExpr {{.*}} 'const K' lvalue ImplicitParam {{.*}} 'k' 'const K'
// CHECK-NEXT: |   `-SYCLKernelEntryPointAttr {{.*}} struct KN<1>


void the_end() {}
// CHECK:      `-FunctionDecl {{.*}} the_end 'void ()' external-linkage
