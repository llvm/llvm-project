// Tests without serialization:
// RUN: %clang_cc1 -std=c++17 -triple spirv64-unknown-unknown -fsycl-is-device \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck %s

// Thes test validates the AST body produced for functions declared with the
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

template<typename T>
auto set_kernel_arg(const T &t) {
  return t;
}

auto set_kernel_arg(EmptySpecial &a) {
  return a.data;
}

template<typename KernelName, typename... Ts>
auto sycl_handle_special_kernel_parameters(Ts...) {
  return [](auto ...Args){ return; };
}

template<typename... Ts>
struct type_list {};

template <typename KernelName, typename... Ts>
auto sycl_kernel_launch(const char *, Ts...) {

    return [&](auto&&... extra_host_args) {
      return type_list<decltype(set_kernel_arg(extra_host_args))...>{};
  };
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
// CHECK-NEXT: | | | |-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (ADL) = 'sycl_kernel_launch' {{.*}}
// CHECK-NEXT: | | | | `-TemplateArgument type 'KN':'type-parameter-0-0'
// CHECK-NEXT: | | | |   `-TemplateTypeParmType {{.*}} 'KN' dependent depth 0 index 0
// CHECK-NEXT: | | | |     `-TemplateTypeParm {{.*}} 'KN'
// CHECK-NEXT: | | | `-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (ADL) = 'sycl_handle_special_kernel_parameters' {{.*}}
// CHECK-NEXT: | | |   `-TemplateArgument type 'KN':'type-parameter-0-0'
// CHECK-NEXT: | | |     `-TemplateTypeParmType {{.*}} 'KN' dependent depth 0 index 0
// CHECK-NEXT: | | |       `-TemplateTypeParm {{.*}} 'KN'
// CHECK-NEXT: | |  `-SYCLKernelEntryPointAttr {{.*}} KN
// CHECK-NEXT: | `-FunctionDecl {{.*}} used k {{.*}} implicit_instantiation instantiated_from {{.*}}
// CHECK-NEXT: |   |-TemplateArgument type 'KN<0>'
// CHECK-NEXT: |   | `-RecordType {{.*}} 'KN<0>' canonical
// CHECK-NEXT: |   |   `-ClassTemplateSpecialization {{.*}} 'KN'
// CHECK-NEXT: |   |-TemplateArgument type '{{.*}}'
// CHECK-NEXT: |   | `-RecordType {{.*}} canonical
// CHECK-NEXT: |   |   `-CXXRecord {{.*}}
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} used Kernel {{.*}}
// CHECK-NEXT: |   |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: |   | |-CompoundStmt {{.*}}
// CHECK-NEXT: |   | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |   | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: |   | |   `-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: |   | |     `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: |   | |-CompoundStmt {{.*}}
// CHECK-NEXT: |   | | `-ExprWithCleanups {{.*}} 'type_list<{{.*}}>'
// CHECK-NEXT: |   | |   `-CXXOperatorCallExpr {{.*}} 'type_list<{{.*}}>' '()'
// CHECK-NEXT: |   | |     |-ImplicitCastExpr {{.*}} 'type_list<{{.*}}> (*)(EmptySpecial &) const' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |     | `-DeclRefExpr {{.*}} 'type_list<{{.*}}> (EmptySpecial &) const' lvalue CXXMethod {{.*}} 'operator()' '{{.*}}'
// CHECK-NEXT: |   | |     |-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: |   | |     | `-MaterializeTemporaryExpr {{.*}} '{{.*}}' lvalue
// CHECK-NEXT: |   | |     |   `-CallExpr {{.*}} '{{.*}}'
// CHECK-NEXT: |   | |     |     |-ImplicitCastExpr {{.*}} '{{.*}}' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |     |     | `-DeclRefExpr {{.*}} '{{.*}}' lvalue Function {{.*}} 'sycl_kernel_launch' {{.*}}
// CHECK-NEXT: |   | |     |     |-ImplicitCastExpr {{.*}} 'const char *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |     |     | `-StringLiteral {{.*}} 'const char[14]' lvalue "_ZTS2KNILi0EE"
// CHECK-NEXT: |   | |     |     `-CXXConstructExpr {{.*}} '{{.*}}' 'void ({{.*}} &&) noexcept'
// CHECK-NEXT: |   | |     |       `-ImplicitCastExpr {{.*}} '{{.*}}' xvalue <NoOp>
// CHECK-NEXT: |   | |     |         `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: |   | |     `-MemberExpr {{.*}} 'EmptySpecial' lvalue .data {{.*}}
// CHECK-NEXT: |   | |       `-MemberExpr {{.*}} 'Wrapper<EmptySpecial>' lvalue . {{.*}}
// CHECK-NEXT: |   | |         `-DeclRefExpr {{.*}} lvalue ParmVar {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: |   | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: |   |   |-ImplicitParamDecl {{.*}} implicit used Kernel {{.*}}
// CHECK-NEXT: |   |   |-ImplicitParamDecl {{.*}} implicit used idk {{.*}}
// CHECK-NEXT: |   |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |   |     |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT: |   |     | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   |     |   |-ImplicitCastExpr {{.*}} 'void (*)(int) const' <FunctionToPointerDecay>
// CHECK-NEXT: |   |     |   | `-DeclRefExpr {{.*}} 'void (int) const' lvalue CXXMethod {{.*}} 'operator()' '{{.*}}'
// CHECK-NEXT: |   |     |   |-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: |   |     |   | `-MaterializeTemporaryExpr {{.*}} '{{.*}}' lvalue
// CHECK-NEXT: |   |     |   |   `-CallExpr {{.*}} '{{.*}}'
// CHECK-NEXT: |   |     |   |     |-ImplicitCastExpr {{.*}} '{{.*}}' <FunctionToPointerDecay>
// CHECK-NEXT: |   |     |   |     | `-DeclRefExpr {{.*}} '{{.*}}' lvalue Function {{.*}} 'sycl_handle_special_kernel_parameters' {{.*}}
// CHECK-NEXT: |   |     |   |     `-CXXConstructExpr {{.*}} 'EmptySpecial' 'void (const EmptySpecial &) noexcept'
// CHECK-NEXT: |   |     |   |       `-ImplicitCastExpr {{.*}} 'const EmptySpecial' lvalue <NoOp>
// CHECK-NEXT: |   |     |   |         `-MemberExpr {{.*}} 'EmptySpecial' lvalue .data {{.*}}
// CHECK-NEXT: |   |     |   |           `-MemberExpr {{.*}} 'Wrapper<EmptySpecial>' lvalue . {{.*}}
// CHECK-NEXT: |   |     |   |             `-DeclRefExpr {{.*}} lvalue ImplicitParam {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: |   |     |   `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     `-DeclRefExpr {{.*}} 'int' lvalue ImplicitParam {{.*}} 'idk' 'int'
// CHECK-NEXT: |   |     `-CompoundStmt {{.*}}
// CHECK-NEXT: |   |       `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   |         |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: |   |         | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: |   |         `-ImplicitCastExpr {{.*}} 'const {{.*}}' lvalue <NoOp>
// CHECK-NEXT: |   |           `-DeclRefExpr {{.*}} lvalue ImplicitParam {{.*}} 'Kernel' {{.*}}
// CHECK-NEXT: |   `-SYCLKernelEntryPointAttr {{.*}} struct KN<0>

void case1() {
    Wrapper<EmptySpecial> KernelArg;
    k<KN<0>>([KernelArg](){});
}
// CHECK: `-FunctionDecl {{.*}} case1 'void ()'
