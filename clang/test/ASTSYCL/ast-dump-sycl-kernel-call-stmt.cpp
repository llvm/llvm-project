// Tests without serialization:
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-device \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck --match-full-lines %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck --match-full-lines %s
//
// Tests with serialization:
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-device \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-device \
// RUN:   -include-pch %t -ast-dump-all /dev/null \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck --match-full-lines %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -include-pch %t -ast-dump-all /dev/null \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck --match-full-lines %s

// These tests validate the AST body produced for functions declared with the
// sycl_kernel_entry_point attribute.

// CHECK: TranslationUnitDecl {{.*}}

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

// A unique invocable type for use with each declared kernel entry point.
template<int> struct K {
  template<typename... Ts>
  void operator()(Ts...) const {}
};


[[clang::sycl_kernel_entry_point(KN<1>)]]
void skep1() {
}
// CHECK:      |-FunctionDecl {{.*}} skep1 'void ()'
// CHECK-NEXT: | |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: | |   `-CompoundStmt {{.*}}
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<1>

template<typename KNT, typename KT>
[[clang::sycl_kernel_entry_point(KNT)]]
void skep2(KT k) {
  k();
}
template
void skep2<KN<2>>(K<2>);
// CHECK:      |-FunctionTemplateDecl {{.*}} skep2
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} KNT
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} KT
// CHECK-NEXT: | |-FunctionDecl {{.*}} skep2 'void (KT)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} k 'KT'
// CHECK-NEXT: | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | `-CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: | | |   `-DeclRefExpr {{.*}} 'KT' lvalue ParmVar {{.*}} 'k' 'KT'
// CHECK-NEXT: | | `-SYCLKernelEntryPointAttr {{.*}} KNT

// CHECK-NEXT: | `-FunctionDecl {{.*}} skep2 'void (K<2>)' explicit_instantiation_definition
// CHECK-NEXT: |   |-TemplateArgument type 'KN<2>'
// CHECK-NEXT: |   | `-RecordType {{.*}} 'KN<2>' canonical
// CHECK-NEXT: |   |   `-ClassTemplateSpecialization {{.*}} 'KN'
// CHECK-NEXT: |   |-TemplateArgument type 'K<2>'
// CHECK-NEXT: |   | `-RecordType {{.*}} 'K<2>' canonical
// CHECK-NEXT: |   |   `-ClassTemplateSpecialization {{.*}} 'K'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} k 'K<2>'
// CHECK-NEXT: |   |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: |   | |-CompoundStmt {{.*}}
// CHECK-NEXT: |   | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |   | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: |   | |   `-ImplicitCastExpr {{.*}} 'const K<2>' lvalue <NoOp>
// CHECK-NEXT: |   | |     `-DeclRefExpr {{.*}} 'K<2>' lvalue ParmVar {{.*}} 'k' 'K<2>'
// CHECK-NEXT: |   | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: |   |   |-ImplicitParamDecl {{.*}} implicit used k 'K<2>'
// CHECK-NEXT: |   |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |   |     `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: |   |       | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: |   |       `-ImplicitCastExpr {{.*}} 'const K<2>' lvalue <NoOp>
// CHECK-NEXT: |   |         `-DeclRefExpr {{.*}} 'K<2>' lvalue ImplicitParam {{.*}} 'k' 'K<2>'
// CHECK-NEXT: |   `-SYCLKernelEntryPointAttr {{.*}} KN<2>

template<typename KNT, typename KT>
[[clang::sycl_kernel_entry_point(KNT)]]
void skep3(KT k) {
  k();
}
template<>
[[clang::sycl_kernel_entry_point(KN<3>)]]
void skep3<KN<3>>(K<3> k) {
  k();
}
// CHECK:      |-FunctionTemplateDecl {{.*}} skep3
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} KNT
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} KT
// CHECK-NEXT: | |-FunctionDecl {{.*}} skep3 'void (KT)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} k 'KT'
// CHECK-NEXT: | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | `-CallExpr {{.*}} '<dependent type>'
// CHECK-NEXT: | | |   `-DeclRefExpr {{.*}} 'KT' lvalue ParmVar {{.*}} 'k' 'KT'
// CHECK-NEXT: | | `-SYCLKernelEntryPointAttr {{.*}} KNT

// CHECK-NEXT: | `-Function {{.*}} 'skep3' 'void (K<3>)'
// CHECK-NEXT: |-FunctionDecl {{.*}} skep3 'void (K<3>)' explicit_specialization
// CHECK-NEXT: | |-TemplateArgument type 'KN<3>'
// CHECK-NEXT: | | `-RecordType {{.*}} 'KN<3>' canonical
// CHECK-NEXT: | |   `-ClassTemplateSpecialization {{.*}} 'KN'
// CHECK-NEXT: | |-TemplateArgument type 'K<3>'
// CHECK-NEXT: | | `-RecordType {{.*}} 'K<3>' canonical
// CHECK-NEXT: | |   `-ClassTemplateSpecialization {{.*}} 'K'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} k 'K<3>'
// CHECK-NEXT: | |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | | |   |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: | | |   | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: | | |   `-ImplicitCastExpr {{.*}} 'const K<3>' lvalue <NoOp>
// CHECK-NEXT: | | |     `-DeclRefExpr {{.*}} 'K<3>' lvalue ParmVar {{.*}} 'k' 'K<3>'
// CHECK-NEXT: | | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit used k 'K<3>'
// CHECK-NEXT: | |   `-CompoundStmt {{.*}}
// CHECK-NEXT: | |     `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: | |       `-ImplicitCastExpr {{.*}} 'const K<3>' lvalue <NoOp>
// CHECK-NEXT: | |         `-DeclRefExpr {{.*}} 'K<3>' lvalue ImplicitParam {{.*}} 'k' 'K<3>'
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<3>

[[clang::sycl_kernel_entry_point(KN<4>)]]
void skep4(K<4> k, int p1, int p2) {
  k(p1, p2);
}
// CHECK:      |-FunctionDecl {{.*}} skep4 'void (K<4>, int, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} k 'K<4>'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} p1 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} p2 'int'
// CHECK-NEXT: | |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | | |   |-ImplicitCastExpr {{.*}} 'void (*)(int, int) const' <FunctionToPointerDecay>
// CHECK-NEXT: | | |   | `-DeclRefExpr {{.*}} 'void (int, int) const' lvalue CXXMethod {{.*}} 'operator()' 'void (int, int) const'
// CHECK-NEXT: | | |   |-ImplicitCastExpr {{.*}} 'const K<4>' lvalue <NoOp>
// CHECK-NEXT: | | |   | `-DeclRefExpr {{.*}} 'K<4>' lvalue ParmVar {{.*}} 'k' 'K<4>'
// CHECK-NEXT: | | |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: | | |   | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'p1' 'int'
// CHECK-NEXT: | | |   `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: | | |     `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'p2' 'int'
// CHECK-NEXT: | | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit used k 'K<4>'
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit used p1 'int'
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit used p2 'int'
// CHECK-NEXT: | |   `-CompoundStmt {{.*}}
// CHECK-NEXT: | |     `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'void (*)(int, int) const' <FunctionToPointerDecay>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'void (int, int) const' lvalue CXXMethod {{.*}} 'operator()' 'void (int, int) const'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'const K<4>' lvalue <NoOp>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'K<4>' lvalue ImplicitParam {{.*}} 'k' 'K<4>'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'int' lvalue ImplicitParam {{.*}} 'p1' 'int'
// CHECK-NEXT: | |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: | |         `-DeclRefExpr {{.*}} 'int' lvalue ImplicitParam {{.*}} 'p2' 'int'
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<4>

[[clang::sycl_kernel_entry_point(KN<5>)]]
void skep5(int unused1, K<5> k, int unused2, int p, int unused3) {
  static int slv = 0;
  int lv = 4;
  k(slv, 1, p, 3, lv, 5, []{ return 6; });
}
// CHECK:      |-FunctionDecl {{.*}} skep5 'void (int, K<5>, int, int, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} unused1 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used k 'K<5>'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} unused2 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used p 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} unused3 'int'
// CHECK-NEXT: | |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | |-CompoundStmt {{.*}}
// CHECK:      | | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit unused1 'int'
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit used k 'K<5>'
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit unused2 'int'
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit used p 'int'
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit unused3 'int'
// CHECK-NEXT: | |   `-CompoundStmt {{.*}}
// CHECK-NEXT: | |     |-DeclStmt {{.*}}
// CHECK-NEXT: | |     | `-VarDecl {{.*}} used slv 'int' static cinit
// CHECK-NEXT: | |     |   `-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: | |     |-DeclStmt {{.*}}
// CHECK-NEXT: | |     | `-VarDecl {{.*}} used lv 'int' cinit
// CHECK-NEXT: | |     |   `-IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: | |     `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'void (*)(int, int, int, int, int, int, (lambda {{.*}}) const' <FunctionToPointerDecay>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'void (int, int, int, int, int, int, (lambda {{.*}})) const' lvalue CXXMethod {{.*}} 'operator()' 'void (int, int, int, int, int, int, (lambda {{.*}})) const'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'const K<5>' lvalue <NoOp>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'K<5>' lvalue ImplicitParam {{.*}} 'k' 'K<5>'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'slv' 'int'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'int' lvalue ImplicitParam {{.*}} 'p' 'int'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'lv' 'int'
// CHECK-NEXT: | |       |-IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT: | |       `-LambdaExpr {{.*}} '(lambda {{.*}})'
// CHECK:      | `-SYCLKernelEntryPointAttr {{.*}} KN<5>

struct S6 {
  void operator()() const;
};
[[clang::sycl_kernel_entry_point(KN<6>)]]
void skep6(const S6 &k) {
  k();
}
// CHECK:      |-FunctionDecl {{.*}} skep6 'void (const S6 &)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used k 'const S6 &'
// CHECK-NEXT: | |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | | |   |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: | | |   | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: | | |   `-DeclRefExpr {{.*}} 'const S6' lvalue ParmVar {{.*}} 'k' 'const S6 &'
// CHECK-NEXT: | | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit used k 'const S6 &'
// CHECK-NEXT: | |   `-CompoundStmt {{.*}}
// CHECK-NEXT: | |     `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: | |       `-DeclRefExpr {{.*}} 'const S6' lvalue ImplicitParam {{.*}} 'k' 'const S6 &'
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<6>

// Parameter types are not required to be complete at the point of a
// non-defining declaration.
struct S7;
[[clang::sycl_kernel_entry_point(KN<7>)]]
void skep7(S7 k);
struct S7 {
  void operator()() const;
};
[[clang::sycl_kernel_entry_point(KN<7>)]]
void skep7(S7 k) {
  k();
}
// CHECK:      |-FunctionDecl {{.*}} skep7 'void (S7)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} k 'S7'
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<7>
// CHECK:      |-FunctionDecl {{.*}} prev {{.*}} skep7 'void (S7)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used k 'S7'
// CHECK-NEXT: | |-SYCLKernelCallStmt {{.*}}
// CHECK-NEXT: | | |-CompoundStmt {{.*}}
// CHECK-NEXT: | | | `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | | |   |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: | | |   | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: | | |   `-ImplicitCastExpr {{.*}} 'const S7' lvalue <NoOp>
// CHECK-NEXT: | | |     `-DeclRefExpr {{.*}} 'S7' lvalue ParmVar {{.*}} 'k' 'S7'
// CHECK-NEXT: | | `-OutlinedFunctionDecl {{.*}}
// CHECK-NEXT: | |   |-ImplicitParamDecl {{.*}} implicit used k 'S7'
// CHECK-NEXT: | |   `-CompoundStmt {{.*}}
// CHECK-NEXT: | |     `-CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} 'void (*)() const' <FunctionToPointerDecay>
// CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod {{.*}} 'operator()' 'void () const'
// CHECK-NEXT: | |       `-ImplicitCastExpr {{.*}} 'const S7' lvalue <NoOp>
// CHECK-NEXT: | |         `-DeclRefExpr {{.*}} 'S7' lvalue ImplicitParam {{.*}} 'k' 'S7'
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<7>


void the_end() {}
// CHECK:      `-FunctionDecl {{.*}} the_end 'void ()'
