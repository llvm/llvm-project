// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers %s -x c++
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter makeWidget -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-OVERLOADS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter broadGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-BROAD %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter coexistGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-COEXIST %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter mismatchGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-MISMATCH %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter aliasGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-ALIAS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter aliasPrecedenceGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-ALIAS-PRECEDENCE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter multiAliasGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-MULTI-ALIAS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter nullableGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-NULLABILITY %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter rawIntGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-RAW-INT %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter constValueGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-CONST %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter volatileValueGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-VOLATILE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter pointerVolatileGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-POINTER-VOLATILE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter pointeeVolatileMismatchGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-POINTEE-VOLATILE-MISMATCH %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter nestedNullableGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-NESTED-NULLABILITY %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter nullableArrayGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-NULLABLE-ARRAY %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter nullableFunctionPointerItselfGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-NULLABLE-FUNCTION-POINTER-ITSELF %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter nullableFunctionPointerGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-NULLABLE-FUNCTION-POINTER %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter nullableNonnullFunctionPointerGlobal -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-NULLABLE-NONNULL-FUNCTION-POINTER %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorNamespace::makeNamespaced -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-NAMESPACE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::setValue -x c++ | FileCheck --check-prefix=CHECK-METHOD-OVERLOADS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::broad -x c++ | FileCheck --check-prefix=CHECK-METHOD-BROAD %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::coexist -x c++ | FileCheck --check-prefix=CHECK-METHOD-COEXIST %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::defaults -x c++ | FileCheck --check-prefix=CHECK-METHOD-DEFAULTS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::configure -x c++ | FileCheck --check-prefix=CHECK-METHOD-STATIC %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::mismatch -x c++ | FileCheck --check-prefix=CHECK-METHOD-MISMATCH %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::alias -x c++ | FileCheck --check-prefix=CHECK-METHOD-ALIAS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::aliasPrecedence -x c++ | FileCheck --check-prefix=CHECK-METHOD-ALIAS-PRECEDENCE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::multiAlias -x c++ | FileCheck --check-prefix=CHECK-METHOD-MULTI-ALIAS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::nullable -x c++ | FileCheck --check-prefix=CHECK-METHOD-NULLABILITY %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::rawInt -x c++ | FileCheck --check-prefix=CHECK-METHOD-RAW-INT %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::constValue -x c++ | FileCheck --check-prefix=CHECK-METHOD-CONST %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersSema -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter SelectorWidget::operator+ -x c++ | FileCheck --check-prefix=CHECK-METHOD-OPERATOR %s

#include "WhereParametersSema.h"

// CHECK-GLOBAL-VOLATILE: FunctionDecl {{.+}} volatileValueGlobal 'void (volatile int)'
// CHECK-GLOBAL-VOLATILE: SwiftNameAttr {{.+}} "volatileValueGlobal(_:)"

// CHECK-GLOBAL-POINTER-VOLATILE: FunctionDecl {{.+}} pointerVolatileGlobal 'void (int *volatile)'
// CHECK-GLOBAL-POINTER-VOLATILE: SwiftNameAttr {{.+}} "pointerVolatileGlobal(_:)"

// CHECK-GLOBAL-POINTEE-VOLATILE-MISMATCH: FunctionDecl {{.+}} pointeeVolatileMismatchGlobal 'void (volatile int *)'
// CHECK-GLOBAL-POINTEE-VOLATILE-MISMATCH-NOT: SwiftNameAttr

// CHECK-GLOBAL-NESTED-NULLABILITY: FunctionDecl {{.+}} nestedNullableGlobal 'void (int * _Nullable * _Nullable)'
// CHECK-GLOBAL-NESTED-NULLABILITY: SwiftNameAttr {{.+}} "nestedNullableGlobal(_:)"

// CHECK-GLOBAL-NULLABLE-ARRAY: FunctionDecl {{.+}} nullableArrayGlobal 'void (int * _Nullable *)'
// CHECK-GLOBAL-NULLABLE-ARRAY: SwiftNameAttr {{.+}} "nullableArrayGlobal(_:)"

// CHECK-GLOBAL-NULLABLE-FUNCTION-POINTER-ITSELF: FunctionDecl {{.+}} nullableFunctionPointerItselfGlobal 'void (void (* _Nullable)(int *))'
// CHECK-GLOBAL-NULLABLE-FUNCTION-POINTER-ITSELF: SwiftNameAttr {{.+}} "nullableFunctionPointerItselfGlobal(_:)"

// CHECK-GLOBAL-NULLABLE-FUNCTION-POINTER: FunctionDecl {{.+}} nullableFunctionPointerGlobal 'void (void (*)(int * _Nullable))'
// CHECK-GLOBAL-NULLABLE-FUNCTION-POINTER: SwiftNameAttr {{.+}} "nullableFunctionPointerGlobal(_:)"

// CHECK-GLOBAL-NULLABLE-NONNULL-FUNCTION-POINTER: FunctionDecl {{.+}} nullableNonnullFunctionPointerGlobal 'void (void (*)(int * _Nullable, int * _Nonnull))'
// CHECK-GLOBAL-NULLABLE-NONNULL-FUNCTION-POINTER: SwiftNameAttr {{.+}} "nullableNonnullFunctionPointerGlobal(_:)"

// CHECK-GLOBAL-OVERLOADS: FunctionDecl {{.+}} makeWidget 'void (int)'
// CHECK-GLOBAL-OVERLOADS-NEXT: ParmVarDecl {{.+}} 'int'
// CHECK-GLOBAL-OVERLOADS-NEXT: SwiftNameAttr {{.+}} "makeIntWidget(_:)"
// CHECK-GLOBAL-OVERLOADS: FunctionDecl {{.+}} makeWidget 'void (double)'
// CHECK-GLOBAL-OVERLOADS-NEXT: ParmVarDecl {{.+}} 'double'
// CHECK-GLOBAL-OVERLOADS-NEXT: SwiftNameAttr {{.+}} "makeDoubleWidget(_:)"
// CHECK-GLOBAL-OVERLOADS: FunctionDecl {{.+}} makeWidget 'void ()'
// CHECK-GLOBAL-OVERLOADS-NEXT: SwiftNameAttr {{.+}} "makeCurrentWidget()"

// CHECK-GLOBAL-BROAD: FunctionDecl {{.+}} broadGlobal 'void (int)'
// CHECK-GLOBAL-BROAD: SwiftPrivateAttr
// CHECK-GLOBAL-BROAD: FunctionDecl {{.+}} broadGlobal 'void (double)'
// CHECK-GLOBAL-BROAD: SwiftPrivateAttr

// CHECK-GLOBAL-COEXIST: FunctionDecl {{.+}} coexistGlobal 'void (int)'
// CHECK-GLOBAL-COEXIST: SwiftPrivateAttr
// CHECK-GLOBAL-COEXIST: SwiftNameAttr {{.+}} "coexistGlobalInt(_:)"
// CHECK-GLOBAL-COEXIST: FunctionDecl {{.+}} coexistGlobal 'void (double)'
// CHECK-GLOBAL-COEXIST: SwiftPrivateAttr
// CHECK-GLOBAL-COEXIST-NOT: SwiftNameAttr

// CHECK-GLOBAL-MISMATCH: FunctionDecl {{.+}} mismatchGlobal 'void (float)'
// CHECK-GLOBAL-MISMATCH-NOT: SwiftNameAttr

// CHECK-GLOBAL-ALIAS: FunctionDecl {{.+}} aliasGlobal 'void (AliasInt)'
// CHECK-GLOBAL-ALIAS: SwiftNameAttr {{.+}} "aliasGlobal(_:)"

// CHECK-GLOBAL-ALIAS-PRECEDENCE: FunctionDecl {{.+}} aliasPrecedenceGlobal 'void (AliasInt)'
// CHECK-GLOBAL-ALIAS-PRECEDENCE-NOT: fallbackAliasPrecedenceGlobal
// CHECK-GLOBAL-ALIAS-PRECEDENCE: SwiftNameAttr {{.+}} "aliasPrecedenceGlobal(_:)"

// CHECK-GLOBAL-MULTI-ALIAS: FunctionDecl {{.+}} multiAliasGlobal 'void (DeepAliasInt)'
// CHECK-GLOBAL-MULTI-ALIAS: SwiftNameAttr {{.+}} "multiAliasGlobal(_:)"

// CHECK-GLOBAL-NULLABILITY: FunctionDecl {{.+}} nullableGlobal 'void (char * _Nonnull)'
// CHECK-GLOBAL-NULLABILITY: SwiftNameAttr {{.+}} "nullableGlobal(_:)"

// CHECK-GLOBAL-RAW-INT: FunctionDecl {{.+}} rawIntGlobal 'void (int)'
// CHECK-GLOBAL-RAW-INT: SwiftNameAttr {{.+}} "rawIntGlobal(_:)"

// CHECK-GLOBAL-CONST: FunctionDecl {{.+}} constValueGlobal 'void (const int)'
// CHECK-GLOBAL-CONST: SwiftNameAttr {{.+}} "constValueGlobal(_:)"

// CHECK-GLOBAL-NAMESPACE: FunctionDecl {{.+}} makeNamespaced 'void (int)'
// CHECK-GLOBAL-NAMESPACE-NEXT: ParmVarDecl {{.+}} 'int'
// CHECK-GLOBAL-NAMESPACE-NEXT: SwiftNameAttr {{.+}} "makeNamespacedInt(_:)"
// CHECK-GLOBAL-NAMESPACE: FunctionDecl {{.+}} makeNamespaced 'void (double)'
// CHECK-GLOBAL-NAMESPACE-NEXT: ParmVarDecl {{.+}} 'double'
// CHECK-GLOBAL-NAMESPACE-NEXT: SwiftNameAttr {{.+}} "makeNamespacedDouble(_:)"

// CHECK-METHOD-OVERLOADS: CXXMethodDecl {{.+}} setValue 'void (int)'
// CHECK-METHOD-OVERLOADS-NEXT: ParmVarDecl {{.+}} 'int'
// CHECK-METHOD-OVERLOADS-NEXT: SwiftNameAttr {{.+}} "setIntValue(_:)"
// CHECK-METHOD-OVERLOADS: CXXMethodDecl {{.+}} setValue 'void (double)'
// CHECK-METHOD-OVERLOADS-NEXT: ParmVarDecl {{.+}} 'double'
// CHECK-METHOD-OVERLOADS-NEXT: SwiftNameAttr {{.+}} "setDoubleValue(_:)"
// CHECK-METHOD-OVERLOADS: CXXMethodDecl {{.+}} setValue 'void ()'
// CHECK-METHOD-OVERLOADS-NEXT: SwiftNameAttr {{.+}} "currentValue()"

// CHECK-METHOD-BROAD: CXXMethodDecl {{.+}} broad 'void (int)'
// CHECK-METHOD-BROAD: SwiftPrivateAttr
// CHECK-METHOD-BROAD: CXXMethodDecl {{.+}} broad 'void (double)'
// CHECK-METHOD-BROAD: SwiftPrivateAttr

// CHECK-METHOD-COEXIST: CXXMethodDecl {{.+}} coexist 'void (int)'
// CHECK-METHOD-COEXIST: SwiftPrivateAttr
// CHECK-METHOD-COEXIST: SwiftNameAttr {{.+}} "coexistInt(_:)"
// CHECK-METHOD-COEXIST: CXXMethodDecl {{.+}} coexist 'void (double)'
// CHECK-METHOD-COEXIST: SwiftPrivateAttr
// CHECK-METHOD-COEXIST-NOT: SwiftNameAttr

// CHECK-METHOD-DEFAULTS: CXXMethodDecl {{.+}} defaults 'void (int, double)'
// CHECK-METHOD-DEFAULTS: SwiftNameAttr {{.+}} "defaultsWithTwoParameters(_:_:)"
// CHECK-METHOD-DEFAULTS: CXXMethodDecl {{.+}} defaults 'void (int)'
// CHECK-METHOD-DEFAULTS-NOT: SwiftNameAttr

// CHECK-METHOD-STATIC: CXXMethodDecl {{.+}} configure 'void (int)' static
// CHECK-METHOD-STATIC: SwiftNameAttr {{.+}} "configureInt(_:)"

// CHECK-METHOD-MISMATCH: CXXMethodDecl {{.+}} mismatch 'void (float)'
// CHECK-METHOD-MISMATCH-NOT: SwiftNameAttr

// CHECK-METHOD-ALIAS: CXXMethodDecl {{.+}} alias 'void (AliasInt)'
// CHECK-METHOD-ALIAS: SwiftNameAttr {{.+}} "alias(_:)"

// CHECK-METHOD-ALIAS-PRECEDENCE: CXXMethodDecl {{.+}} aliasPrecedence 'void (AliasInt)'
// CHECK-METHOD-ALIAS-PRECEDENCE-NOT: fallbackAliasPrecedence
// CHECK-METHOD-ALIAS-PRECEDENCE: SwiftNameAttr {{.+}} "aliasPrecedence(_:)"

// CHECK-METHOD-MULTI-ALIAS: CXXMethodDecl {{.+}} multiAlias 'void (DeepAliasInt)'
// CHECK-METHOD-MULTI-ALIAS: SwiftNameAttr {{.+}} "multiAlias(_:)"

// CHECK-METHOD-NULLABILITY: CXXMethodDecl {{.+}} nullable 'void (char * _Nonnull)'
// CHECK-METHOD-NULLABILITY: SwiftNameAttr {{.+}} "nullable(_:)"

// CHECK-METHOD-RAW-INT: CXXMethodDecl {{.+}} rawInt 'void (int)'
// CHECK-METHOD-RAW-INT: SwiftNameAttr {{.+}} "rawInt(_:)"

// CHECK-METHOD-CONST: CXXMethodDecl {{.+}} constValue 'void (const int)'
// CHECK-METHOD-CONST: SwiftNameAttr {{.+}} "constValue(_:)"

// CHECK-METHOD-OPERATOR: CXXMethodDecl {{.+}} operator+ 'SelectorWidget (int)'
// CHECK-METHOD-OPERATOR-NEXT: ParmVarDecl {{.+}} 'int'
// CHECK-METHOD-OPERATOR-NEXT: SwiftNameAttr {{.+}} "plusInt(_:)"
// CHECK-METHOD-OPERATOR: CXXMethodDecl {{.+}} operator+ 'SelectorWidget (double)'
// CHECK-METHOD-OPERATOR-NEXT: ParmVarDecl {{.+}} 'double'
// CHECK-METHOD-OPERATOR-NEXT: SwiftNameAttr {{.+}} "plusDouble(_:)"
