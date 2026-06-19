// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -fsyntax-only -I %S/Inputs/Headers %s -x c++
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedEmpty -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-EMPTY %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedDefaults -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-DEFAULTS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedWhitespace -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-WHITESPACE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedUnsigned -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-UNSIGNED %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedTemplateSpacing -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-TEMPLATE-SPACING %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedPointerSpacing -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-POINTER-SPACING %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedRValueReferenceSpacing -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-RVALUE-REFERENCE-SPACING %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedConstValue -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-CONST-VALUE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedConstSpelling -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-CONST-SPELLING %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedConstSuffixSpelling -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-CONST-SUFFIX-SPELLING %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedPointerConst -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-POINTER-CONST %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedPointeeConst -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-POINTEE-CONST %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedPointeeConstMismatch -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-POINTEE-CONST-MISMATCH %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedAlias -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-ALIAS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedDeepAlias -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-DEEP-ALIAS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedDeepAliasSource -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-DEEP-ALIAS-SOURCE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedIntermediateAliasMismatch -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-INTERMEDIATE-ALIAS-MISMATCH %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedConstAlias -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-CONST-ALIAS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedNullable -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-NULLABLE %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter normalizedRawInt -x c++ | FileCheck --check-prefix=CHECK-GLOBAL-RAW-INT %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter NormalizationWidget::empty -x c++ | FileCheck --check-prefix=CHECK-METHOD-EMPTY %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter NormalizationWidget::defaults -x c++ | FileCheck --check-prefix=CHECK-METHOD-DEFAULTS %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter NormalizationWidget::configure -x c++ | FileCheck --check-prefix=CHECK-METHOD-STATIC %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter NormalizationWidget::pointerSpacing -x c++ | FileCheck --check-prefix=CHECK-METHOD-POINTER-SPACING %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter NormalizationWidget::pointeeConstMismatch -x c++ | FileCheck --check-prefix=CHECK-METHOD-POINTEE-CONST-MISMATCH %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/WhereParametersNormalization -fdisable-module-hash -fapinotes-modules -Wno-apinotes -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter NormalizationWidget::deepAlias -x c++ | FileCheck --check-prefix=CHECK-METHOD-DEEP-ALIAS %s

#include "WhereParametersNormalization.h"

// CHECK-GLOBAL-EMPTY: FunctionDecl {{.+}} normalizedEmpty 'void ()'
// CHECK-GLOBAL-EMPTY-NEXT: SwiftNameAttr {{.+}} "normalizedEmpty()"
// CHECK-GLOBAL-EMPTY: FunctionDecl {{.+}} normalizedEmpty 'void (int)'
// CHECK-GLOBAL-EMPTY-NOT: SwiftNameAttr

// CHECK-GLOBAL-DEFAULTS: FunctionDecl {{.+}} normalizedDefaults 'void (int, double)'
// CHECK-GLOBAL-DEFAULTS: SwiftNameAttr {{.+}} "normalizedDefaults(_:_:)"
// CHECK-GLOBAL-DEFAULTS: FunctionDecl {{.+}} normalizedDefaults 'void (int)'
// CHECK-GLOBAL-DEFAULTS-NOT: SwiftNameAttr

// CHECK-GLOBAL-WHITESPACE: FunctionDecl {{.+}} normalizedWhitespace 'void (unsigned int)'
// CHECK-GLOBAL-WHITESPACE: SwiftNameAttr {{.+}} "normalizedWhitespace(_:)"

// CHECK-GLOBAL-UNSIGNED: FunctionDecl {{.+}} normalizedUnsigned 'void (unsigned int)'
// CHECK-GLOBAL-UNSIGNED: SwiftNameAttr {{.+}} "normalizedUnsigned(_:)"

// CHECK-GLOBAL-TEMPLATE-SPACING: FunctionDecl {{.+}} normalizedTemplateSpacing 'void (NormalizationBox<int, double>)'
// CHECK-GLOBAL-TEMPLATE-SPACING: SwiftNameAttr {{.+}} "normalizedTemplateSpacing(_:)"

// CHECK-GLOBAL-POINTER-SPACING: FunctionDecl {{.+}} normalizedPointerSpacing 'void (int *)'
// CHECK-GLOBAL-POINTER-SPACING: SwiftNameAttr {{.+}} "normalizedPointerSpacing(_:)"

// CHECK-GLOBAL-RVALUE-REFERENCE-SPACING: FunctionDecl {{.+}} normalizedRValueReferenceSpacing 'void (int &&)'
// CHECK-GLOBAL-RVALUE-REFERENCE-SPACING: SwiftNameAttr {{.+}} "normalizedRValueReferenceSpacing(_:)"

// CHECK-GLOBAL-CONST-VALUE: FunctionDecl {{.+}} normalizedConstValue 'void (const int)'
// CHECK-GLOBAL-CONST-VALUE: SwiftNameAttr {{.+}} "normalizedConstValue(_:)"

// CHECK-GLOBAL-CONST-SPELLING: FunctionDecl {{.+}} normalizedConstSpelling 'void (int)'
// CHECK-GLOBAL-CONST-SPELLING: SwiftNameAttr {{.+}} "normalizedConstSpelling(_:)"

// CHECK-GLOBAL-CONST-SUFFIX-SPELLING: FunctionDecl {{.+}} normalizedConstSuffixSpelling 'void (int)'
// CHECK-GLOBAL-CONST-SUFFIX-SPELLING: SwiftNameAttr {{.+}} "normalizedConstSuffixSpelling(_:)"

// CHECK-GLOBAL-POINTER-CONST: FunctionDecl {{.+}} normalizedPointerConst 'void (int *const)'
// CHECK-GLOBAL-POINTER-CONST: SwiftNameAttr {{.+}} "normalizedPointerConst(_:)"

// CHECK-GLOBAL-POINTEE-CONST: FunctionDecl {{.+}} normalizedPointeeConst 'void (const int *)'
// CHECK-GLOBAL-POINTEE-CONST: SwiftNameAttr {{.+}} "normalizedPointeeConst(_:)"

// CHECK-GLOBAL-POINTEE-CONST-MISMATCH: FunctionDecl {{.+}} normalizedPointeeConstMismatch 'void (const int *)'
// CHECK-GLOBAL-POINTEE-CONST-MISMATCH-NOT: SwiftNameAttr

// CHECK-GLOBAL-ALIAS: FunctionDecl {{.+}} normalizedAlias 'void (NormalizationAliasInt)'
// CHECK-GLOBAL-ALIAS: SwiftNameAttr {{.+}} "normalizedAlias(_:)"

// CHECK-GLOBAL-DEEP-ALIAS: FunctionDecl {{.+}} normalizedDeepAlias 'void (NormalizationDeepAliasInt)'
// CHECK-GLOBAL-DEEP-ALIAS: SwiftNameAttr {{.+}} "normalizedDeepAlias(_:)"

// CHECK-GLOBAL-DEEP-ALIAS-SOURCE: FunctionDecl {{.+}} normalizedDeepAliasSource 'void (NormalizationDeepAliasInt)'
// CHECK-GLOBAL-DEEP-ALIAS-SOURCE: SwiftNameAttr {{.+}} "normalizedDeepAliasSource(_:)"

// CHECK-GLOBAL-INTERMEDIATE-ALIAS-MISMATCH: FunctionDecl {{.+}} normalizedIntermediateAliasMismatch 'void (NormalizationDeepAliasInt)'
// CHECK-GLOBAL-INTERMEDIATE-ALIAS-MISMATCH-NOT: SwiftNameAttr

// CHECK-GLOBAL-CONST-ALIAS: FunctionDecl {{.+}} normalizedConstAlias 'void (NormalizationConstAliasInt)'
// CHECK-GLOBAL-CONST-ALIAS: SwiftNameAttr {{.+}} "normalizedConstAlias(_:)"

// CHECK-GLOBAL-NULLABLE: FunctionDecl {{.+}} normalizedNullable 'void (char * _Nullable)'
// CHECK-GLOBAL-NULLABLE: SwiftNameAttr {{.+}} "normalizedNullable(_:)"

// CHECK-GLOBAL-RAW-INT: FunctionDecl {{.+}} normalizedRawInt 'void (int)'
// CHECK-GLOBAL-RAW-INT: SwiftNameAttr {{.+}} "normalizedRawInt(_:)"

// CHECK-METHOD-EMPTY: CXXMethodDecl {{.+}} empty 'void ()'
// CHECK-METHOD-EMPTY-NEXT: SwiftNameAttr {{.+}} "empty()"
// CHECK-METHOD-EMPTY: CXXMethodDecl {{.+}} empty 'void (int)'
// CHECK-METHOD-EMPTY-NOT: SwiftNameAttr

// CHECK-METHOD-DEFAULTS: CXXMethodDecl {{.+}} defaults 'void (int, double)'
// CHECK-METHOD-DEFAULTS: SwiftNameAttr {{.+}} "defaults(_:_:)"
// CHECK-METHOD-DEFAULTS: CXXMethodDecl {{.+}} defaults 'void (int)'
// CHECK-METHOD-DEFAULTS-NOT: SwiftNameAttr

// CHECK-METHOD-STATIC: CXXMethodDecl {{.+}} configure 'void (int)' static
// CHECK-METHOD-STATIC: SwiftNameAttr {{.+}} "configure(_:)"

// CHECK-METHOD-POINTER-SPACING: CXXMethodDecl {{.+}} pointerSpacing 'void (int *)'
// CHECK-METHOD-POINTER-SPACING: SwiftNameAttr {{.+}} "pointerSpacing(_:)"

// CHECK-METHOD-POINTEE-CONST-MISMATCH: CXXMethodDecl {{.+}} pointeeConstMismatch 'void (const int *)'
// CHECK-METHOD-POINTEE-CONST-MISMATCH-NOT: SwiftNameAttr

// CHECK-METHOD-DEEP-ALIAS: CXXMethodDecl {{.+}} deepAlias 'void (NormalizationDeepAliasInt)'
// CHECK-METHOD-DEEP-ALIAS: SwiftNameAttr {{.+}} "deepAlias(_:)"
