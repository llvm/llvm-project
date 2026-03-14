// RUN: %clang_cc1 %s -chain-include %s -ast-dump | FileCheck -strict-whitespace %s

// CHECK: TranslationUnitDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> <undeserialized declarations>
