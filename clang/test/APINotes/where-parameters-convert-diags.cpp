// RUN: not %clang_cc1 -fsyntax-only -fapinotes %s -I %S/Inputs/WhereParametersConvertDiag 2>&1 | FileCheck %s

#include "WhereParametersConvertDiag.h"

// CHECK: error: 'Where' is not supported by binary API notes yet
