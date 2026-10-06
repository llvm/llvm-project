// RUN: %clang_cc1 -fsyntax-only -fapinotes %s -I %S/Inputs/WhereParametersConvertDiag
// RUN: not %clang_cc1 -fsyntax-only -fapinotes %s -I %S/Inputs/WhereParametersEmptyWhereDiag 2>&1 | FileCheck %s --check-prefix=EMPTY-WHERE

#include "WhereParametersConvertDiag.h"

// EMPTY-WHERE-COUNT-2: error: 'Where' requires 'Parameters'
