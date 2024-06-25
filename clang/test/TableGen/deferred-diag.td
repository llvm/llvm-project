// RUN: clang-tblgen -gen-clang-diags-defs -I%S %s -o - 2>&1 | \
// RUN:    FileCheck --strict-whitespace %s
include "DiagnosticBase.inc"

// Test usage of Deferrable and NonDeferrable in diagnostics.

def test_default : Error<"this error is non-deferrable by default">;
// CHECK-DAG: DIAG(test_default, {{.*}}SFINAE_SubstitutionFailure, false, true, true, false, 0)

def test_deferrable : Error<"this error is deferrable">, Deferrable;
// CHECK-DAG: DIAG(test_deferrable, {{.*}} SFINAE_SubstitutionFailure, false, true, true, true, 0)

def test_non_deferrable : Error<"this error is non-deferrable">, NonDeferrable;
// CHECK-DAG: DIAG(test_non_deferrable, {{.*}} SFINAE_SubstitutionFailure, false, true, true, false, 0)

let Deferrable = 1 in {

def test_let : Error<"this error is deferrable by let">;
// CHECK-DAG: DIAG(test_let, {{.*}} SFINAE_SubstitutionFailure, false, true, true, true, 0)

// Make sure TextSubstitution is allowed in the let Deferrable block.
def textsub : TextSubstitution<"%select{text1|text2}0">;

def test_let2 : Error<"this error is deferrable by let %sub{textsub}0">;
// CHECK-DAG: DIAG(test_let2, {{.*}} SFINAE_SubstitutionFailure, false, true, true, true, 0)

}
