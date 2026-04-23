// RUN: not clang-move --nonsense %s -- 2>&1 | FileCheck %s --check-prefix=CLANG-MOVE
// RUN: not clang-reorder-fields --nonsense %s -- 2>&1 | FileCheck %s --check-prefix=CLANG-REORDER-FIELDS
// RUN: not clang-change-namespace --nonsense %s -- 2>&1 | FileCheck %s --check-prefix=CLANG-CHANGE-NAMESPACE
// RUN: not clang-include-fixer --nonsense %s -- 2>&1 | FileCheck %s --check-prefix=CLANG-INCLUDE-FIXER
// RUN: not find-all-symbols --nonsense %s -- 2>&1 | FileCheck %s --check-prefix=FIND-ALL-SYMBOLS

// CLANG-MOVE: clang-move: Unknown command line argument '--nonsense'
// CLANG-REORDER-FIELDS: clang-reorder-fields: Unknown command line argument '--nonsense'
// CLANG-CHANGE-NAMESPACE: clang-change-namespace: Unknown command line argument '--nonsense'
// CLANG-INCLUDE-FIXER: clang-include-fixer: Unknown command line argument '--nonsense'
// FIND-ALL-SYMBOLS: find-all-symbols: Unknown command line argument '--nonsense'

int main() { return 0; }
