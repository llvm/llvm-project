void dontRenameBuiltins(int x) {
  __builtin_assume(x != 0);
  __builtin_trap();
}

// RUN: not clang-refactor-test rename-initiate -at=%s:2:3 -at=%s:3:3 -new-name=foo %s 2>&1 | FileCheck %s
// CHECK: error: could not rename symbol at the given location

// RUN: not clang-refactor-test list-actions -at=%s:2:3 %s 2>&1 | FileCheck --check-prefix=CHECK-BUILTIN %s
// CHECK-BUILTIN: Failed to initiate 1 actions because:
// CHECK-BUILTIN-NEXT: Rename: '__builtin_assume' is a builtin function that cannot be renamed
// CHECK-BUILTIN-NEXT: No refactoring actions are available at the given location

#include <system-header.h>

void dontRenameSystemSymbols() {
  systemFunction();
}
// RUN: not clang-refactor-test rename-initiate -at=%s:17:3 -new-name=foo %s -isystem %S/Inputs 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYSTEM %s
// CHECK-SYSTEM: 'systemFunction' cannot be renamed because it is declared in a system header

struct External {
  static void foo();
} __attribute__((external_source_symbol(language="Swift")));

void dontRenameExternalSourceSymbols() {
  External::foo();
}
// RUN: not clang-refactor-test rename-initiate -at=%s:27:3 -new-name=foo %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-EXTERNAL1 %s
// CHECK-EXTERNAL1: 'External' is declared in a Swift file; rename can be initiated in a Swift file only

// RUN: not clang-refactor-test rename-initiate -at=%s:27:13 -new-name=foo %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-EXTERNAL2 %s
// CHECK-EXTERNAL2: 'foo' is declared in a Swift file; rename can be initiated in a Swift file only

// Ensure that operators can't be renamed:
struct Stream {
};

Stream &operator <<(Stream &, int);

void renameArgsNotOperator(Stream x) { // CHECK-OP-X: rename local [[@LINE]]:35 -> [[@LINE]]:36
  int y = 0; // CHECK-OP-Y: rename local [[@LINE]]:7 -> [[@LINE]]:8
  x << // CHECK-OP-X: rename local [[@LINE]]:3 -> [[@LINE]]:4
  y << // CHECK-OP-Y: rename local [[@LINE]]:3 -> [[@LINE]]:4
  y; // CHECK-OP-Y: rename local [[@LINE]]:3 -> [[@LINE]]:4
}
// RUN: clang-refactor-test rename-initiate -at=%s:43:3 -new-name=foo %s | FileCheck --check-prefixes=CHECK-OP-X %s
// RUN: clang-refactor-test rename-initiate -at=%s:44:3 -at=%s:45:3 -new-name=foo %s | FileCheck --check-prefixes=CHECK-OP-Y %s

struct SystemStruct;

// RUN: not clang-refactor-test rename-initiate -at=%s:50:8 -new-name=foo %s -isystem %S/Inputs 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYSTEM2 %s
// CHECK-SYSTEM2: 'SystemStruct' cannot be renamed because it is declared in a system header

typedef struct SystemStruct SystemTypedef;

// RUN: not clang-refactor-test rename-initiate -at=%s:55:29 -new-name=foo %s -isystem %S/Inputs 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYSTEM3 %s
// CHECK-SYSTEM3: 'SystemTypedef' cannot be renamed because it is declared in a system header

enum SystemEnum;

// RUN: not clang-refactor-test rename-initiate -at=%s:60:6 -new-name=foo %s -isystem %S/Inputs 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYSTEM4 %s
// CHECK-SYSTEM4: 'SystemEnum' cannot be renamed because it is declared in a system header

void systemFunction();

// RUN: not clang-refactor-test rename-initiate -at=%s:65:6 -new-name=foo %s -isystem %S/Inputs 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYSTEM %s

int systemVariable;

// RUN: not clang-refactor-test rename-initiate -at=%s:69:5 -new-name=foo %s -isystem %S/Inputs 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SYSTEM5 %s
// CHECK-SYSTEM5: 'systemVariable' cannot be renamed because it is declared in a system header
