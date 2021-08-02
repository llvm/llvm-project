// RUN: not %clang -c -fminimize-whitespace %s 2>&1 | FileCheck %s --check-prefix=ON
// ON: error: invalid argument '-fminimize-whitespace' only allowed with '-E'

// RUN: not %clang -c -fno-minimize-whitespace %s 2>&1 | FileCheck %s  --check-prefix=OFF
// OFF: error: invalid argument '-fno-minimize-whitespace' only allowed with '-E'

// RUN: not %clang -E -fminimize-whitespace -x assembler-with-cpp %s 2>&1 | FileCheck %s --check-prefix=ASM
// ASM: error: '-fminimize-whitespace' invalid for input of type assembler-with-cpp
