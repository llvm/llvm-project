// RUN: not %clang -c -fminimize-whitespace %s 2>&1 | FileCheck %s --check-prefix=ON -DOPT=-fminimize-whitespace
// RUN: not %clang -c -fkeep-system-includes %s 2>&1 | FileCheck %s --check-prefix=ON -DOPT=-fkeep-system-includes
// ON: error: invalid argument '[[OPT]]' only allowed with '-E'

// RUN: not %clang -c -fno-minimize-whitespace %s 2>&1 | FileCheck %s  --check-prefix=OFF -DOPT=-fno-minimize-whitespace
// RUN: not %clang -c -fno-keep-system-includes %s 2>&1 | FileCheck %s  --check-prefix=OFF -DOPT=-fno-keep-system-includes
// OFF: error: invalid argument '[[OPT]]' only allowed with '-E'

// RUN: not %clang -E -fminimize-whitespace -x assembler-with-cpp %s 2>&1 | FileCheck %s --check-prefix=ASM -DOPT=-fminimize-whitespace
// RUN: not %clang -E -fkeep-system-includes -x assembler-with-cpp %s 2>&1 | FileCheck %s --check-prefix=ASM -DOPT=-fkeep-system-includes
// ASM: error: '[[OPT]]' invalid for input of type assembler-with-cpp
