// RUN: %clang -### -target amdgcn-amd-amdhsa -mcpu=gfx900 -mcode-object-v3 %s 2>&1 | FileCheck --check-prefix=CODE-OBJECT-V3 %s
// CODE-OBJECT-V3: warning: argument '-mcode-object-v3' is deprecated, use '-mcode-object-version=3' instead [-Wdeprecated]
// CODE-OBJECT-V3: "-mllvm" "--amdhsa-code-object-version=3"

// RUN: %clang -### -target amdgcn-amd-amdhsa amdgcn -mcpu=gfx900 -mno-code-object-v3 %s 2>&1 | FileCheck --check-prefix=NO-CODE-OBJECT-V3 %s
// NO-CODE-OBJECT-V3: warning: argument '-mno-code-object-v3' is deprecated, use '-mcode-object-version=2' instead [-Wdeprecated]
// NO-CODE-OBJECT-V3: "-mllvm" "--amdhsa-code-object-version=2"

// RUN: %clang -### -target amdgcn-amd-amdhsa -mcpu=gfx900 -mcode-object-v3 -mno-code-object-v3 -mcode-object-v3 %s 2>&1 | FileCheck --check-prefix=MUL-CODE-OBJECT-V3 %s
// MUL-CODE-OBJECT-V3: warning: argument '-mcode-object-v3' is deprecated, use '-mcode-object-version=3' instead [-Wdeprecated]
// MUL-CODE-OBJECT-V3: "-mllvm" "--amdhsa-code-object-version=3"
