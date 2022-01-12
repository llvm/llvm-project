// Check object emission.
// RUN: %clang -### --target=spirv64 -x cl -c %s 2>&1 | FileCheck --check-prefix=SPV64 %s
// RUN: %clang -### --target=spirv64 %s 2>&1 | FileCheck --check-prefix=SPV64 %s
// RUN: %clang -### --target=spirv64 -x ir -c %s 2>&1 | FileCheck --check-prefix=SPV64 %s
// RUN: %clang -### --target=spirv64 -x clcpp -c %s 2>&1 | FileCheck --check-prefix=SPV64 %s
// RUN: %clang -### --target=spirv64 -x c -c %s 2>&1 | FileCheck --check-prefix=SPV64 %s

// SPV64: "-cc1" "-triple" "spirv64"
// SPV64-SAME: "-o" [[BC:".*bc"]]
// SPV64: {{llvm-spirv.*"}} [[BC]] "-o" {{".*o"}}

// RUN: %clang -### --target=spirv32 -x cl -c %s 2>&1 | FileCheck --check-prefix=SPV32 %s
// RUN: %clang -### --target=spirv32 %s 2>&1 | FileCheck --check-prefix=SPV32 %s
// RUN: %clang -### --target=spirv32 -x ir -c %s 2>&1 | FileCheck --check-prefix=SPV32 %s
// RUN: %clang -### --target=spirv32 -x clcpp -c %s 2>&1 | FileCheck --check-prefix=SPV32 %s
// RUN: %clang -### --target=spirv32 -x c -c %s 2>&1 | FileCheck --check-prefix=SPV32 %s

// SPV32: "-cc1" "-triple" "spirv32"
// SPV32-SAME: "-o" [[BC:".*bc"]]
// SPV32: {{llvm-spirv.*"}} [[BC]] "-o" {{".*o"}}

//-----------------------------------------------------------------------------
// Check Assembly emission.
// RUN: %clang -### --target=spirv64 -x cl -S %s 2>&1 | FileCheck --check-prefix=SPT64 %s
// RUN: %clang -### --target=spirv64 -x ir -S %s 2>&1 | FileCheck --check-prefix=SPT64 %s
// RUN: %clang -### --target=spirv64 -x clcpp -c %s 2>&1 | FileCheck --check-prefix=SPV64 %s
// RUN: %clang -### --target=spirv64 -x c -S %s 2>&1 | FileCheck --check-prefix=SPT64 %s

// SPT64: "-cc1" "-triple" "spirv64"
// SPT64-SAME: "-o" [[BC:".*bc"]]
// SPT64: {{llvm-spirv.*"}} [[BC]] "--spirv-tools-dis" "-o" {{".*s"}}

// RUN: %clang -### --target=spirv32 -x cl -S %s 2>&1 | FileCheck --check-prefix=SPT32 %s
// RUN: %clang -### --target=spirv32 -x ir -S %s 2>&1 | FileCheck --check-prefix=SPT32 %s
// RUN: %clang -### --target=spirv32 -x clcpp -c %s 2>&1 | FileCheck --check-prefix=SPV32 %s
// RUN: %clang -### --target=spirv32 -x c -S %s 2>&1 | FileCheck --check-prefix=SPT32 %s

// SPT32: "-cc1" "-triple" "spirv32"
// SPT32-SAME: "-o" [[BC:".*bc"]]
// SPT32: {{llvm-spirv.*"}} [[BC]] "--spirv-tools-dis" "-o" {{".*s"}}

//-----------------------------------------------------------------------------
// Check assembly input -> object output
// RUN: %clang -### --target=spirv64 -x assembler -c %s 2>&1 | FileCheck --check-prefix=ASM %s
// RUN: %clang -### --target=spirv32 -x assembler -c %s 2>&1 | FileCheck --check-prefix=ASM %s
// ASM: {{llvm-spirv.*"}} {{".*"}} "-to-binary" "-o" {{".*o"}}

//-----------------------------------------------------------------------------
// Check --save-temps.
// RUN: %clang -### --target=spirv64 -x cl -c %s --save-temps 2>&1 | FileCheck --check-prefix=TMP %s

// TMP: "-cc1" "-triple" "spirv64"
// TMP-SAME: "-E"
// TMP-SAME: "-o" [[I:".*i"]]
// TMP: "-cc1" "-triple" "spirv64"
// TMP-SAME: "-o" [[BC:".*bc"]]
// TMP-SAME: [[I]]
// TMP: {{llvm-spirv.*"}} [[BC]] "--spirv-tools-dis" "-o" [[S:".*s"]]
// TMP: {{llvm-spirv.*"}} [[S]] "-to-binary" "-o" {{".*o"}}

//-----------------------------------------------------------------------------
// Check linking when multiple input files are passed.
// RUN: %clang -### -target spirv64 %s %s 2>&1 | FileCheck --check-prefix=SPLINK %s

// SPLINK: "-cc1" "-triple" "spirv64"
// SPLINK-SAME: "-o" [[BC:".*bc"]]
// SPLINK: {{llvm-spirv.*"}} [[BC]] "-o" [[SPV1:".*o"]]
// SPLINK: "-cc1" "-triple" "spirv64"
// SPLINK-SAME: "-o" [[BC:".*bc"]]
// SPLINK: {{llvm-spirv.*"}} [[BC]] "-o" [[SPV2:".*o"]]
// SPLINK: {{spirv-link.*"}} [[SPV1]] [[SPV2]] "-o" "a.out"
