/// Check driver handling for "-fvisibility-global-new-delete-hidden" and "-fvisibility-global-new-delete=".

/// These options are not added by default.
// RUN: %clang -### --target=x86_64-unknown-unknown -c -emit-llvm %s 2>&1 | \
// RUN:   FileCheck -check-prefix=DEFAULTS %s
// DEFAULTS-NOT: "-fvisibility-global-new-delete="
// DEFAULTS-NOT: "-fvisibility-global-new-delete-hidden"

// DEFINE: %{implicit-check-nots} = --implicit-check-not=-fvisibility-global-new-delete= --implicit-check-not=-fvisibility-global-new-delete-hidden

/// "-fvisibility-global-new-delete=source" added by default for PS5.
// RUN: %clang -### --target=x86_64-sie-ps5 -c -emit-llvm %s 2>&1 | \
// RUN:   FileCheck -check-prefix=PS5 %s 
// PS5: "-fvisibility-global-new-delete=source"

/// -fvisibility-global-new-delete-hidden added explicitly.
// RUN: %clang -### --target=x86_64-unknown-unknown -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete-hidden %s 2>&1 | FileCheck -check-prefixes=VGNDH,DEPRECATED %s %{implicit-check-nots}
// RUN: %clang -### --target=x86_64-sie-ps5 -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete-hidden %s 2>&1 | FileCheck -check-prefixes=VGNDH,DEPRECATED %s %{implicit-check-nots}
// DEPRECATED-DAG:  clang: warning: argument '-fvisibility-global-new-delete-hidden' is deprecated, use '-fvisibility-global-new-delete=force-hidden' instead [-Wdeprecated]
// VGNDH-DAG: "-fvisibility-global-new-delete=force-hidden"

/// -fvisibility-global-new-delete=force-hidden added explicitly.
// RUN: %clang -### --target=x86_64-unknown-unknown -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete=force-hidden %s 2>&1 | FileCheck -check-prefixes=VGNDH %s %{implicit-check-nots}
// RUN: %clang -### --target=x86_64-sie-ps5 -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete=force-hidden %s 2>&1 | FileCheck -check-prefixes=VGNDH %s %{implicit-check-nots}

/// -fvisibility-global-new-delete=force-protected added explicitly.
// RUN: %clang -### --target=x86_64-unknown-unknown -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete=force-protected %s 2>&1 | FileCheck -check-prefixes=VGNDP %s %{implicit-check-nots}
// RUN: %clang -### --target=x86_64-sie-ps5 -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete=force-protected %s 2>&1 | FileCheck -check-prefixes=VGNDP %s %{implicit-check-nots}
// VGNDP-DAG: "-fvisibility-global-new-delete=force-protected"

/// -fvisibility-global-new-delete=force-default added explicitly.
// RUN: %clang -### --target=x86_64-unknown-unknown -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete=force-default %s 2>&1 | FileCheck -check-prefixes=VGNDD %s %{implicit-check-nots}
// RUN: %clang -### --target=x86_64-sie-ps5 -c -emit-llvm \
// RUN:   -fvisibility-global-new-delete=force-default %s 2>&1 | FileCheck -check-prefixes=VGNDD %s %{implicit-check-nots}
// VGNDD-DAG: "-fvisibility-global-new-delete=force-default"

/// last specfied used: -fvisibility-global-new-delete-hidden.
// RUN: %clang -### --target=x86_64-unknown-unknown -c -emit-llvm \ 
// RUN:   -fvisibility-global-new-delete=force-default -fvisibility-global-new-delete=force-protected -fvisibility-global-new-delete-hidden %s 2>&1 | \
// RUN:     FileCheck -check-prefixes=VGNDH,DEPRECATED %s %{implicit-check-nots}

/// last specfied used: -fvisibility-global-new-delete=.
// RUN: %clang -### --target=x86_64-unknown-unknown -c -emit-llvm \ 
// RUN:   -fvisibility-global-new-delete-hidden -fvisibility-global-new-delete=force-default -fvisibility-global-new-delete=force-protected %s 2>&1 | \
// RUN:     FileCheck -check-prefixes=VGNDP,DEPRECATED %s %{implicit-check-nots}
