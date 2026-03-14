// RUN: %clang --version 2>&1 | FileCheck %s

CHECK: clang version

// Don't add llvm::TargetRegistry::printRegisteredTargetsForVersion()
// to --version output, see D79210 and D33900.
CHECK-NOT: Registered Targets:
