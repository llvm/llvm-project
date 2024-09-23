// The -time option prints timing information for the various subcommands in a
// format similar to that used by gcc. When compiling and linking, this will
// include the time to call clang-${LLVM_VERSION_MAJOR} and the linker. Since
// the name of the linker could vary across platforms, and name of the compiler
// could be something different, just check that whatever is printed to stderr
// looks like timing information.

// RUN: %clang -time -c -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
// RUN: %clang -time -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
// RUN: %clang -time -S -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
// RUN: %clang -time -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=COMPILE-AND-LINK

// COMPILE-ONLY: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}
// COMPILE-ONLY-NOT: {{.}}

// COMPILE-AND-LINK: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}
// COMPILE-AND-LINK: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}

int main() {
  return 0;
}
