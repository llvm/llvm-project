// The -time option prints timing information for the various subcommands in a
// format similar to that used by gcc. When compiling and linking, this will
// include the time to call clang-${LLVM_VERSION_MAJOR} and the linker. Since
// the name of the linker could vary across platforms, and name of the compiler
// could be something different, just check that whatever is printed to stderr
// looks like timing information.

// Ideally, this should be tested on various platforms, but that requires the
// the full toolchain, including a linker to be present. The initial author of
// the test only had access to Linux on x86 which is why this is only enabled
// there. More platforms ought to be added if possible.

// REQUIRES: x86-registered-target
// REQUIRES: x86_64-linux

// RUN: %clang --target=x86_64-pc-linux -time -c -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
// RUN: %clang --target=x86_64-pc-linux -time -S -emit-llvm -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
// RUN: %clang --target=x86_64-pc-linux -time -S -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=COMPILE-ONLY
// RUN: %clang --target=x86_64-pc-linux -time -o /dev/null %s 2>&1 \
// RUN:     | FileCheck %s --check-prefix=COMPILE-AND-LINK

// COMPILE-ONLY: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}
// COMPILE-ONLY-NOT: {{.}}

// COMPILE-AND-LINK: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}
// COMPILE-AND-LINK: # {{.+}} {{[0-9]+(.[0-9]+)?}} {{[0-9]+(.[0-9]+)?}}

int main() {
  return 0;
}
