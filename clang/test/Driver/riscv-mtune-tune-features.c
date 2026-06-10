// Check the advanced -mtune syntax with tune feature string

// RUN: %clang -### --target=riscv64 -mexperimental-mtune-syntax \
// RUN:     -mtune=sifive-x390:full-vec-fp64 -c %s 2>&1 | \
// RUN:     FileCheck --check-prefix=X390 %s
// X390: "-target-feature" "-single-element-vec-fp64"
// X390: "-tune-cpu" "sifive-x390"

// RUN: %clang -### --target=riscv64 -mexperimental-mtune-syntax \
// RUN:     -mtune=sifive-x280:single-element-vec-fp64 -c %s 2>&1 | \
// RUN:     FileCheck --check-prefix=X280 %s
// X280: "-target-feature" "+single-element-vec-fp64"
// X280: "-tune-cpu" "sifive-x280"

// RUN: not %clang --target=riscv64 -mtune=sifive-x390:full-vec-fp64 -c %s 2>&1 | \
// RUN:     FileCheck --check-prefix=NO-EXPERIMENTAL %s
// RUN: not %clang --target=riscv64 -mexperimental-mtune-syntax \
// RUN:     -mtune=sifive-x390:full-vec-fp64 -mno-experimental-mtune-syntax -c %s 2>&1 | \
// RUN:     FileCheck --check-prefix=NO-EXPERIMENTAL %s
// NO-EXPERIMENTAL: invalid -mtune string 'sifive-x390:full-vec-fp64':
// NO-EXPERIMENTAL-SAME: require '-mexperimental-mtune-syntax' to use with tune feature string

// RUN: not %clang --target=riscv64 -mexperimental-mtune-syntax \
// RUN:     -mtune=sifive-p470:full-vec-fp64 -c %s 2>&1 | \
// RUN:     FileCheck --check-prefix=NO-DIRECTIVE %s
// NO-DIRECTIVE: invalid tune feature string 'full-vec-fp64':
// NO-DIRECTIVE-SAME: Processor 'sifive-p470' has no configurable tuning features

// RUN: not %clang --target=riscv64 -mexperimental-mtune-syntax \
// RUN:     -mtune=sifive-x280:full-vec-fp64,single-element-vec-fp64 -c %s 2>&1 | \
// RUN:     FileCheck --check-prefix=INVALID-DIRECTIVE %s
// INVALID-DIRECTIVE: invalid tune feature string 'full-vec-fp64,single-element-vec-fp64':
// INVALID-DIRECTIVE-SAME: Feature(s) 'single-element-vec-fp64' cannot appear in both positive and negative directives

// RUN: not %clang --target=riscv64 -mexperimental-mtune-syntax \
// RUN:     -mtune=sifive-x280:prefer-w-inst -c %s 2>&1 | \
// RUN:     FileCheck --check-prefix=UNSUPPORTED-DIRECTIVE %s
// UNSUPPORTED-DIRECTIVE: invalid tune feature string 'prefer-w-inst':
// UNSUPPORTED-DIRECTIVE-SAME: Directive 'prefer-w-inst' is not allowed to be used with processor 'sifive-x280'
