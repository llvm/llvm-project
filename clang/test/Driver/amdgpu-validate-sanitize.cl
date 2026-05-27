// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx900:xnack+ \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1250    \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx900 \
// RUN:   -fsanitize=undefined \
// RUN:   -fsanitize=unsigned-integer-overflow     \
// RUN:   -fsanitize=float-divide-by-zero     \
// RUN:   -fsanitize=unsigned-integer-overflow \
// RUN:   -fsanitize=unsigned-shift-base \
// RUN:   -fsanitize=implicit-conversion \
// RUN:   -fsanitize=nullability \
// RUN:   -fsanitize=local-bounds \
// RUN:   -fsanitize=alloc-token \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=GENERIC %s


// FIXME: This should error, but is silently ignored
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx900:xnack- \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=ERR %s

// CHECK: "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-mlink-bitcode-file" "{{.*}}asanrtl.bc"
// CHECK-SAME: "-fsanitize=address"


// GENERIC: "-fsanitize=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,float-divide-by-zero,function,integer-divide-by-zero,nonnull-attribute,null,nullability-arg,nullability-assign,nullability-return,pointer-overflow,return,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,unreachable,vla-bound,unsigned-integer-overflow,unsigned-shift-base,implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change,implicit-bitfield-conversion,local-bounds,alloc-token" "-fsanitize-recover=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,float-divide-by-zero,function,integer-divide-by-zero,nonnull-attribute,null,nullability-arg,nullability-assign,nullability-return,pointer-overflow,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,vla-bound,unsigned-integer-overflow,unsigned-shift-base,implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change,implicit-bitfield-conversion" "-fsanitize-trap=local-bounds" "-fsanitize-merge=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,function,integer-divide-by-zero,nonnull-attribute,null,pointer-overflow,return,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,unreachable,vla-bound"


// FIXME: Should not be forwarding argument
// ERR-NOT: asanrtl.bc
// ERR: "-fsanitize=address"
