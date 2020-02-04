// Slow FMAF and slow f32 denormals
// RUN: %clang -### -target amdgcn--amdhsa -c -mcpu=pitcairn %s 2>&1 | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s
// RUN: %clang -### -cl-denorms-are-zero -o - -target amdgcn--amdhsa -c -mcpu=pitcairn %s 2>&1 | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s

// Fast FMAF, but slow f32 denormals
// RUN: %clang -### -target amdgcn--amdhsa -c -mcpu=tahiti %s 2>&1 | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s
// RUN: %clang -### -cl-denorms-are-zero -o - -target amdgcn--amdhsa -c -mcpu=tahiti %s 2>&1 | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s

// Fast F32 denormals, but slow FMAF
// RUN: %clang -### -target amdgcn--amdhsa -c -mcpu=fiji %s 2>&1 | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s
// RUN: %clang -### -cl-denorms-are-zero -o - -target amdgcn--amdhsa -c -mcpu=fiji %s 2>&1 | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s

// Fast F32 denormals and fast FMAF
// RUN: %clang -### -target amdgcn--amdhsa -c -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefixes=AMDGCN,AMDGCN-DENORM %s
// RUN: %clang -### -cl-denorms-are-zero -o - -target amdgcn--amdhsa -c -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefixes=AMDGCN,AMDGCN-FLUSH %s

// AMDGCN-FLUSH: "-fdenormal-fp-math-f32=preserve-sign,preserve-sign"

// This should be omitted and default to ieee
// AMDGCN-DENORM-NOT: "-fdenormal-fp-math-f32"
