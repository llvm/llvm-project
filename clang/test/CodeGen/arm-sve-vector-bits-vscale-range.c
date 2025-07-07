// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-MINMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=1
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=2 -mvscale-max=2 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-MINMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=2
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=4 -mvscale-max=4 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-MINMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=4
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=8 -mvscale-max=8 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-MINMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=8
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=16 -mvscale-max=16 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-MINMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=16
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +sme -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-MINMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=1
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +sme -mvscale-min=2 -mvscale-max=2 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-MINMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=2
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=1 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=1
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=2 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=2
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=4 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=4
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=8 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=8
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=16 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOMAX,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE -D#VBITS=16
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -target-feature +sme -mvscale-min=1 -mvscale-max=0 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-UNBOUNDED,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -mvscale-min=1 -mvscale-max=0 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-UNBOUNDED,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NONE,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -target-feature +sme -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NONE,CHECK-NOSTREAMING,CHECK-NOCOMPATIBLE
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -target-feature +sme -mvscale-streaming-min=1 -mvscale-streaming-max=1 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NONE,CHECK-STREAMING,CHECK-NOCOMPATIBLE -D#STREAMINGVBITS=1
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -target-feature +sme -mvscale-streaming-min=4 -mvscale-streaming-max=4 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NONE,CHECK-STREAMING,CHECK-NOCOMPATIBLE -D#STREAMINGVBITS=4
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -target-feature +sme -mvscale-streaming-min=4 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NONE,CHECK-STREAMING-NOMAX,CHECK-NOCOMPATIBLE -D#STREAMINGVBITS=4
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -target-feature +sme -mvscale-min=2 -mvscale-streaming-min=4 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOMAX,CHECK-STREAMING-NOMAX,CHECK-COMPATIBLE-NOMAX -D#VBITS=2 -D#STREAMINGVBITS=4
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -target-feature +sme -mvscale-max=2 -mvscale-streaming-max=4 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-NOMIN,CHECK-STREAMING-NOMIN,CHECK-COMPATIBLE-NOMIN -D#VBITS=2 -D#STREAMINGVBITS=4
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -target-feature +sme \
// RUN:            -mvscale-min=2 -mvscale-streaming-min=4 -mvscale-max=2 -mvscale-streaming-max=4 -emit-llvm -o - %s | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-MINMAX,CHECK-STREAMING,CHECK-COMPATIBLE -D#VBITS=2 -D#STREAMINGVBITS=4


// CHECK-LABEL: @func() #0
// CHECK-LABEL: @func2() #1
// CHECK-LABEL: @func3() #2
// CHECK-MINMAX: attributes #0 = { {{.*}} vscale_range([[#VBITS]],[[#VBITS]]) {{.*}} }
// CHECK-NOMAX: attributes #0 = { {{.*}} vscale_range([[#VBITS]],16) {{.*}} }
// CHECK-NOMIN: attributes #0 = { {{.*}} vscale_range(1,[[#VBITS]]) {{.*}} }
// CHECK-UNBOUNDED: attributes #0 = { {{.*}} vscale_range(1,16) {{.*}} }
// CHECK-NONE: attributes #0 = { {{.*}} vscale_range(1,16) {{.*}} }
// CHECK-STREAMING: attributes #1 = { {{.*}} vscale_range([[#STREAMINGVBITS]],[[#STREAMINGVBITS]])
// CHECK-STREAMING-NOMAX: attributes #1 = { {{.*}} vscale_range([[#STREAMINGVBITS]],16)
// CHECK-STREAMING-NOMIN: attributes #1 = { {{.*}} vscale_range(1,[[#STREAMINGVBITS]]) {{.*}} }
// CHECK-NOSTREAMING: attributes #1 = { {{.*}} vscale_range(1,16) {{.*}} }
// CHECK-NOCOMPATIBLE: attributes #2 = { {{.*}} vscale_range(1,16) {{.*}} }
// CHECK-COMPATIBLE: attributes #2 = { {{.*}} vscale_range([[#VBITS]],[[#STREAMINGVBITS]]) {{.*}} }
// CHECK-COMPATIBLE-NOMAX: attributes #2 = { {{.*}} vscale_range([[#VBITS]],16) {{.*}} }
// CHECK-COMPATIBLE-NOMIN: attributes #2 = { {{.*}} vscale_range(1,[[#STREAMINGVBITS]]) {{.*}} }
void func(void) {}
__arm_locally_streaming void func2(void) {}
void func3(void) __arm_streaming_compatible {}
