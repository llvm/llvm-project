// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -E -dM %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -E -dM %s \
// RUN:   -o - | FileCheck %s

// CHECK-NOT: __riscv_xsfcease {{.*$}}
// CHECK-NOT: __riscv_xsfvcp {{.*$}}
// CHECK-NOT: __riscv_xsfvfbfexp16e {{.*$}}
// CHECK-NOT: __riscv_xsfvfexp16e {{.*$}}
// CHECK-NOT: __riscv_xsfvfexp32e {{.*$}}
// CHECK-NOT: __riscv_xsfvfexpa {{.*$}}
// CHECK-NOT: __riscv_xsfvfexpa64e {{.*$}}
// CHECK-NOT: __riscv_xsfvfnrclipxfqf {{.*$}}
// CHECK-NOT: __riscv_xsfvfwmaccqqq {{.*$}}
// CHECK-NOT: __riscv_xsfqmaccdod {{.*$}}
// CHECK-NOT: __riscv_xsfvqmaccqoq {{.*$}}
// CHECK-NOT: __riscv_xsifivecdiscarddlone {{.*$}}
// CHECK-NOT: __riscv_xsifivecflushdlone {{.*$}}
// CHECK-NOT: __riscv_xsfmm128t {{.*$}}
// CHECK-NOT: __riscv_xsfmm16t {{.*$}}
// CHECK-NOT: __riscv_xsfmm32a8i {{.*$}}
// CHECK-NOT: __riscv_xsfmm32a8f {{.*$}}
// CHECK-NOT: __riscv_xsfmm32a16f {{.*$}}
// CHECK-NOT: __riscv_xsfmm32a32f {{.*$}}
// CHECK-NOT: __riscv_xsfmm32a32t {{.*$}}
// CHECK-NOT: __riscv_xsfmm64a64f {{.*$}}
// CHECK-NOT: __riscv_xsfmm64t {{.*$}}
// CHECK-NOT: __riscv_xsfmmbase {{.*$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfcease -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFCEASE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfcease -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFCEASE-EXT %s
// CHECK-XSFCEASE-EXT: __riscv_xsfcease 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvcp -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVCP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvcp -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVCP-EXT %s
// CHECK-XSFVCP-EXT: __riscv_xsfvcp 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvfbfexp16e_zvfbfmin -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFBFEXP16E-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvfbfexp16e_zvfbfmin -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFBFEXP16E-EXT %s
// CHECK-XSFVFBFEXP16E-EXT: __riscv_xsfvfbfexp16e 5000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvfexp16e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFEXP16E-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvfexp16e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFEXP16E-EXT %s
// CHECK-XSFVFEXP16E-EXT: __riscv_xsfvfexp16e 5000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvfexp32e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFEXP32E-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvfexp32e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFEXP32E-EXT %s
// CHECK-XSFVFEXP32E-EXT: __riscv_xsfvfexp32e 5000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvfexpa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFEXPA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvfexpa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFEXPA-EXT %s
// CHECK-XSFVFEXPA-EXT: __riscv_xsfvfexpa 2000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvfexpa64e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFEXPA64E-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvfexpa64e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFEXPA64E-EXT %s
// CHECK-XSFVFEXPA64E-EXT: __riscv_xsfvfexpa64e 2000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvfnrclipxfqf -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFNRCLIPXFQF-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvfnrclipxfqf -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFNRCLIPXFQF-EXT %s
// CHECK-XSFVFNRCLIPXFQF-EXT: __riscv_xsfvfnrclipxfqf 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvfwmaccqqq -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFWMACCQQQ-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvfwmaccqqq -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVFWMACCQQQ-EXT %s
// CHECK-XSFVFWMACCQQQ-EXT: __riscv_xsfvfwmaccqqq 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvqmaccdod -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVQMACCDOD-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvqmaccdod -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVQMACCDOD-EXT %s
// CHECK-XSFVQMACCDOD-EXT: __riscv_xsfvqmaccdod 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsfvqmaccqoq -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVQMACCQOQ-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsfvqmaccqoq -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFVQMACCQOQ-EXT %s
// CHECK-XSFVQMACCQOQ-EXT: __riscv_xsfvqmaccqoq 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsifivecdiscarddlone -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSIFIVECDISCARDDLONE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsifivecdiscarddlone -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSIFIVECDISCARDDLONE-EXT %s
// CHECK-XSIFIVECDISCARDDLONE-EXT: __riscv_xsifivecdiscarddlone 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixsifivecflushdlone -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSIFIVECFLUSHDLONE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixsifivecflushdlone -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSIFIVECFLUSHDLONE-EXT %s
// CHECK-XSIFIVECFLUSHDLONE-EXT: __riscv_xsifivecflushdlone 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm128t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM128T %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm128t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM128T %s
// CHECK-XSFMM128T: __riscv_xsfmm128t  6000{{$}}
//
// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm16t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM16T %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm16t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM16T %s
// CHECK-XSFMM16T: __riscv_xsfmm16t  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32a8i -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a8I %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32a8i -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a8I %s
// CHECK-XSFMM32a8I: __riscv_xsfmm32a8i  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32a8f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32A8F %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32a8f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32A8F %s
// CHECK-XSFMM32A8F: __riscv_xsfmm32a8f  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32a16f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a16F %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32a16f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a16F %s
// CHECK-XSFMM32a16F: __riscv_xsfmm32a16f  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32a32f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a32F %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32a32f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32a32F %s
// CHECK-XSFMM32a32F: __riscv_xsfmm32a32f  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm32t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32T %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm32t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM32T %s
// CHECK-XSFMM32T: __riscv_xsfmm32t  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm64a64f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM64a64f %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm64a64f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM64a64f %s
// CHECK-XSFMM64a64f: __riscv_xsfmm64a64f  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmm64t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM64T %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmm64t -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMM64T %s
// CHECK-XSFMM64T: __riscv_xsfmm64t  6000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_xsfmmbase -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMMBASE %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_xsfmmbase -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XSFMMBASE %s
// CHECK-XSFMMBASE: __riscv_xsfmmbase  6000{{$}}
