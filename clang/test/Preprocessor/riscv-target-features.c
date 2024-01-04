// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -o - | FileCheck %s

// CHECK-NOT: __riscv_a {{.*$}}
// CHECK-NOT: __riscv_atomic
// CHECK-NOT: __riscv_c {{.*$}}
// CHECK-NOT: __riscv_compressed {{.*$}}
// CHECK-NOT: __riscv_d {{.*$}}
// CHECK-NOT: __riscv_f {{.*$}}
// CHECK-NOT: __riscv_flen {{.*$}}
// CHECK-NOT: __riscv_fdiv {{.*$}}
// CHECK-NOT: __riscv_fsqrt {{.*$}}
// CHECK-NOT: __riscv_h {{.*$}}
// CHECK-NOT: __riscv_div {{.*$}}
// CHECK-NOT: __riscv_m {{.*$}}
// CHECK-NOT: __riscv_mul {{.*$}}
// CHECK-NOT: __riscv_muldiv {{.*$}}
// CHECK-NOT: __riscv_svinval {{.*$}}
// CHECK-NOT: __riscv_svnapot {{.*$}}
// CHECK-NOT: __riscv_svpbmt {{.*$}}
// CHECK-NOT: __riscv_v {{.*$}}
// CHECK-NOT: __riscv_v_elen {{.*$}}
// CHECK-NOT: __riscv_v_elen_fp {{.*$}}
// CHECK-NOT: __riscv_v_intrinsic {{.*$}}
// CHECK-NOT: __riscv_v_min_vlen {{.*$}}
// CHECK-NOT: __riscv_vector {{.*$}}
// CHECK-NOT: __riscv_xcvalu {{.*$}}
// CHECK-NOT: __riscv_xcvbi {{.*$}}
// CHECK-NOT: __riscv_xcvbitmanip {{.*$}}
// CHECK-NOT: __riscv_xcvelw {{.*$}}
// CHECK-NOT: __riscv_xcvmac {{.*$}}
// CHECK-NOT: __riscv_xcvmem {{.*$}}
// CHECK-NOT: __riscv_xcvsimd {{.*$}}
// CHECK-NOT: __riscv_xsfvcp {{.*$}}
// CHECK-NOT: __riscv_xsfvfnrclipxfqf {{.*$}}
// CHECK-NOT: __riscv_xsfvfwmaccqqq {{.*$}}
// CHECK-NOT: __riscv_xsfqmaccdod {{.*$}}
// CHECK-NOT: __riscv_xsfvqmaccqoq {{.*$}}
// CHECK-NOT: __riscv_xtheadba {{.*$}}
// CHECK-NOT: __riscv_xtheadbb {{.*$}}
// CHECK-NOT: __riscv_xtheadbs {{.*$}}
// CHECK-NOT: __riscv_xtheadcmo {{.*$}}
// CHECK-NOT: __riscv_xtheadcondmov {{.*$}}
// CHECK-NOT: __riscv_xtheadfmemidx {{.*$}}
// CHECK-NOT: __riscv_xtheadmac {{.*$}}
// CHECK-NOT: __riscv_xtheadmemidx {{.*$}}
// CHECK-NOT: __riscv_xtheadmempair {{.*$}}
// CHECK-NOT: __riscv_xtheadsync {{.*$}}
// CHECK-NOT: __riscv_xtheadvdot {{.*$}}
// CHECK-NOT: __riscv_xventanacondops {{.*$}}
// CHECK-NOT: __riscv_zawrs {{.*$}}
// CHECK-NOT: __riscv_zba {{.*$}}
// CHECK-NOT: __riscv_zbb {{.*$}}
// CHECK-NOT: __riscv_zbc {{.*$}}
// CHECK-NOT: __riscv_zbkb {{.*$}}
// CHECK-NOT: __riscv_zbkc {{.*$}}
// CHECK-NOT: __riscv_zbkx {{.*$}}
// CHECK-NOT: __riscv_zbs {{.*$}}
// CHECK-NOT: __riscv_zca {{.*$}}
// CHECK-NOT: __riscv_zcb {{.*$}}
// CHECK-NOT: __riscv_zcd {{.*$}}
// CHECK-NOT: __riscv_zce {{.*$}}
// CHECK-NOT: __riscv_zcf {{.*$}}
// CHECK-NOT: __riscv_zcmp {{.*$}}
// CHECK-NOT: __riscv_zcmt {{.*$}}
// CHECK-NOT: __riscv_zdinx {{.*$}}
// CHECK-NOT: __riscv_zfh {{.*$}}
// CHECK-NOT: __riscv_zfhmin {{.*$}}
// CHECK-NOT: __riscv_zfinx {{.*$}}
// CHECK-NOT: __riscv_zhinx {{.*$}}
// CHECK-NOT: __riscv_zhinxmin {{.*$}}
// CHECK-NOT: __riscv_zicbom {{.*$}}
// CHECK-NOT: __riscv_zicbop {{.*$}}
// CHECK-NOT: __riscv_zicboz {{.*$}}
// CHECK-NOT: __riscv_zicntr {{.*$}}
// CHECK-NOT: __riscv_zicsr {{.*$}}
// CHECK-NOT: __riscv_zifencei {{.*$}}
// CHECK-NOT: __riscv_zihintntl {{.*$}}
// CHECK-NOT: __riscv_zihintpause {{.*$}}
// CHECK-NOT: __riscv_zihpm {{.*$}}
// CHECK-NOT: __riscv_zk {{.*$}}
// CHECK-NOT: __riscv_zkn {{.*$}}
// CHECK-NOT: __riscv_zknd {{.*$}}
// CHECK-NOT: __riscv_zkne {{.*$}}
// CHECK-NOT: __riscv_zknh {{.*$}}
// CHECK-NOT: __riscv_zkr {{.*$}}
// CHECK-NOT: __riscv_zks {{.*$}}
// CHECK-NOT: __riscv_zksed {{.*$}}
// CHECK-NOT: __riscv_zksh {{.*$}}
// CHECK-NOT: __riscv_zkt {{.*$}}
// CHECK-NOT: __riscv_zmmul {{.*$}}
// CHECK-NOT: __riscv_zve32f {{.*$}}
// CHECK-NOT: __riscv_zve32x {{.*$}}
// CHECK-NOT: __riscv_zve64d {{.*$}}
// CHECK-NOT: __riscv_zve64f {{.*$}}
// CHECK-NOT: __riscv_zve64x {{.*$}}
// CHECK-NOT: __riscv_zvfh {{.*$}}
// CHECK-NOT: __riscv_zvl32b {{.*$}}
// CHECK-NOT: __riscv_zvl64b {{.*$}}
// CHECK-NOT: __riscv_zvl128b {{.*$}}
// CHECK-NOT: __riscv_zvl256b {{.*$}}
// CHECK-NOT: __riscv_zvl512b {{.*$}}
// CHECK-NOT: __riscv_zvl1024b {{.*$}}
// CHECK-NOT: __riscv_zvl2048b {{.*$}}
// CHECK-NOT: __riscv_zvl4096b {{.*$}}
// CHECK-NOT: __riscv_zvl8192b {{.*$}}
// CHECK-NOT: __riscv_zvl16384b {{.*$}}
// CHECK-NOT: __riscv_zvl32768b {{.*$}}
// CHECK-NOT: __riscv_zvl65536b {{.*$}}

// Experimental extensions

// CHECK-NOT: __riscv_smaia {{.*$}}
// CHECK-NOT: __riscv_ssaia {{.*$}}
// CHECK-NOT: __riscv_zacas {{.*$}}
// CHECK-NOT: __riscv_zfa {{.*$}}
// CHECK-NOT: __riscv_zfbfmin {{.*$}}
// CHECK-NOT: __riscv_zicfilp {{.*$}}
// CHECK-NOT: __riscv_zicfiss {{.*$}}
// CHECK-NOT: __riscv_zicond {{.*$}}
// CHECK-NOT: __riscv_zimop {{.*$}}
// CHECK-NOT: __riscv_zcmop {{.*$}}
// CHECK-NOT: __riscv_ztso {{.*$}}
// CHECK-NOT: __riscv_zvbb {{.*$}}
// CHECK-NOT: __riscv_zvbc {{.*$}}
// CHECK-NOT: __riscv_zvfbfmin {{.*$}}
// CHECK-NOT: __riscv_zvfbfwma {{.*$}}
// CHECK-NOT: __riscv_zvkg {{.*$}}
// CHECK-NOT: __riscv_zvkn {{.*$}}
// CHECK-NOT: __riscv_zvknc {{.*$}}
// CHECK-NOT: __riscv_zvkned {{.*$}}
// CHECK-NOT: __riscv_zvkng {{.*$}}
// CHECK-NOT: __riscv_zvknha {{.*$}}
// CHECK-NOT: __riscv_zvknhb {{.*$}}
// CHECK-NOT: __riscv_zvks {{.*$}}
// CHECK-NOT: __riscv_zvksc {{.*$}}
// CHECK-NOT: __riscv_zvksed {{.*$}}
// CHECK-NOT: __riscv_zvksg {{.*$}}
// CHECK-NOT: __riscv_zvksh {{.*$}}
// CHECK-NOT: __riscv_zvkt {{.*$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ia -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-A-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ia -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-A-EXT %s
// CHECK-A-EXT: __riscv_a 2001000{{$}}
// CHECK-A-EXT: __riscv_atomic 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ic -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-C-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ic -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-C-EXT %s
// CHECK-C-EXT: __riscv_c 2000000{{$}}
// CHECK-C-EXT: __riscv_compressed 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-D-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-D-EXT %s
// CHECK-D-EXT: __riscv_d 2002000{{$}}
// CHECK-D-EXT: __riscv_fdiv 1
// CHECK-D-EXT: __riscv_flen 64
// CHECK-D-EXT: __riscv_fsqrt 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32if -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-F-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64if -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-F-EXT %s
// CHECK-F-EXT: __riscv_f 2002000{{$}}
// CHECK-F-EXT: __riscv_fdiv 1
// CHECK-F-EXT: __riscv_flen 32
// CHECK-F-EXT: __riscv_fsqrt 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ifd -mabi=ilp32 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SOFT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ifd -mabi=lp64 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SOFT %s
// CHECK-SOFT: __riscv_float_abi_soft 1
// CHECK-SOFT-NOT: __riscv_float_abi_single
// CHECK-SOFT-NOT: __riscv_float_abi_double

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ifd -mabi=ilp32f -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SINGLE %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ifd -mabi=lp64f -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SINGLE %s
// CHECK-SINGLE: __riscv_float_abi_single 1
// CHECK-SINGLE-NOT: __riscv_float_abi_soft
// CHECK-SINGLE-NOT: __riscv_float_abi_double

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-DOUBLE %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ifd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-DOUBLE %s
// CHECK-DOUBLE: __riscv_float_abi_double 1
// CHECK-DOUBLE-NOT: __riscv_float_abi_soft
// CHECK-DOUBLE-NOT: __riscv_float_abi_single

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ih -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-H-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ih -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-H-EXT %s
// CHECK-H-EXT: __riscv_h 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32i -x c -E -dM %s \
// RUN: -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64i -x c -E -dM %s \
// RUN: -o - | FileCheck %s
// CHECK: __riscv_i 2001000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32im -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-M-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64im -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-M-EXT %s
// CHECK-M-EXT: __riscv_div 1
// CHECK-M-EXT: __riscv_m 2000000{{$}}
// CHECK-M-EXT: __riscv_mul 1
// CHECK-M-EXT: __riscv_muldiv 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32isvinval -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SVINVAL-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64isvinval -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SVINVAL-EXT %s
// CHECK-SVINVAL-EXT: __riscv_svinval 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32isvnapot -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SVNAPOT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64isvnapot -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SVNAPOT-EXT %s
// CHECK-SVNAPOT-EXT: __riscv_svnapot 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32isvpbmt -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SVPBMT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64isvpbmt -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SVPBMT-EXT %s
// CHECK-SVPBMT-EXT: __riscv_svpbmt 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-V-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-V-EXT %s
// CHECK-V-EXT: __riscv_v 1000000{{$}}
// CHECK-V-EXT: __riscv_v_elen 64
// CHECK-V-EXT: __riscv_v_elen_fp 64
// CHECK-V-EXT: __riscv_v_intrinsic 12000{{$}}
// CHECK-V-EXT: __riscv_v_min_vlen 128
// CHECK-V-EXT: __riscv_vector 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixcvalu -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVALU-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixcvalu -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVALU-EXT %s
// CHECK-XCVALU-EXT: __riscv_xcvalu 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixcvbi -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVBI-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixcvbi -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVBI-EXT %s
// CHECK-XCVBI-EXT: __riscv_xcvbi 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixcvbitmanip -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVBITMANIP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixcvbitmanip -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVBITMANIP-EXT %s
// CHECK-XCVBITMANIP-EXT: __riscv_xcvbitmanip 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixcvmac -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVMAC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixcvmac -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVMAC-EXT %s
// CHECK-XCVMAC-EXT: __riscv_xcvmac 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixcvsimd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVSIMD-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixcvsimd -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XCVSIMD-EXT %s
// CHECK-XCVSIMD-EXT: __riscv_xcvsimd 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixsfvcp -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVCP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixsfvcp -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVCP-EXT %s
// CHECK-XSFVCP-EXT: __riscv_xsfvcp 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixsfvfnrclipxfqf -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVFNRCLIPXFQF-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixsfvfnrclipxfqf -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVFNRCLIPXFQF-EXT %s
// CHECK-XSFVFNRCLIPXFQF-EXT: __riscv_xsfvfnrclipxfqf 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixsfvfwmaccqqq -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVFWMACCQQQ-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixsfvfwmaccqqq -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVFWMACCQQQ-EXT %s
// CHECK-XSFVFWMACCQQQ-EXT: __riscv_xsfvfwmaccqqq 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixsfvqmaccdod -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVQMACCDOD-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixsfvqmaccdod -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVQMACCDOD-EXT %s
// CHECK-XSFVQMACCDOD-EXT: __riscv_xsfvqmaccdod 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixsfvqmaccqoq -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVQMACCQOQ-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixsfvqmaccqoq -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XSFVQMACCQOQ-EXT %s
// CHECK-XSFVQMACCQOQ-EXT: __riscv_xsfvqmaccqoq 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadba -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADBA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadba -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADBA-EXT %s
// CHECK-XTHEADBA-EXT: __riscv_xtheadba 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadbb -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADBB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadbb -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADBB-EXT %s
// CHECK-XTHEADBB-EXT: __riscv_xtheadbb 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadbs -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADBS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadbs -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADBS-EXT %s
// CHECK-XTHEADBS-EXT: __riscv_xtheadbs 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadcmo -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADCMO-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadcmo -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADCMO-EXT %s
// CHECK-XTHEADCMO-EXT: __riscv_xtheadcmo 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadcondmov -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADCONDMOV-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadcondmov -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADCONDMOV-EXT %s
// CHECK-XTHEADCONDMOV-EXT: __riscv_xtheadcondmov 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadfmemidx -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADFMEMIDX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadfmemidx -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADFMEMIDX-EXT %s
// CHECK-XTHEADFMEMIDX-EXT: __riscv_xtheadfmemidx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadmac -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADMAC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadmac -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADMAC-EXT %s
// CHECK-XTHEADMAC-EXT: __riscv_xtheadmac 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadmemidx -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADMEMIDX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadmemidx -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADMEMIDX-EXT %s
// CHECK-XTHEADMEMIDX-EXT: __riscv_xtheadmemidx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadmempair -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADMEMPAIR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadmempair -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADMEMPAIR-EXT %s
// CHECK-XTHEADMEMPAIR-EXT: __riscv_xtheadmempair 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadsync -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADSYNC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadsync -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADSYNC-EXT %s
// CHECK-XTHEADSYNC-EXT: __riscv_xtheadsync 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixtheadvdot -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADVDOT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixtheadvdot -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XTHEADVDOT-EXT %s
// CHECK-XTHEADVDOT-EXT: __riscv_xtheadvdot 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ixventanacondops -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XVENTANACONDOPS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ixventanacondops -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-XVENTANACONDOPS-EXT %s
// CHECK-XVENTANACONDOPS-EXT: __riscv_xventanacondops 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izawrs -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZAWRS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izawrs -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZAWRS-EXT %s
// CHECK-ZAWRS-EXT: __riscv_zawrs 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izba1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izba -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izba1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izba -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// CHECK-ZBA-EXT: __riscv_zba 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbb -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbb -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// CHECK-ZBB-EXT: __riscv_zbb 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbc -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbc -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// CHECK-ZBC-EXT: __riscv_zbc 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbkb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBKB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbkb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBKB-EXT %s
// CHECK-ZBKB-EXT: __riscv_zbkb

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbkc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBKC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbkc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBKC-EXT %s
// CHECK-ZBKC-EXT: __riscv_zbkc

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbkx1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBKX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbkx1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBKX-EXT %s
// CHECK-ZBKX-EXT: __riscv_zbkx

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbs1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izbs -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbs1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izbs -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// CHECK-ZBS-EXT: __riscv_zbs 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izca1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izca1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCA-EXT %s
// CHECK-ZCA-EXT: __riscv_zca 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izcb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izcb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCB-EXT %s
// CHECK-ZCB-EXT: __riscv_zca 1000000{{$}}
// CHECK-ZCB-EXT: __riscv_zcb 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izcd1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCD-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izcd1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCD-EXT %s
// CHECK-ZCD-EXT: __riscv_zcd 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izce1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izce1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCE-EXT %s
// CHECK-ZCE-EXT: __riscv_zce 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izcf1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCF-EXT %s
// CHECK-ZCF-EXT: __riscv_zcf 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izcmp1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCMP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izcmp1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCMP-EXT %s
// CHECK-ZCMP-EXT: __riscv_zca 1000000{{$}}
// CHECK-ZCMP-EXT: __riscv_zcmp 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izcmt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCMT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izcmt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCMT-EXT %s
// CHECK-ZCMT-EXT: __riscv_zca 1000000{{$}}
// CHECK-ZCMT-EXT: __riscv_zcmt 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izdinx1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZDINX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izdinx1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZDINX-EXT %s
// CHECK-ZDINX-EXT: __riscv_zdinx 1000000{{$}}
// CHECK-ZDINX-EXT: __riscv_zfinx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izfh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFH-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izfh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFH-EXT %s
// CHECK-ZFH-EXT: __riscv_f 2002000{{$}}
// CHECK-ZFH-EXT: __riscv_zfh 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izfhmin1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFHMIN-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izfhmin1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFHMIN-EXT %s
// CHECK-ZFHMIN-EXT: __riscv_f 2002000{{$}}
// CHECK-ZFHMIN-EXT: __riscv_zfhmin 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izfinx1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFINX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izfinx1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFINX-EXT %s
// CHECK-ZFINX-EXT: __riscv_zfinx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izhinx1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZHINX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izhinx1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZHINX-EXT %s
// CHECK-ZHINX-EXT: __riscv_zhinx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izhinxmin1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZHINXMIN-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64izhinxmin1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZHINXMIN-EXT %s
// CHECK-ZHINXMIN-EXT: __riscv_zhinxmin 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izicbom -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICBOM-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izicbom -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICBOM-EXT %s
// CHECK-ZICBOM-EXT: __riscv_zicbom 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izicbop -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICBOP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izicbop -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICBOP-EXT %s
// CHECK-ZICBOP-EXT: __riscv_zicbop 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izicboz -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICBOZ-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izicboz -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICBOZ-EXT %s
// CHECK-ZICBOZ-EXT: __riscv_zicboz 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izicntr -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICNTR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izicntr -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICNTR-EXT %s
// CHECK-ZICNTR-EXT: __riscv_zicntr 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izicsr2p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICSR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izicsr2p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICSR-EXT %s
// CHECK-ZICSR-EXT: __riscv_zicsr 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izifencei2p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIFENCEI-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izifencei2p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIFENCEI-EXT %s
// CHECK-ZIFENCEI-EXT: __riscv_zifencei 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izihintntl1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIHINTNTL-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izihintntl1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIHINTNTL-EXT %s
// CHECK-ZIHINTNTL-EXT: __riscv_zihintntl 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izihintpause2p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIHINTPAUSE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izihintpause2p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIHINTPAUSE-EXT %s
// CHECK-ZIHINTPAUSE-EXT: __riscv_zihintpause 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izihpm -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIHPM-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izihpm -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIHPM-EXT %s
// CHECK-ZIHPM-EXT: __riscv_zihpm 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izk1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZK-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izk1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZK-EXT %s
// CHECK-ZK-EXT: __riscv_zk

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32i_zkn_zkt_zkr -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZK %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64i_zkn_zkt_zkr -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZK %s
// CHECK-COMBINE-INTO-ZK: __riscv_zk 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izkn1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKN-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izkn1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKN-EXT %s
// CHECK-ZKN-EXT: __riscv_zkn

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32i_zbkb_zbkc_zbkx_zkne_zknd_zknh -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZKN %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64i_zbkb_zbkc_zbkx_zkne_zknd_zknh -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZKN %s
// CHECK-COMBINE-INTO-ZKN: __riscv_zkn 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izknd1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKND-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izknd1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKND-EXT %s
// CHECK-ZKND-EXT: __riscv_zknd

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izkne1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKNE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izkne1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKNE-EXT %s
// CHECK-ZKNE-EXT: __riscv_zkne

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izknh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKNH-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izknh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKNH-EXT %s
// CHECK-ZKNH-EXT: __riscv_zknh

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izkr1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izkr1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKR-EXT %s
// CHECK-ZKR-EXT: __riscv_zkr

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32i_zbkb_zbkc_zbkx_zksed_zksh -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZKS %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64i_zbkb_zbkc_zbkx_zksed_zksh -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZKS %s
// CHECK-COMBINE-INTO-ZKS: __riscv_zks 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izks1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izks1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKS-EXT %s
// CHECK-ZKS-EXT: __riscv_zks

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izksed1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKSED-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izksed1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKSED-EXT %s
// CHECK-ZKSED-EXT: __riscv_zksed

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izksh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKSH-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izksh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKSH-EXT %s
// CHECK-ZKSH-EXT: __riscv_zksh

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZKT-EXT %s
// CHECK-ZKT-EXT: __riscv_zkt

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izmmul1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZMMUL-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izmmul1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZMMUL-EXT %s
// CHECK-ZMMUL-EXT: __riscv_zmmul

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ifzve32f1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE32F-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ifzve32f1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE32F-EXT %s
// CHECK-ZVE32F-EXT: __riscv_v_elen 32
// CHECK-ZVE32F-EXT: __riscv_v_elen_fp 32
// CHECK-ZVE32F-EXT: __riscv_v_intrinsic 12000{{$}}
// CHECK-ZVE32F-EXT: __riscv_v_min_vlen 32
// CHECK-ZVE32F-EXT: __riscv_vector 1
// CHECK-ZVE32F-EXT: __riscv_zve32f 1000000{{$}}
// CHECK-ZVE32F-EXT: __riscv_zve32x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izve32x1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE32X-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izve32x1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE32X-EXT %s
// CHECK-ZVE32X-EXT: __riscv_v_elen 32
// CHECK-ZVE32X-EXT: __riscv_v_elen_fp 0
// CHECK-ZVE32X-EXT: __riscv_v_intrinsic 12000{{$}}
// CHECK-ZVE32X-EXT: __riscv_v_min_vlen 32
// CHECK-ZVE32X-EXT: __riscv_vector 1
// CHECK-ZVE32X-EXT: __riscv_zve32x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ifdzve64d1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE64D-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ifdzve64d1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE64D-EXT %s
// CHECK-ZVE64D-EXT: __riscv_v_elen 64
// CHECK-ZVE64D-EXT: __riscv_v_elen_fp 64
// CHECK-ZVE64D-EXT: __riscv_v_intrinsic 12000{{$}}
// CHECK-ZVE64D-EXT: __riscv_v_min_vlen 64
// CHECK-ZVE64D-EXT: __riscv_vector 1
// CHECK-ZVE64D-EXT: __riscv_zve32f 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_zve32x 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_zve64d 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_zve64f 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_zve64x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32ifzve64f1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE64F-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64ifzve64f1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE64F-EXT %s
// CHECK-ZVE64F-EXT: __riscv_v_elen 64
// CHECK-ZVE64F-EXT: __riscv_v_elen_fp 32
// CHECK-ZVE64F-EXT: __riscv_v_intrinsic 12000{{$}}
// CHECK-ZVE64F-EXT: __riscv_v_min_vlen 64
// CHECK-ZVE64F-EXT: __riscv_vector 1
// CHECK-ZVE64F-EXT: __riscv_zve32f 1000000{{$}}
// CHECK-ZVE64F-EXT: __riscv_zve32x 1000000{{$}}
// CHECK-ZVE64F-EXT: __riscv_zve64f 1000000{{$}}
// CHECK-ZVE64F-EXT: __riscv_zve64x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izve64x1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE64X-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izve64x1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVE64X-EXT %s
// CHECK-ZVE64X-EXT: __riscv_v_elen 64
// CHECK-ZVE64X-EXT: __riscv_v_elen_fp 0
// CHECK-ZVE64X-EXT: __riscv_v_intrinsic 12000{{$}}
// CHECK-ZVE64X-EXT: __riscv_v_min_vlen 64
// CHECK-ZVE64X-EXT: __riscv_vector 1
// CHECK-ZVE64X-EXT: __riscv_zve32x 1000000{{$}}
// CHECK-ZVE64X-EXT: __riscv_zve64x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izvfh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVFH-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izvfh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVFH-EXT %s
// CHECK-ZVFH-EXT: __riscv_zvfh

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izve32x1p0_zvl32b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL32b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izve32x1p0_zvl32b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL32b %s
// CHECK-ZVL32b: __riscv_v_min_vlen 32

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izve32x1p0_zvl64b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL64b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izve32x1p0_zvl64b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL64b %s
// CHECK-ZVL64b: __riscv_v_min_vlen 64

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl128b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL128b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl128b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL128b %s
// CHECK-ZVL128b: __riscv_v_min_vlen 128

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl256b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL256b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl256b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL256b %s
// CHECK-ZVL256b: __riscv_v_min_vlen 256

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl512b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL512b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl512b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL512b %s
// CHECK-ZVL512b: __riscv_v_min_vlen 512

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl1024b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL1024b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl1024b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL1024b %s
// CHECK-ZVL1024b: __riscv_v_min_vlen 1024

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl2048b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL2048b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl2048b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL2048b %s
// CHECK-ZVL2048b: __riscv_v_min_vlen 2048

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl4096b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL4096b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl4096b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL4096b %s
// CHECK-ZVL4096b: __riscv_v_min_vlen 4096

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl8192b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL8192b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl8192b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL8192b %s
// CHECK-ZVL8192b: __riscv_v_min_vlen 8192

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl16384b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL16384b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl16384b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL16384b %s
// CHECK-ZVL16384b: __riscv_v_min_vlen 16384

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl32768b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL32768b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl32768b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL32768b %s
// CHECK-ZVL32768b: __riscv_v_min_vlen 32768

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32iv1p0_zvl65536b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL65536b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64iv1p0_zvl65536b1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVL65536b %s
// CHECK-ZVL65536b: __riscv_v_min_vlen 65536

// Experimental extensions

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32ismaia1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SMAIA-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64ismaia1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SMAIA-EXT %s
// CHECK-SMAIA-EXT: __riscv_smaia  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32issaia1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SSAIA-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64issaia1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-SSAIA-EXT %s
// CHECK-SSAIA-EXT: __riscv_ssaia  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zacas1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZACAS-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zacas1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZACAS-EXT %s
// CHECK-ZACAS-EXT: __riscv_zacas 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN: -march=rv32izfa -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN: -march=rv64izfa -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFA-EXT %s
// CHECK-ZFA-EXT: __riscv_zfa 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32izfbfmin0p8 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFBFMIN-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64izfbfmin0p8 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZFBFMIN-EXT %s
// CHECK-ZFBFMIN-EXT: __riscv_zfbfmin 8000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp0p4 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICFILP-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp0p4 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICFILP-EXT %s
// CHECK-ZICFILP-EXT: __riscv_zicfilp 4000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicond1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICOND-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicond1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICOND-EXT %s
// CHECK-ZICOND-EXT: __riscv_zicond  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zimop0p1 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIMOP-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zimop0p1 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZIMOP-EXT %s
// CHECK-ZIMOP-EXT: __riscv_zimop  1000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zcmop0p2 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCMOP-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zcmop0p2 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZCMOP-EXT %s
// CHECK-ZCMOP-EXT: __riscv_zcmop  2000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv32iztso0p1 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZTSO-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -menable-experimental-extensions \
// RUN: -march=rv64iztso0p1 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZTSO-EXT %s
// CHECK-ZTSO-EXT: __riscv_ztso 1000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvbb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVBB-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvbb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVBB-EXT %s
// CHECK-ZVBB-EXT: __riscv_zvbb  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvbc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVBC-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvbc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVBC-EXT %s
// CHECK-ZVBC-EXT: __riscv_zvbc  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32ifzvfbfmin0p8 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVFBFMIN-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64ifzvfbfmin0p8 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVFBFMIN-EXT %s
// CHECK-ZVFBFMIN-EXT: __riscv_zvfbfmin 8000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32ifzvfbfwma0p8 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVFBFWMA-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64ifzvfbfwma0p8 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVFBFWMA-EXT %s
// CHECK-ZVFBFWMA-EXT: __riscv_zvfbfwma 8000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve32x_zvkg1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKG-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve32x_zvkg1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKG-EXT %s
// CHECK-ZVKG-EXT: __riscv_zvkg  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvkn1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKN-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvkn1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKN-EXT %s
// CHECK-ZVKN-EXT: __riscv_zvkn 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32iv_zvbb1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKN %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64iv_zvbb1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKN %s
// CHECK-COMBINE-INTO-ZVKN: __riscv_zvkn 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvknc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNC-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvknc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNC-EXT %s
// CHECK-ZVKNC-EXT: __riscv_zvknc 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32iv_zvbb1p0_zvbc1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKNC %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64iv_zvbb1p0_zvbc1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKNC %s
// CHECK-COMBINE-INTO-ZVKNC: __riscv_zvkn 1000000{{$}}
// CHECK-COMBINE-INTO-ZVKNC: __riscv_zvknc 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve32x_zvkned1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNED-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve32x_zvkned1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNED-EXT %s
// CHECK-ZVKNED-EXT: __riscv_zvkned 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvkng1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNG-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvkng1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNG-EXT %s
// CHECK-ZVKNG-EXT: __riscv_zvkng 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32iv_zvbb1p0_zvkg1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKNG %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64iv_zvbb1p0_zvkg1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKNG %s
// CHECK-COMBINE-INTO-ZVKNG: __riscv_zvkn 1000000{{$}}
// CHECK-COMBINE-INTO-ZVKNG: __riscv_zvkng 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve32x_zvknha1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNHA-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve32x_zvknha1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNHA-EXT %s
// CHECK-ZVKNHA-EXT: __riscv_zvknha 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvknhb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNHB-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvknhb1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKNHB-EXT %s
// CHECK-ZVKNHB-EXT: __riscv_zvknhb  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvks1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKS-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvks1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKS-EXT %s
// CHECK-ZVKS-EXT: __riscv_zvks 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32iv_zvbb1p0_zvksed1p0_zvksh1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKS %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64iv_zvbb1p0_zvksed1p0_zvksh1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKS %s
// CHECK-COMBINE-INTO-ZVKS: __riscv_zvks 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvksc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKSC-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvksc1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKSC-EXT %s
// CHECK-ZVKSC-EXT: __riscv_zvksc 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32iv_zvbb1p0_zvbc1p0_zvksed1p0_zvksh1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKSC %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64iv_zvbb1p0_zvbc1p0_zvksed1p0_zvksh1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKSC %s
// CHECK-COMBINE-INTO-ZVKSC: __riscv_zvks 1000000{{$}}
// CHECK-COMBINE-INTO-ZVKSC: __riscv_zvksc 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve32x_zvksed1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKSED-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve32x_zvksed1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKSED-EXT %s
// CHECK-ZVKSED-EXT: __riscv_zvksed  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve64x_zvksg1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKSG-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve64x_zvksg1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKSG-EXT %s
// CHECK-ZVKSG-EXT: __riscv_zvksg 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32iv_zvbb1p0_zvkg1p0_zvksed1p0_zvksh1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKSG %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64iv_zvbb1p0_zvkg1p0_zvksed1p0_zvksh1p0_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKSG %s
// CHECK-COMBINE-INTO-ZVKSG: __riscv_zvks 1000000{{$}}
// CHECK-COMBINE-INTO-ZVKSG: __riscv_zvksg 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve32x_zvksh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKSH-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve32x_zvksh1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKSH-EXT %s
// CHECK-ZVKSH-EXT: __riscv_zvksh  1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zve32x_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKT-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zve32x_zvkt1p0 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZVKT-EXT %s
// CHECK-ZVKT-EXT: __riscv_zvkt 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -x c -E -dM %s \
// RUN: -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-AVOID
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -x c -E -dM %s \
// RUN: -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-AVOID
// CHECK-MISALIGNED-AVOID: __riscv_misaligned_avoid 1

// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -E -dM %s \
// RUN:   -munaligned-access -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-FAST
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -E -dM %s \
// RUN:   -munaligned-access -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-FAST
// CHECK-MISALIGNED-FAST: __riscv_misaligned_fast 1

// RUN: %clang -target riscv32 -menable-experimental-extensions \
// RUN: -march=rv32izicfiss0p4 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICFISS-EXT %s
// RUN: %clang -target riscv64 -menable-experimental-extensions \
// RUN: -march=rv64izicfiss0p4 -x c -E -dM %s \
// RUN: -o - | FileCheck --check-prefix=CHECK-ZICFISS-EXT %s
// CHECK-ZICFISS-EXT: __riscv_zicfiss 4000{{$}}
