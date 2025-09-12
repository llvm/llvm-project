// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -E -dM %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -E -dM %s \
// RUN:   -o - | FileCheck %s

// CHECK-NOT: __riscv_32e {{.*$}}
// CHECK-NOT: __riscv_64e {{.*$}}
// CHECK-NOT: __riscv_a {{.*$}}
// CHECK-NOT: __riscv_atomic
// CHECK-NOT: __riscv_b {{.*$}}
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
// CHECK-NOT: __riscv_q {{.*$}}
// CHECK-NOT: __riscv_sha {{.*$}}
// CHECK-NOT: __riscv_shcounterenw {{.*$}}
// CHECK-NOT: __riscv_shgatpa {{.*$}}
// CHECK-NOT: __riscv_shlcofideleg {{.*$}}
// CHECK-NOT: __riscv_shtvala {{.*$}}
// CHECK-NOT: __riscv_shvsatpa {{.*$}}
// CHECK-NOT: __riscv_shvstvala {{.*$}}
// CHECK-NOT: __riscv_shvstvecd {{.*$}}
// CHECK-NOT: __riscv_smaia {{.*$}}
// CHECK-NOT: __riscv_smcdeleg {{.*$}}
// CHECK-NOT: __riscv_smcntrpmf {{.*$}}
// CHECK-NOT: __riscv_smcsrind {{.*$}}
// CHECK-NOT: __riscv_smdbltrp {{.*$}}
// CHECK-NOT: __riscv_smepmp {{.*$}}
// CHECK-NOT: __riscv_smrnmi {{.*$}}
// CHECK-NOT: __riscv_smstateen {{.*$}}
// CHECK-NOT: __riscv_ssaia {{.*$}}
// CHECK-NOT: __riscv_ssccfg {{.*$}}
// CHECK-NOT: __riscv_ssccptr {{.*$}}
// CHECK-NOT: __riscv_sscofpmf {{.*$}}
// CHECK-NOT: __riscv_sscounterenw {{.*$}}
// CHECK-NOT: __riscv_sscsrind {{.*$}}
// CHECK-NOT: __riscv_ssdbltrp {{.*$}}
// CHECK-NOT: __riscv_ssqosid{{.*$}}
// CHECK-NOT: __riscv_ssstateen {{.*$}}
// CHECK-NOT: __riscv_ssstrict {{.*$}}
// CHECK-NOT: __riscv_sstc {{.*$}}
// CHECK-NOT: __riscv_sstvala {{.*$}}
// CHECK-NOT: __riscv_sstvecd {{.*$}}
// CHECK-NOT: __riscv_ssu64xl {{.*$}}
// CHECK-NOT: __riscv_svade {{.*$}}
// CHECK-NOT: __riscv_svadu {{.*$}}
// CHECK-NOT: __riscv_svbare {{.*$}}
// CHECK-NOT: __riscv_svinval {{.*$}}
// CHECK-NOT: __riscv_svnapot {{.*$}}
// CHECK-NOT: __riscv_svpbmt {{.*$}}
// CHECK-NOT: __riscv_svvptc {{.*$}}
// CHECK-NOT: __riscv_v {{.*$}}
// CHECK-NOT: __riscv_v_elen {{.*$}}
// CHECK-NOT: __riscv_v_elen_fp {{.*$}}
// CHECK-NOT: __riscv_v_intrinsic {{.*$}}
// CHECK-NOT: __riscv_v_min_vlen {{.*$}}
// CHECK-NOT: __riscv_vector {{.*$}}
// CHECK-NOT: __riscv_xventanacondops {{.*$}}
// CHECK-NOT: __riscv_za128rs {{.*$}}
// CHECK-NOT: __riscv_za64rs {{.*$}}
// CHECK-NOT: __riscv_zaamo {{.*$}}
// CHECK-NOT: __riscv_zabha {{.*$}}
// CHECK-NOT: __riscv_zacas {{.*$}}
// CHECK-NOT: __riscv_zalrsc {{.*$}}
// CHECK-NOT: __riscv_zama16b {{.*$}}
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
// CHECK-NOT: __riscv_zclsd {{.*$}}
// CHECK-NOT: __riscv_zcmop {{.*$}}
// CHECK-NOT: __riscv_zcmp {{.*$}}
// CHECK-NOT: __riscv_zcmt {{.*$}}
// CHECK-NOT: __riscv_zdinx {{.*$}}
// CHECK-NOT: __riscv_zfa {{.*$}}
// CHECK-NOT: __riscv_zfh {{.*$}}
// CHECK-NOT: __riscv_zfhmin {{.*$}}
// CHECK-NOT: __riscv_zfinx {{.*$}}
// CHECK-NOT: __riscv_zhinx {{.*$}}
// CHECK-NOT: __riscv_zhinxmin {{.*$}}
// CHECK-NOT: __riscv_zibi {{.*$}}
// CHECK-NOT: __riscv_zic64b {{.*$}}
// CHECK-NOT: __riscv_zicbom {{.*$}}
// CHECK-NOT: __riscv_zicbop {{.*$}}
// CHECK-NOT: __riscv_zicboz {{.*$}}
// CHECK-NOT: __riscv_ziccamoa {{.*$}}
// CHECK-NOT: __riscv_ziccamoc {{.*$}}
// CHECK-NOT: __riscv_ziccif {{.*$}}
// CHECK-NOT: __riscv_zicclsm {{.*$}}
// CHECK-NOT: __riscv_ziccrse {{.*$}}
// CHECK-NOT: __riscv_zicntr {{.*$}}
// CHECK-NOT: __riscv_zicond {{.*$}}
// CHECK-NOT: __riscv_zicsr {{.*$}}
// CHECK-NOT: __riscv_zifencei {{.*$}}
// CHECK-NOT: __riscv_zihintntl {{.*$}}
// CHECK-NOT: __riscv_zihintpause {{.*$}}
// CHECK-NOT: __riscv_zihpm {{.*$}}
// CHECK-NOT: __riscv_zilsd {{.*$}}
// CHECK-NOT: __riscv_zimop {{.*$}}
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
// CHECK-NOT: __riscv_zvbb {{.*$}}
// CHECK-NOT: __riscv_zvbc {{.*$}}
// CHECK-NOT: __riscv_zve32f {{.*$}}
// CHECK-NOT: __riscv_zve32x {{.*$}}
// CHECK-NOT: __riscv_zve64d {{.*$}}
// CHECK-NOT: __riscv_zve64f {{.*$}}
// CHECK-NOT: __riscv_zve64x {{.*$}}
// CHECK-NOT: __riscv_zvfh {{.*$}}
// CHECK-NOT: __riscv_zvkb {{.*$}}
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

// CHECK-NOT: __riscv_sdext{{.*$}}
// CHECK-NOT: __riscv_sdtrig{{.*$}}
// CHECK-NOT: __riscv_smctr{{.*$}}
// CHECK-NOT: __riscv_smmpm{{.*$}}
// CHECK-NOT: __riscv_smnpm{{.*$}}
// CHECK-NOT: __riscv_ssctr{{.*$}}
// CHECK-NOT: __riscv_ssnpm{{.*$}}
// CHECK-NOT: __riscv_sspm{{.*$}}
// CHECK-NOT: __riscv_supm{{.*$}}
// CHECK-NOT: __riscv_zalasr {{.*$}}
// CHECK-NOT: __riscv_zfbfmin {{.*$}}
// CHECK-NOT: __riscv_zicfilp {{.*$}}
// CHECK-NOT: __riscv_zicfiss {{.*$}}
// CHECK-NOT: __riscv_ztso {{.*$}}
// CHECK-NOT: __riscv_zvbc32e {{.*$}}
// CHECK-NOT: __riscv_zvfbfa {{.*$}}
// CHECK-NOT: __riscv_zvfbfmin {{.*$}}
// CHECK-NOT: __riscv_zvfbfwma {{.*$}}
// CHECK-NOT: __riscv_zvkgs {{.*$}}
// CHECK-NOT: __riscv_zvqdotq {{.*$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ia -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-A-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ia -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-A-EXT %s
// CHECK-A-EXT: __riscv_a 2001000{{$}}
// CHECK-A-EXT: __riscv_atomic 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ib -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-B-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ib -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-B-EXT %s
// CHECK-B-EXT: __riscv_b 1000000{{$}}
// CHECK-B-EXT: __riscv_zba 1000000{{$}}
// CHECK-B-EXT: __riscv_zbb 1000000{{$}}
// CHECK-B-EXT: __riscv_zbs 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ic -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-C-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ic -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-C-EXT %s
// CHECK-C-EXT: __riscv_c 2000000{{$}}
// CHECK-C-EXT: __riscv_compressed 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ifd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-D-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ifd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-D-EXT %s
// CHECK-D-EXT: __riscv_d 2002000{{$}}
// CHECK-D-EXT: __riscv_fdiv 1
// CHECK-D-EXT: __riscv_flen 64
// CHECK-D-EXT: __riscv_fsqrt 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefixes=CHECK-E-EXT,CHECK-RV32E %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefixes=CHECK-E-EXT,CHECK-RV64E %s
// CHECK-RV32E: __riscv_32e 1
// CHECK-RV64E: __riscv_64e 1
// CHECK-E-EXT: __riscv_abi_rve 1
// CHECK-E-EXT: __riscv_e 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32if -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-F-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64if -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-F-EXT %s
// CHECK-F-EXT: __riscv_f 2002000{{$}}
// CHECK-F-EXT: __riscv_fdiv 1
// CHECK-F-EXT: __riscv_flen 32
// CHECK-F-EXT: __riscv_fsqrt 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ifd -mabi=ilp32 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SOFT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ifd -mabi=lp64 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SOFT %s
// CHECK-SOFT: __riscv_float_abi_soft 1
// CHECK-SOFT-NOT: __riscv_float_abi_single
// CHECK-SOFT-NOT: __riscv_float_abi_double

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ifd -mabi=ilp32f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SINGLE %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ifd -mabi=lp64f -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SINGLE %s
// CHECK-SINGLE: __riscv_float_abi_single 1
// CHECK-SINGLE-NOT: __riscv_float_abi_soft
// CHECK-SINGLE-NOT: __riscv_float_abi_double

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ifd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-DOUBLE %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ifd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-DOUBLE %s
// CHECK-DOUBLE: __riscv_float_abi_double 1
// CHECK-DOUBLE-NOT: __riscv_float_abi_soft
// CHECK-DOUBLE-NOT: __riscv_float_abi_single

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i -mabi=ilp32e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ILP32E %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64i -mabi=lp64e -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-LP64E %s
// CHECK-ILP32E: __riscv_abi_rve 1
// CHECK-LP64E: __riscv_abi_rve 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ih -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-H-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ih -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-H-EXT %s
// CHECK-H-EXT: __riscv_h 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i -E -dM %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64i -E -dM %s \
// RUN:   -o - | FileCheck %s
// CHECK: __riscv_i 2001000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32im -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-M-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64im -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-M-EXT %s
// CHECK-M-EXT: __riscv_div 1
// CHECK-M-EXT: __riscv_m 2000000{{$}}
// CHECK-M-EXT: __riscv_mul 1
// CHECK-M-EXT: __riscv_muldiv 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ifdq -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-Q-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ifdq -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-Q-EXT %s
// CHECK-Q-EXT: __riscv_fdiv 1
// CHECK-Q-EXT: __riscv_flen 128
// CHECK-Q-EXT: __riscv_fsqrt 1
// CHECK-Q-EXT: __riscv_q 2002000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isha -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHCOUNTERENW-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isha -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHCOUNTERENW-EXT %s
// CHECK-SHA-EXT: __riscv_sha 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ishcounterenw -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHCOUNTERENW-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ishcounterenw -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHCOUNTERENW-EXT %s
// CHECK-SHCOUNTERENW-EXT: __riscv_shcounterenw 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ishgatpa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHGATPA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ishgatpa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHGATPA-EXT %s
// CHECK-SHGATPA-EXT: __riscv_shgatpa 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ishlcofideleg -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHLCOFIDELEG-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ishlcofideleg -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHLCOFIDELEG-EXT %s
// CHECK-SHLCOFIDELEG-EXT: __riscv_shlcofideleg 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ishtvala -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHTVALA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ishtvala -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHTVALA-EXT %s
// CHECK-SHTVALA-EXT: __riscv_shtvala 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ishvsatpa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHVSATPA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ishvsatpa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHVSATPA-EXT %s
// CHECK-SHVSATPA-EXT: __riscv_shvsatpa 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ishvstvala -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHVSTVALA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ishvstvala -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHVSTVALA-EXT %s
// CHECK-SHVSTVALA-EXT: __riscv_shvstvala 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ishvstvecd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHVSTVECD-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ishvstvecd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SHVSTVECD-EXT %s
// CHECK-SHVSTVECD-EXT: __riscv_shvstvecd 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32issccfg -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCCFG-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64issccfg -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCCFG-EXT %s
// CHECK-SSCCFG-EXT: __riscv_ssccfg 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32issccptr -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCCPTR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64issccptr -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCCPTR-EXT %s
// CHECK-SSCCPTR-EXT: __riscv_ssccptr 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isscofpmf -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCOFPMF-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isscofpmf -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCOFPMF-EXT %s
// CHECK-SSCOFPMF-EXT: __riscv_sscofpmf 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isscounterenw -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCOUNTERENW-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isscounterenw -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCOUNTERENW-EXT %s
// CHECK-SSCOUNTERENW-EXT: __riscv_sscounterenw 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ismstateen -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMSTATEEN-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ismstateen -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMSTATEEN-EXT %s
// CHECK-SMSTATEEN-EXT: __riscv_smstateen 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32issstateen -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSSTATEEN-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64issstateen -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSSTATEEN-EXT %s
// CHECK-SSSTATEEN-EXT: __riscv_ssstateen 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32issstrict -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSSTRICT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64issstrict -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSSTRICT-EXT %s
// CHECK-SSSTRICT-EXT: __riscv_ssstrict 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isstc -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSTC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isstc -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSTC-EXT %s
// CHECK-SSTC-EXT: __riscv_sstc 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isstvala -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSTVALA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isstvala -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSTVALA-EXT %s
// CHECK-SSTVALA-EXT: __riscv_sstvala 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isstvecd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSTVECD-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isstvecd -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSTVECD-EXT %s
// CHECK-SSTVECD-EXT: __riscv_sstvecd 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32issu64xl -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSU64XL-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64issu64xl -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSU64XL-EXT %s
// CHECK-SSU64XL-EXT: __riscv_ssu64xl 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isvade -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVADE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isvade -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVADE-EXT %s
// CHECK-SVADE-EXT: __riscv_svade 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isvadu -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVADU-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isvadu -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVADU-EXT %s
// CHECK-SVADU-EXT: __riscv_svadu 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isvbare -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVBARE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isvbare -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVBARE-EXT %s
// CHECK-SVBARE-EXT: __riscv_svbare 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isvinval -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVINVAL-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isvinval -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVINVAL-EXT %s
// CHECK-SVINVAL-EXT: __riscv_svinval 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isvnapot -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVNAPOT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isvnapot -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVNAPOT-EXT %s
// CHECK-SVNAPOT-EXT: __riscv_svnapot 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isvpbmt -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVPBMT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isvpbmt -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVPBMT-EXT %s
// CHECK-SVPBMT-EXT: __riscv_svpbmt 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32isvvptc -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVVPTC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64isvvptc -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVVPTC-EXT %s
// CHECK-SVVPTC-EXT: __riscv_svvptc 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-V-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-V-EXT %s
// CHECK-V-EXT: __riscv_v 1000000{{$}}
// CHECK-V-EXT: __riscv_v_elen 64
// CHECK-V-EXT: __riscv_v_elen_fp 64
// CHECK-V-EXT: __riscv_v_intrinsic 1000000{{$}}
// CHECK-V-EXT: __riscv_v_min_vlen 128
// CHECK-V-EXT: __riscv_vector 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ixventanacondops -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XVENTANACONDOPS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ixventanacondops -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-XVENTANACONDOPS-EXT %s
// CHECK-XVENTANACONDOPS-EXT: __riscv_xventanacondops 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iza128rs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZA128RS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iza128rs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZA128RS-EXT %s
// CHECK-ZA128RS-EXT: __riscv_za128rs 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iza64rs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZA64RS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iza64rs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZA64RS-EXT %s
// CHECK-ZA64RS-EXT: __riscv_za64rs 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zaamo1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZAAMO-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zaamo1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZAAMO-EXT %s
// CHECK-ZAAMO-EXT: __riscv_zaamo 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ia_zabha1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZABHA-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ia_zabha1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZABHA-EXT %s
// CHECK-ZABHA-EXT: __riscv_zabha 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ia_zacas1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZACAS-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ia_zacas1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZACAS-EXT %s
// CHECK-ZACAS-EXT: __riscv_zacas 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zalrsc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZALRSC-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zalrsc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZALRSC-EXT %s
// CHECK-ZALRSC-EXT: __riscv_zalrsc 1000000{{$}}

// RUN: %clang --target=riscv32 -march=rv32izama16b -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZAMA16B-EXT %s
// RUN: %clang --target=riscv64 -march=rv64izama16b -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZAMA16B-EXT %s
// CHECK-ZAMA16B-EXT: __riscv_zama16b  1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izawrs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZAWRS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izawrs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZAWRS-EXT %s
// CHECK-ZAWRS-EXT: __riscv_zawrs 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izba1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izba -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izba1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izba -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBA-EXT %s
// CHECK-ZBA-EXT: __riscv_zba 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbb -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbb -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBB-EXT %s
// CHECK-ZBB-EXT: __riscv_zbb 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbc -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbc -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBC-EXT %s
// CHECK-ZBC-EXT: __riscv_zbc 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbkb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBKB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbkb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBKB-EXT %s
// CHECK-ZBKB-EXT: __riscv_zbkb

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbkc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBKC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbkc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBKC-EXT %s
// CHECK-ZBKC-EXT: __riscv_zbkc

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbkx1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBKX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbkx1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBKX-EXT %s
// CHECK-ZBKX-EXT: __riscv_zbkx

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbs1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izbs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbs1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izbs -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZBS-EXT %s
// CHECK-ZBS-EXT: __riscv_zbs 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izca1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izca1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCA-EXT %s
// CHECK-ZCA-EXT: __riscv_zca 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izcb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCB-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izcb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCB-EXT %s
// CHECK-ZCB-EXT: __riscv_zca 1000000{{$}}
// CHECK-ZCB-EXT: __riscv_zcb 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izcd1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCD-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izcd1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCD-EXT %s
// CHECK-ZCD-EXT: __riscv_zcd 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izce1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izce1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCE-EXT %s
// CHECK-ZCE-EXT: __riscv_zce 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izcf1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCF-EXT %s
// CHECK-ZCF-EXT: __riscv_zcf 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i_zclsd1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCLSD-EXT %s
// CHECK-ZCLSD-EXT: __riscv_zclsd  1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i_zcmop1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCMOP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64i_zcmop1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCMOP-EXT %s
// CHECK-ZCMOP-EXT: __riscv_zcmop  1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izcmp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCMP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izcmp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCMP-EXT %s
// CHECK-ZCMP-EXT: __riscv_zca 1000000{{$}}
// CHECK-ZCMP-EXT: __riscv_zcmp 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izcmt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCMT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izcmt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZCMT-EXT %s
// CHECK-ZCMT-EXT: __riscv_zca 1000000{{$}}
// CHECK-ZCMT-EXT: __riscv_zcmt 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izdinx1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZDINX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izdinx1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZDINX-EXT %s
// CHECK-ZDINX-EXT: __riscv_zdinx 1000000{{$}}
// CHECK-ZDINX-EXT: __riscv_zfinx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izfh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFH-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izfh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFH-EXT %s
// CHECK-ZFH-EXT: __riscv_f 2002000{{$}}
// CHECK-ZFH-EXT: __riscv_zfh 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izfhmin1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFHMIN-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izfhmin1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFHMIN-EXT %s
// CHECK-ZFHMIN-EXT: __riscv_f 2002000{{$}}
// CHECK-ZFHMIN-EXT: __riscv_zfhmin 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izfinx1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFINX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izfinx1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFINX-EXT %s
// CHECK-ZFINX-EXT: __riscv_zfinx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izhinx1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZHINX-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izhinx1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZHINX-EXT %s
// CHECK-ZHINX-EXT: __riscv_zhinx 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izhinxmin1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZHINXMIN-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64izhinxmin1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZHINXMIN-EXT %s
// CHECK-ZHINXMIN-EXT: __riscv_zhinxmin 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zibi0p1 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIBI-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zibi0p1 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIBI-EXT %s
// CHECK-ZIBI-EXT: __riscv_zibi

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izic64b -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIC64B-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izic64b -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIC64B-EXT %s
// CHECK-ZIC64B-EXT: __riscv_zic64b 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izicbom -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICBOM-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izicbom -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICBOM-EXT %s
// CHECK-ZICBOM-EXT: __riscv_zicbom 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izicbop -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICBOP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izicbop -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICBOP-EXT %s
// CHECK-ZICBOP-EXT: __riscv_zicbop 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izicboz -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICBOZ-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izicboz -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICBOZ-EXT %s
// CHECK-ZICBOZ-EXT: __riscv_zicboz 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iziccamoa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCAMOA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iziccamoa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCAMOA-EXT %s
// CHECK-ZICCAMOA-EXT: __riscv_ziccamoa 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iziccamoc -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCAMOC-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iziccamoc -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCAMOC-EXT %s
// CHECK-ZICCAMOC-EXT: __riscv_ziccamoc 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iziccif -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCIF-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iziccif -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCIF-EXT %s
// CHECK-ZICCIF-EXT: __riscv_ziccif 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izicclsm -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCLSM-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izicclsm -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCLSM-EXT %s
// CHECK-ZICCLSM-EXT: __riscv_zicclsm 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iziccrse -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCRSE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iziccrse -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICCRSE-EXT %s
// CHECK-ZICCRSE-EXT: __riscv_ziccrse 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izicntr -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICNTR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izicntr -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICNTR-EXT %s
// CHECK-ZICNTR-EXT: __riscv_zicntr 2000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zicond -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICOND-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zicond -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICOND-EXT %s
// CHECK-ZICOND-EXT: __riscv_zicond  1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izicsr2p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICSR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izicsr2p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICSR-EXT %s
// CHECK-ZICSR-EXT: __riscv_zicsr 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izifencei2p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIFENCEI-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izifencei2p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIFENCEI-EXT %s
// CHECK-ZIFENCEI-EXT: __riscv_zifencei 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izihintntl1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIHINTNTL-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izihintntl1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIHINTNTL-EXT %s
// CHECK-ZIHINTNTL-EXT: __riscv_zihintntl 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izihintpause2p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIHINTPAUSE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izihintpause2p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIHINTPAUSE-EXT %s
// CHECK-ZIHINTPAUSE-EXT: __riscv_zihintpause 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izihpm -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIHPM-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izihpm -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIHPM-EXT %s
// CHECK-ZIHPM-EXT: __riscv_zihpm 2000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i_zilsd1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZILSD-EXT %s
// CHECK-ZILSD-EXT: __riscv_zilsd  1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i_zimop1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIMOP-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64i_zimop1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZIMOP-EXT %s
// CHECK-ZIMOP-EXT: __riscv_zimop  1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izk1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZK-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izk1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZK-EXT %s
// CHECK-ZK-EXT: __riscv_zk

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i_zkn_zkt_zkr -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZK %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64i_zkn_zkt_zkr -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZK %s
// CHECK-COMBINE-INTO-ZK: __riscv_zk 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izkn1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKN-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izkn1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKN-EXT %s
// CHECK-ZKN-EXT: __riscv_zkn

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i_zbkb_zbkc_zbkx_zkne_zknd_zknh -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZKN %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64i_zbkb_zbkc_zbkx_zkne_zknd_zknh -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZKN %s
// CHECK-COMBINE-INTO-ZKN: __riscv_zkn 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izknd1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKND-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izknd1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKND-EXT %s
// CHECK-ZKND-EXT: __riscv_zknd

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izkne1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKNE-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izkne1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKNE-EXT %s
// CHECK-ZKNE-EXT: __riscv_zkne

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izknh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKNH-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izknh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKNH-EXT %s
// CHECK-ZKNH-EXT: __riscv_zknh

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izkr1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKR-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izkr1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKR-EXT %s
// CHECK-ZKR-EXT: __riscv_zkr

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32i_zbkb_zbkc_zbkx_zksed_zksh -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZKS %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64i_zbkb_zbkc_zbkx_zksed_zksh -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZKS %s
// CHECK-COMBINE-INTO-ZKS: __riscv_zks 1

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izks1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKS-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izks1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKS-EXT %s
// CHECK-ZKS-EXT: __riscv_zks

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izksed1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKSED-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izksed1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKSED-EXT %s
// CHECK-ZKSED-EXT: __riscv_zksed

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izksh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKSH-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izksh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKSH-EXT %s
// CHECK-ZKSH-EXT: __riscv_zksh

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKT-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZKT-EXT %s
// CHECK-ZKT-EXT: __riscv_zkt

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izmmul1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZMMUL-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izmmul1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZMMUL-EXT %s
// CHECK-ZMMUL-EXT: __riscv_zmmul

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ifzve32f1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE32F-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ifzve32f1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE32F-EXT %s
// CHECK-ZVE32F-EXT: __riscv_v_elen 32
// CHECK-ZVE32F-EXT: __riscv_v_elen_fp 32
// CHECK-ZVE32F-EXT: __riscv_v_intrinsic 1000000{{$}}
// CHECK-ZVE32F-EXT: __riscv_v_min_vlen 32
// CHECK-ZVE32F-EXT: __riscv_vector 1
// CHECK-ZVE32F-EXT: __riscv_zve32f 1000000{{$}}
// CHECK-ZVE32F-EXT: __riscv_zve32x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izve32x1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE32X-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izve32x1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE32X-EXT %s
// CHECK-ZVE32X-EXT: __riscv_v_elen 32
// CHECK-ZVE32X-EXT: __riscv_v_elen_fp 0
// CHECK-ZVE32X-EXT: __riscv_v_intrinsic 1000000{{$}}
// CHECK-ZVE32X-EXT: __riscv_v_min_vlen 32
// CHECK-ZVE32X-EXT: __riscv_vector 1
// CHECK-ZVE32X-EXT: __riscv_zve32x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ifdzve64d1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE64D-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ifdzve64d1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE64D-EXT %s
// CHECK-ZVE64D-EXT: __riscv_v_elen 64
// CHECK-ZVE64D-EXT: __riscv_v_elen_fp 64
// CHECK-ZVE64D-EXT: __riscv_v_intrinsic 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_v_min_vlen 64
// CHECK-ZVE64D-EXT: __riscv_vector 1
// CHECK-ZVE64D-EXT: __riscv_zve32f 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_zve32x 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_zve64d 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_zve64f 1000000{{$}}
// CHECK-ZVE64D-EXT: __riscv_zve64x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32ifzve64f1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE64F-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64ifzve64f1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE64F-EXT %s
// CHECK-ZVE64F-EXT: __riscv_v_elen 64
// CHECK-ZVE64F-EXT: __riscv_v_elen_fp 32
// CHECK-ZVE64F-EXT: __riscv_v_intrinsic 1000000{{$}}
// CHECK-ZVE64F-EXT: __riscv_v_min_vlen 64
// CHECK-ZVE64F-EXT: __riscv_vector 1
// CHECK-ZVE64F-EXT: __riscv_zve32f 1000000{{$}}
// CHECK-ZVE64F-EXT: __riscv_zve32x 1000000{{$}}
// CHECK-ZVE64F-EXT: __riscv_zve64f 1000000{{$}}
// CHECK-ZVE64F-EXT: __riscv_zve64x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izve64x1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE64X-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izve64x1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVE64X-EXT %s
// CHECK-ZVE64X-EXT: __riscv_v_elen 64
// CHECK-ZVE64X-EXT: __riscv_v_elen_fp 0
// CHECK-ZVE64X-EXT: __riscv_v_intrinsic 1000000{{$}}
// CHECK-ZVE64X-EXT: __riscv_v_min_vlen 64
// CHECK-ZVE64X-EXT: __riscv_vector 1
// CHECK-ZVE64X-EXT: __riscv_zve32x 1000000{{$}}
// CHECK-ZVE64X-EXT: __riscv_zve64x 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izvfh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVFH-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izvfh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVFH-EXT %s
// CHECK-ZVFH-EXT: __riscv_zvfh

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izve32x1p0_zvl32b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL32b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izve32x1p0_zvl32b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL32b %s
// CHECK-ZVL32b: __riscv_v_min_vlen 32

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izve32x1p0_zvl64b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL64b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izve32x1p0_zvl64b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL64b %s
// CHECK-ZVL64b: __riscv_v_min_vlen 64

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl128b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL128b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl128b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL128b %s
// CHECK-ZVL128b: __riscv_v_min_vlen 128

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl256b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL256b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl256b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL256b %s
// CHECK-ZVL256b: __riscv_v_min_vlen 256

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl512b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL512b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl512b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL512b %s
// CHECK-ZVL512b: __riscv_v_min_vlen 512

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl1024b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL1024b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl1024b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL1024b %s
// CHECK-ZVL1024b: __riscv_v_min_vlen 1024

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl2048b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL2048b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl2048b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL2048b %s
// CHECK-ZVL2048b: __riscv_v_min_vlen 2048

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl4096b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL4096b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl4096b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL4096b %s
// CHECK-ZVL4096b: __riscv_v_min_vlen 4096

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl8192b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL8192b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl8192b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL8192b %s
// CHECK-ZVL8192b: __riscv_v_min_vlen 8192

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl16384b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL16384b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl16384b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL16384b %s
// CHECK-ZVL16384b: __riscv_v_min_vlen 16384

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl32768b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL32768b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl32768b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL32768b %s
// CHECK-ZVL32768b: __riscv_v_min_vlen 32768

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iv1p0_zvl65536b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL65536b %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iv1p0_zvl65536b1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVL65536b %s
// CHECK-ZVL65536b: __riscv_v_min_vlen 65536

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ismaia1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMAIA-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ismaia1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMAIA-EXT %s
// CHECK-SMAIA-EXT: __riscv_smaia  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32issaia1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSAIA-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64issaia1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSAIA-EXT %s
// CHECK-SSAIA-EXT: __riscv_ssaia  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ismcntrpmf1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMCNTRPMF-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ismcntrpmf1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMCNTRPMF-EXT %s
// CHECK-SMCNTRPMF-EXT: __riscv_smcntrpmf  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ismcsrind1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMCSRIND-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ismcsrind1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMCSRIND-EXT %s
// CHECK-SMCSRIND-EXT: __riscv_smcsrind  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32isscsrind1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCSRIND-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64isscsrind1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCSRIND-EXT %s
// CHECK-SSCSRIND-EXT: __riscv_sscsrind  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ismdbltrp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMDBLTRP-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ismdbltrp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMDBLTRP-EXT %s
// CHECK-SMDBLTRP-EXT: __riscv_smdbltrp  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32issdbltrp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSDBLTRP-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64issdbltrp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSDBLTRP-EXT %s
// CHECK-SSDBLTRP-EXT: __riscv_ssdbltrp  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_ssqosid1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSQOSID-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_ssqosid1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSQOSID-EXT %s
// CHECK-SSQOSID-EXT: __riscv_ssqosid 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ismcdeleg1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMCDELEG-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ismcdeleg1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMCDELEG-EXT %s
// CHECK-SMCDELEG-EXT: __riscv_smcdeleg  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ismepmp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMEPMP-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ismepmp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMEPMP-EXT %s
// CHECK-SMEPMP-EXT: __riscv_smepmp  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32ismrnmi1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMRNMI-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64ismrnmi1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMRNMI-EXT %s
// CHECK-SMRNMI-EXT: __riscv_smrnmi  1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32izfa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFA-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64izfa -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFA-EXT %s
// CHECK-ZFA-EXT: __riscv_zfa 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvbb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVBB-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvbb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVBB-EXT %s
// CHECK-ZVBB-EXT: __riscv_zvbb  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvbc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVBC-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvbc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVBC-EXT %s
// CHECK-ZVBC-EXT: __riscv_zvbc  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvkb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKB-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvkb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKB-EXT %s
// CHECK-ZVKB-EXT: __riscv_zvkb  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_zvkg1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKG-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_zvkg1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKG-EXT %s
// CHECK-ZVKG-EXT: __riscv_zvkg  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvkn1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKN-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvkn1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKN-EXT %s
// CHECK-ZVKN-EXT: __riscv_zvkn 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32iv_zvkb1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKN %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64iv_zvkb1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKN %s
// CHECK-COMBINE-INTO-ZVKN: __riscv_zvkn 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvknc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNC-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvknc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNC-EXT %s
// CHECK-ZVKNC-EXT: __riscv_zvknc 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32iv_zvkb1p0_zvbc1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKNC %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64iv_zvkb1p0_zvbc1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKNC %s
// CHECK-COMBINE-INTO-ZVKNC: __riscv_zvkn 1000000{{$}}
// CHECK-COMBINE-INTO-ZVKNC: __riscv_zvknc 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_zvkned1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNED-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_zvkned1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNED-EXT %s
// CHECK-ZVKNED-EXT: __riscv_zvkned 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvkng1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNG-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvkng1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNG-EXT %s
// CHECK-ZVKNG-EXT: __riscv_zvkng 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32iv_zvkb1p0_zvkg1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKNG %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64iv_zvkb1p0_zvkg1p0_zvkned1p0_zvknhb1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKNG %s
// CHECK-COMBINE-INTO-ZVKNG: __riscv_zvkn 1000000{{$}}
// CHECK-COMBINE-INTO-ZVKNG: __riscv_zvkng 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_zvknha1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNHA-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_zvknha1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNHA-EXT %s
// CHECK-ZVKNHA-EXT: __riscv_zvknha 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvknhb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNHB-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvknhb1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKNHB-EXT %s
// CHECK-ZVKNHB-EXT: __riscv_zvknhb  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvks1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKS-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvks1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKS-EXT %s
// CHECK-ZVKS-EXT: __riscv_zvks 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32iv_zvkb1p0_zvksed1p0_zvksh1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKS %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64iv_zvkb1p0_zvksed1p0_zvksh1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKS %s
// CHECK-COMBINE-INTO-ZVKS: __riscv_zvks 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvksc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKSC-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvksc1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKSC-EXT %s
// CHECK-ZVKSC-EXT: __riscv_zvksc 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32iv_zvkb1p0_zvbc1p0_zvksed1p0_zvksh1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKSC %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64iv_zvkb1p0_zvbc1p0_zvksed1p0_zvksh1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKSC %s
// CHECK-COMBINE-INTO-ZVKSC: __riscv_zvks 1000000{{$}}
// CHECK-COMBINE-INTO-ZVKSC: __riscv_zvksc 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_zvksed1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKSED-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_zvksed1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKSED-EXT %s
// CHECK-ZVKSED-EXT: __riscv_zvksed  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve64x_zvksg1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKSG-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve64x_zvksg1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKSG-EXT %s
// CHECK-ZVKSG-EXT: __riscv_zvksg 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32iv_zvkb1p0_zvkg1p0_zvksed1p0_zvksh1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKSG %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64iv_zvkb1p0_zvkg1p0_zvksed1p0_zvksh1p0_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-COMBINE-INTO-ZVKSG %s
// CHECK-COMBINE-INTO-ZVKSG: __riscv_zvks 1000000{{$}}
// CHECK-COMBINE-INTO-ZVKSG: __riscv_zvksg 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_zvksh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKSH-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_zvksh1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKSH-EXT %s
// CHECK-ZVKSH-EXT: __riscv_zvksh  1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_zve32x_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKT-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_zve32x_zvkt1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKT-EXT %s
// CHECK-ZVKT-EXT: __riscv_zvkt 1000000{{$}}

// Experimental extensions
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zalasr0p1 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZALASR-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zalasr0p1 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZALASR-EXT %s
// CHECK-ZALASR-EXT: __riscv_zalasr 1000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32izfbfmin1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFBFMIN-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64izfbfmin1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZFBFMIN-EXT %s
// CHECK-ZFBFMIN-EXT: __riscv_zfbfmin 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicfilp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICFILP-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zicfilp1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICFILP-EXT %s
// CHECK-ZICFILP-EXT: __riscv_zicfilp 1000000{{$}}

// RUN: %clang --target=riscv32-unknown-linux-gnu \
// RUN:   -march=rv32iztso1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZTSO-EXT %s
// RUN: %clang --target=riscv64-unknown-linux-gnu \
// RUN:   -march=rv64iztso1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZTSO-EXT %s
// CHECK-ZTSO-EXT: __riscv_ztso 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32ifzvfbfa0p1 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVFBFA-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64ifzvfbfa0p1 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVFBFA-EXT %s
// CHECK-ZVFBFA-EXT: __riscv_zvfbfa 1000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zve32x_zvbc32e0p7 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVBC32E-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zve32x_zvbc32e0p7 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVBC32E-EXT %s
// CHECK-ZVBC32E-EXT: __riscv_zvbc32e 7000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32ifzvfbfmin1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVFBFMIN-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64ifzvfbfmin1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVFBFMIN-EXT %s
// CHECK-ZVFBFMIN-EXT: __riscv_zvfbfmin 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32ifzvfbfwma1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVFBFWMA-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64ifzvfbfwma1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVFBFWMA-EXT %s
// CHECK-ZVFBFWMA-EXT: __riscv_zvfbfwma 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zve32x_zvkgs0p7 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKGS-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zve32x_zvkgs0p7 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVKGS-EXT %s
// CHECK-ZVKGS-EXT: __riscv_zvkgs 7000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zve32x_zvqdotq0p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVQDOTQ-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_zve32x_zvqdotq0p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZVQDOTQ-EXT %s
// CHECK-ZVQDOTQ-EXT: __riscv_zvqdotq 0{{$}}

// RUN: %clang -target riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32izicfiss1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICFISS-EXT %s
// RUN: %clang -target riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64izicfiss1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-ZICFISS-EXT %s
// CHECK-ZICFISS-EXT: __riscv_zicfiss 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_ssnpm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSNPM-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_ssnpm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSNPM-EXT %s
// CHECK-SSNPM-EXT: __riscv_ssnpm 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_smnpm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMNPM-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_smnpm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMNPM-EXT %s
// CHECK-SMNPM-EXT: __riscv_smnpm 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_smmpm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMMPM-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_smmpm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMMPM-EXT %s
// CHECK-SMMPM-EXT: __riscv_smmpm 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_sspm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSPM-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_sspm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSPM-EXT %s
// CHECK-SSPM-EXT: __riscv_sspm 1000000{{$}}

// RUN: %clang --target=riscv32 \
// RUN:   -march=rv32i_supm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SUPM-EXT %s
// RUN: %clang --target=riscv64 \
// RUN:   -march=rv64i_supm1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SUPM-EXT %s
// CHECK-SUPM-EXT: __riscv_supm 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_sdext1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SDEXT-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_sdext1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SDEXT-EXT %s
// CHECK-SDEXT-EXT: __riscv_sdext 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_sdtrig1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SDTRIG-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_sdtrig1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SDTRIG-EXT %s
// CHECK-SDTRIG-EXT: __riscv_sdtrig 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_smctr1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMCTR-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_smctr1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SMCTR-EXT %s
// CHECK-SMCTR-EXT: __riscv_smctr 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_ssctr1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCTR-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_ssctr1p0 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SSCTR-EXT %s
// CHECK-SSCTR-EXT: __riscv_ssctr 1000000{{$}}

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_svukte0p3 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVUKTE-EXT %s
// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN:   -march=rv64i_svukte0p3 -E -dM %s \
// RUN:   -o - | FileCheck --check-prefix=CHECK-SVUKTE-EXT %s
// CHECK-SVUKTE-EXT: __riscv_svukte 3000{{$}}

// Misaligned

// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-AVOID
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-AVOID
// CHECK-MISALIGNED-AVOID: __riscv_misaligned_avoid 1

// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i -E -dM %s \
// RUN:   -mno-strict-align -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-FAST
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i -E -dM %s \
// RUN:   -mno-strict-align -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-FAST
// RUN: %clang --target=riscv64-unknown-linux-gnu -mcpu=sifive-p450 -E -dM %s \
// RUN:  -o - | FileCheck %s --check-prefix=CHECK-MISALIGNED-FAST
// CHECK-MISALIGNED-FAST: __riscv_misaligned_fast 1
