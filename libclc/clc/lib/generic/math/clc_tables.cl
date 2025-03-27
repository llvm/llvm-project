//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/tables.h>

DECLARE_TABLE(float, LOGE_TBL_LO, 129) = {
    0x0.000000p+0f, 0x1.fe0000p-8f, 0x1.fc0000p-7f, 0x1.7b8000p-6f,
    0x1.f82000p-6f, 0x1.39e000p-5f, 0x1.774000p-5f, 0x1.b42000p-5f,
    0x1.f0a000p-5f, 0x1.164000p-4f, 0x1.340000p-4f, 0x1.51a000p-4f,
    0x1.6f0000p-4f, 0x1.8c2000p-4f, 0x1.a92000p-4f, 0x1.c5e000p-4f,
    0x1.e26000p-4f, 0x1.fec000p-4f, 0x1.0d6000p-3f, 0x1.1b6000p-3f,
    0x1.294000p-3f, 0x1.370000p-3f, 0x1.44c000p-3f, 0x1.526000p-3f,
    0x1.5fe000p-3f, 0x1.6d6000p-3f, 0x1.7aa000p-3f, 0x1.87e000p-3f,
    0x1.952000p-3f, 0x1.a22000p-3f, 0x1.af2000p-3f, 0x1.bc2000p-3f,
    0x1.c8e000p-3f, 0x1.d5c000p-3f, 0x1.e26000p-3f, 0x1.ef0000p-3f,
    0x1.fb8000p-3f, 0x1.040000p-2f, 0x1.0a2000p-2f, 0x1.104000p-2f,
    0x1.166000p-2f, 0x1.1c8000p-2f, 0x1.228000p-2f, 0x1.288000p-2f,
    0x1.2e8000p-2f, 0x1.346000p-2f, 0x1.3a6000p-2f, 0x1.404000p-2f,
    0x1.460000p-2f, 0x1.4be000p-2f, 0x1.51a000p-2f, 0x1.576000p-2f,
    0x1.5d0000p-2f, 0x1.62c000p-2f, 0x1.686000p-2f, 0x1.6e0000p-2f,
    0x1.738000p-2f, 0x1.792000p-2f, 0x1.7ea000p-2f, 0x1.842000p-2f,
    0x1.89a000p-2f, 0x1.8f0000p-2f, 0x1.946000p-2f, 0x1.99c000p-2f,
    0x1.9f2000p-2f, 0x1.a48000p-2f, 0x1.a9c000p-2f, 0x1.af0000p-2f,
    0x1.b44000p-2f, 0x1.b98000p-2f, 0x1.bea000p-2f, 0x1.c3c000p-2f,
    0x1.c8e000p-2f, 0x1.ce0000p-2f, 0x1.d32000p-2f, 0x1.d82000p-2f,
    0x1.dd4000p-2f, 0x1.e24000p-2f, 0x1.e74000p-2f, 0x1.ec2000p-2f,
    0x1.f12000p-2f, 0x1.f60000p-2f, 0x1.fae000p-2f, 0x1.ffc000p-2f,
    0x1.024000p-1f, 0x1.04a000p-1f, 0x1.072000p-1f, 0x1.098000p-1f,
    0x1.0be000p-1f, 0x1.0e4000p-1f, 0x1.108000p-1f, 0x1.12e000p-1f,
    0x1.154000p-1f, 0x1.178000p-1f, 0x1.19e000p-1f, 0x1.1c2000p-1f,
    0x1.1e8000p-1f, 0x1.20c000p-1f, 0x1.230000p-1f, 0x1.254000p-1f,
    0x1.278000p-1f, 0x1.29c000p-1f, 0x1.2c0000p-1f, 0x1.2e4000p-1f,
    0x1.306000p-1f, 0x1.32a000p-1f, 0x1.34e000p-1f, 0x1.370000p-1f,
    0x1.392000p-1f, 0x1.3b6000p-1f, 0x1.3d8000p-1f, 0x1.3fa000p-1f,
    0x1.41c000p-1f, 0x1.43e000p-1f, 0x1.460000p-1f, 0x1.482000p-1f,
    0x1.4a4000p-1f, 0x1.4c6000p-1f, 0x1.4e6000p-1f, 0x1.508000p-1f,
    0x1.52a000p-1f, 0x1.54a000p-1f, 0x1.56a000p-1f, 0x1.58c000p-1f,
    0x1.5ac000p-1f, 0x1.5cc000p-1f, 0x1.5ee000p-1f, 0x1.60e000p-1f,
    0x1.62e000p-1f,
};

DECLARE_TABLE(float, LOGE_TBL_HI, 129) = {
    0x0.000000p+0f,  0x1.535882p-23f, 0x1.5161f8p-20f, 0x1.1b07d4p-18f,
    0x1.361cf0p-19f, 0x1.0f73fcp-18f, 0x1.63d8cap-19f, 0x1.bae232p-18f,
    0x1.86008ap-20f, 0x1.36eea2p-16f, 0x1.d7961ap-16f, 0x1.073f06p-16f,
    0x1.a515cap-17f, 0x1.45d630p-16f, 0x1.b4e92ap-18f, 0x1.523d6ep-18f,
    0x1.076e2ap-16f, 0x1.2263b6p-17f, 0x1.7e7cd0p-15f, 0x1.2ad52ep-15f,
    0x1.52f81ep-15f, 0x1.fc201ep-15f, 0x1.2b6ccap-15f, 0x1.cbc742p-16f,
    0x1.3070a6p-15f, 0x1.fce33ap-20f, 0x1.890210p-15f, 0x1.a06520p-15f,
    0x1.6a73d0p-17f, 0x1.bc1fe2p-15f, 0x1.c94e80p-15f, 0x1.0ce85ap-16f,
    0x1.f7c79ap-15f, 0x1.0b5a7cp-18f, 0x1.076e2ap-15f, 0x1.5b97b8p-16f,
    0x1.186d5ep-15f, 0x1.2ca5a6p-17f, 0x1.24e272p-14f, 0x1.8bf9aep-14f,
    0x1.5cabaap-14f, 0x1.3182d2p-15f, 0x1.41fbcep-14f, 0x1.5a13dep-14f,
    0x1.c575c2p-15f, 0x1.dd9a98p-14f, 0x1.3155a4p-16f, 0x1.843434p-17f,
    0x1.8bc21cp-14f, 0x1.7e55dcp-16f, 0x1.5b0e5ap-15f, 0x1.dc5d14p-16f,
    0x1.bdbf58p-14f, 0x1.05e572p-15f, 0x1.903d36p-15f, 0x1.1d5456p-15f,
    0x1.d7f6bap-14f, 0x1.4abfbap-15f, 0x1.f07704p-15f, 0x1.a3b43cp-15f,
    0x1.9c360ap-17f, 0x1.1e8736p-14f, 0x1.941c20p-14f, 0x1.958116p-14f,
    0x1.23ecbep-14f, 0x1.024396p-16f, 0x1.d93534p-15f, 0x1.293246p-14f,
    0x1.eef798p-15f, 0x1.625a4cp-16f, 0x1.4d9da6p-14f, 0x1.d7a7ccp-14f,
    0x1.f7c79ap-14f, 0x1.af0b84p-14f, 0x1.fcfc00p-15f, 0x1.e7258ap-14f,
    0x1.a81306p-16f, 0x1.1034f8p-15f, 0x1.09875ap-16f, 0x1.99d246p-14f,
    0x1.1ebf5ep-15f, 0x1.23fa70p-14f, 0x1.588f78p-14f, 0x1.2e0856p-14f,
    0x1.52a5a4p-13f, 0x1.df9da8p-13f, 0x1.f2e0e6p-16f, 0x1.bd3d5cp-15f,
    0x1.cb9094p-15f, 0x1.261746p-15f, 0x1.f39e2cp-13f, 0x1.719592p-13f,
    0x1.87a5e8p-14f, 0x1.eabbd8p-13f, 0x1.cd68cep-14f, 0x1.b81f70p-13f,
    0x1.7d79c0p-15f, 0x1.b9a324p-14f, 0x1.30d7bep-13f, 0x1.5bce98p-13f,
    0x1.5e1288p-13f, 0x1.37fec2p-13f, 0x1.d3da88p-14f, 0x1.d0db90p-15f,
    0x1.d7334ep-13f, 0x1.133912p-13f, 0x1.44ece6p-16f, 0x1.17b546p-13f,
    0x1.e0d356p-13f, 0x1.0893fep-14f, 0x1.026a70p-13f, 0x1.5b84d0p-13f,
    0x1.8fe846p-13f, 0x1.9fe2f8p-13f, 0x1.8bc21cp-13f, 0x1.53d1eap-13f,
    0x1.f0bb60p-14f, 0x1.e6bf32p-15f, 0x1.d811b6p-13f, 0x1.13cc00p-13f,
    0x1.6932dep-16f, 0x1.246798p-13f, 0x1.f9d5b2p-13f, 0x1.5b6b9ap-14f,
    0x1.404c34p-13f, 0x1.b1dc6cp-13f, 0x1.54920ap-20f, 0x1.97a23cp-16f,
    0x1.0bfbe8p-15f,
};

CLC_TABLE_FUNCTION(float, LOGE_TBL_LO, loge_tbl_lo);
CLC_TABLE_FUNCTION(float, LOGE_TBL_HI, loge_tbl_hi);

DECLARE_TABLE(float, LOG_INV_TBL, 129) = {
    0x1.000000p+1f, 0x1.fc07f0p+0f, 0x1.f81f82p+0f, 0x1.f4465ap+0f,
    0x1.f07c20p+0f, 0x1.ecc07cp+0f, 0x1.e9131ap+0f, 0x1.e573acp+0f,
    0x1.e1e1e2p+0f, 0x1.de5d6ep+0f, 0x1.dae608p+0f, 0x1.d77b66p+0f,
    0x1.d41d42p+0f, 0x1.d0cb58p+0f, 0x1.cd8568p+0f, 0x1.ca4b30p+0f,
    0x1.c71c72p+0f, 0x1.c3f8f0p+0f, 0x1.c0e070p+0f, 0x1.bdd2b8p+0f,
    0x1.bacf92p+0f, 0x1.b7d6c4p+0f, 0x1.b4e81cp+0f, 0x1.b20364p+0f,
    0x1.af286cp+0f, 0x1.ac5702p+0f, 0x1.a98ef6p+0f, 0x1.a6d01ap+0f,
    0x1.a41a42p+0f, 0x1.a16d40p+0f, 0x1.9ec8eap+0f, 0x1.9c2d14p+0f,
    0x1.99999ap+0f, 0x1.970e50p+0f, 0x1.948b10p+0f, 0x1.920fb4p+0f,
    0x1.8f9c18p+0f, 0x1.8d3018p+0f, 0x1.8acb90p+0f, 0x1.886e60p+0f,
    0x1.861862p+0f, 0x1.83c978p+0f, 0x1.818182p+0f, 0x1.7f4060p+0f,
    0x1.7d05f4p+0f, 0x1.7ad220p+0f, 0x1.78a4c8p+0f, 0x1.767dcep+0f,
    0x1.745d18p+0f, 0x1.724288p+0f, 0x1.702e06p+0f, 0x1.6e1f76p+0f,
    0x1.6c16c2p+0f, 0x1.6a13cep+0f, 0x1.681682p+0f, 0x1.661ec6p+0f,
    0x1.642c86p+0f, 0x1.623fa8p+0f, 0x1.605816p+0f, 0x1.5e75bcp+0f,
    0x1.5c9882p+0f, 0x1.5ac056p+0f, 0x1.58ed24p+0f, 0x1.571ed4p+0f,
    0x1.555556p+0f, 0x1.539094p+0f, 0x1.51d07ep+0f, 0x1.501502p+0f,
    0x1.4e5e0ap+0f, 0x1.4cab88p+0f, 0x1.4afd6ap+0f, 0x1.49539ep+0f,
    0x1.47ae14p+0f, 0x1.460cbcp+0f, 0x1.446f86p+0f, 0x1.42d662p+0f,
    0x1.414142p+0f, 0x1.3fb014p+0f, 0x1.3e22ccp+0f, 0x1.3c995ap+0f,
    0x1.3b13b2p+0f, 0x1.3991c2p+0f, 0x1.381382p+0f, 0x1.3698e0p+0f,
    0x1.3521d0p+0f, 0x1.33ae46p+0f, 0x1.323e34p+0f, 0x1.30d190p+0f,
    0x1.2f684cp+0f, 0x1.2e025cp+0f, 0x1.2c9fb4p+0f, 0x1.2b404ap+0f,
    0x1.29e412p+0f, 0x1.288b02p+0f, 0x1.27350cp+0f, 0x1.25e228p+0f,
    0x1.24924ap+0f, 0x1.234568p+0f, 0x1.21fb78p+0f, 0x1.20b470p+0f,
    0x1.1f7048p+0f, 0x1.1e2ef4p+0f, 0x1.1cf06ap+0f, 0x1.1bb4a4p+0f,
    0x1.1a7b96p+0f, 0x1.194538p+0f, 0x1.181182p+0f, 0x1.16e068p+0f,
    0x1.15b1e6p+0f, 0x1.1485f0p+0f, 0x1.135c82p+0f, 0x1.12358ep+0f,
    0x1.111112p+0f, 0x1.0fef02p+0f, 0x1.0ecf56p+0f, 0x1.0db20ap+0f,
    0x1.0c9714p+0f, 0x1.0b7e6ep+0f, 0x1.0a6810p+0f, 0x1.0953f4p+0f,
    0x1.084210p+0f, 0x1.073260p+0f, 0x1.0624dep+0f, 0x1.051980p+0f,
    0x1.041042p+0f, 0x1.03091cp+0f, 0x1.020408p+0f, 0x1.010102p+0f,
    0x1.000000p+0f,
};

CLC_TABLE_FUNCTION(float, LOG_INV_TBL, log_inv_tbl);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

DECLARE_TABLE(double, LN_TBL_LO, 65) = {
    0x0.0000000000000p+0, 0x1.fc0a800000000p-7, 0x1.f829800000000p-6,
    0x1.7745800000000p-5, 0x1.f0a3000000000p-5, 0x1.341d700000000p-4,
    0x1.6f0d200000000p-4, 0x1.a926d00000000p-4, 0x1.e270700000000p-4,
    0x1.0d77e00000000p-3, 0x1.2955280000000p-3, 0x1.44d2b00000000p-3,
    0x1.5ff3000000000p-3, 0x1.7ab8900000000p-3, 0x1.9525a80000000p-3,
    0x1.af3c900000000p-3, 0x1.c8ff780000000p-3, 0x1.e270700000000p-3,
    0x1.fb91800000000p-3, 0x1.0a324c0000000p-2, 0x1.1675c80000000p-2,
    0x1.22941c0000000p-2, 0x1.2e8e280000000p-2, 0x1.3a64c40000000p-2,
    0x1.4618bc0000000p-2, 0x1.51aad80000000p-2, 0x1.5d1bd80000000p-2,
    0x1.686c800000000p-2, 0x1.739d7c0000000p-2, 0x1.7eaf800000000p-2,
    0x1.89a3380000000p-2, 0x1.9479400000000p-2, 0x1.9f323c0000000p-2,
    0x1.a9cec80000000p-2, 0x1.b44f740000000p-2, 0x1.beb4d80000000p-2,
    0x1.c8ff7c0000000p-2, 0x1.d32fe40000000p-2, 0x1.dd46a00000000p-2,
    0x1.e744240000000p-2, 0x1.f128f40000000p-2, 0x1.faf5880000000p-2,
    0x1.02552a0000000p-1, 0x1.0723e40000000p-1, 0x1.0be72e0000000p-1,
    0x1.109f380000000p-1, 0x1.154c3c0000000p-1, 0x1.19ee6a0000000p-1,
    0x1.1e85f40000000p-1, 0x1.23130c0000000p-1, 0x1.2795e00000000p-1,
    0x1.2c0e9e0000000p-1, 0x1.307d720000000p-1, 0x1.34e2880000000p-1,
    0x1.393e0c0000000p-1, 0x1.3d90260000000p-1, 0x1.41d8fe0000000p-1,
    0x1.4618bc0000000p-1, 0x1.4a4f840000000p-1, 0x1.4e7d800000000p-1,
    0x1.52a2d20000000p-1, 0x1.56bf9c0000000p-1, 0x1.5ad4040000000p-1,
    0x1.5ee02a0000000p-1, 0x1.62e42e0000000p-1,
};

CLC_TABLE_FUNCTION(double, LN_TBL_LO, ln_tbl_lo);

DECLARE_TABLE(double, LN_TBL_HI, 65) = {
    0x0.0000000000000p+0,  0x1.61f807c79f3dbp-28, 0x1.873c1980267c8p-25,
    0x1.ec65b9f88c69ep-26, 0x1.8022c54cc2f99p-26, 0x1.2c37a3a125330p-25,
    0x1.15cad69737c93p-25, 0x1.d256ab1b285e9p-27, 0x1.b8abcb97a7aa2p-26,
    0x1.f34239659a5dcp-25, 0x1.e07fd48d30177p-25, 0x1.b32df4799f4f6p-25,
    0x1.c29e4f4f21cf8p-25, 0x1.086c848df1b59p-30, 0x1.cf456b4764130p-27,
    0x1.3a02ffcb63398p-25, 0x1.1e6a6886b0976p-25, 0x1.b8abcb97a7aa2p-25,
    0x1.b578f8aa35552p-25, 0x1.139c871afb9fcp-25, 0x1.5d5d30701ce64p-25,
    0x1.de7bcb2d12142p-25, 0x1.d708e984e1664p-25, 0x1.56945e9c72f36p-26,
    0x1.0e2f613e85bdap-29, 0x1.cb7e0b42724f6p-28, 0x1.fac04e52846c7p-25,
    0x1.e9b14aec442bep-26, 0x1.b5de8034e7126p-25, 0x1.dc157e1b259d3p-25,
    0x1.b05096ad69c62p-28, 0x1.c2116faba4cddp-26, 0x1.65fcc25f95b47p-25,
    0x1.a9a08498d4850p-26, 0x1.de647b1465f77p-25, 0x1.da71b7bf7861dp-26,
    0x1.e6a6886b09760p-28, 0x1.f0075eab0ef64p-25, 0x1.3071282fb989bp-28,
    0x1.0eb43c3f1bed2p-25, 0x1.faf06ecb35c84p-26, 0x1.ef1e63db35f68p-27,
    0x1.69743fb1a71a5p-27, 0x1.c1cdf404e5796p-25, 0x1.094aa0ada625ep-27,
    0x1.e2d4c96fde3ecp-25, 0x1.2f4d5e9a98f34p-25, 0x1.467c96ecc5cbep-25,
    0x1.e7040d03dec5ap-25, 0x1.7bebf4282de36p-25, 0x1.289b11aeb783fp-25,
    0x1.a891d1772f538p-26, 0x1.34f10be1fb591p-25, 0x1.d9ce1d316eb93p-25,
    0x1.3562a19a9c442p-25, 0x1.4e2adf548084cp-26, 0x1.08ce55cc8c97ap-26,
    0x1.0e2f613e85bdap-28, 0x1.db03ebb0227bfp-25, 0x1.1b75bb09cb098p-25,
    0x1.96f16abb9df22p-27, 0x1.5b3f399411c62p-25, 0x1.86b3e59f65355p-26,
    0x1.2482ceae1ac12p-26, 0x1.efa39ef35793cp-25,
};

CLC_TABLE_FUNCTION(double, LN_TBL_HI, ln_tbl_hi);

#endif // cl_khr_fp64
