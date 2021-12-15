//===-- Common constants for math functions ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common_constants.h"

namespace __llvm_libc {

// Lookup table for (1/f) where f = 1 + n*2^(-7), n = 0..127.
const double ONE_OVER_F[128] = {
    0x1.0000000000000p+0, 0x1.fc07f01fc07f0p-1, 0x1.f81f81f81f820p-1,
    0x1.f44659e4a4271p-1, 0x1.f07c1f07c1f08p-1, 0x1.ecc07b301ecc0p-1,
    0x1.e9131abf0b767p-1, 0x1.e573ac901e574p-1, 0x1.e1e1e1e1e1e1ep-1,
    0x1.de5d6e3f8868ap-1, 0x1.dae6076b981dbp-1, 0x1.d77b654b82c34p-1,
    0x1.d41d41d41d41dp-1, 0x1.d0cb58f6ec074p-1, 0x1.cd85689039b0bp-1,
    0x1.ca4b3055ee191p-1, 0x1.c71c71c71c71cp-1, 0x1.c3f8f01c3f8f0p-1,
    0x1.c0e070381c0e0p-1, 0x1.bdd2b899406f7p-1, 0x1.bacf914c1bad0p-1,
    0x1.b7d6c3dda338bp-1, 0x1.b4e81b4e81b4fp-1, 0x1.b2036406c80d9p-1,
    0x1.af286bca1af28p-1, 0x1.ac5701ac5701bp-1, 0x1.a98ef606a63bep-1,
    0x1.a6d01a6d01a6dp-1, 0x1.a41a41a41a41ap-1, 0x1.a16d3f97a4b02p-1,
    0x1.9ec8e951033d9p-1, 0x1.9c2d14ee4a102p-1, 0x1.999999999999ap-1,
    0x1.970e4f80cb872p-1, 0x1.948b0fcd6e9e0p-1, 0x1.920fb49d0e229p-1,
    0x1.8f9c18f9c18fap-1, 0x1.8d3018d3018d3p-1, 0x1.8acb90f6bf3aap-1,
    0x1.886e5f0abb04ap-1, 0x1.8618618618618p-1, 0x1.83c977ab2beddp-1,
    0x1.8181818181818p-1, 0x1.7f405fd017f40p-1, 0x1.7d05f417d05f4p-1,
    0x1.7ad2208e0ecc3p-1, 0x1.78a4c8178a4c8p-1, 0x1.767dce434a9b1p-1,
    0x1.745d1745d1746p-1, 0x1.724287f46debcp-1, 0x1.702e05c0b8170p-1,
    0x1.6e1f76b4337c7p-1, 0x1.6c16c16c16c17p-1, 0x1.6a13cd1537290p-1,
    0x1.6816816816817p-1, 0x1.661ec6a5122f9p-1, 0x1.642c8590b2164p-1,
    0x1.623fa77016240p-1, 0x1.6058160581606p-1, 0x1.5e75bb8d015e7p-1,
    0x1.5c9882b931057p-1, 0x1.5ac056b015ac0p-1, 0x1.58ed2308158edp-1,
    0x1.571ed3c506b3ap-1, 0x1.5555555555555p-1, 0x1.5390948f40febp-1,
    0x1.51d07eae2f815p-1, 0x1.5015015015015p-1, 0x1.4e5e0a72f0539p-1,
    0x1.4cab88725af6ep-1, 0x1.4afd6a052bf5bp-1, 0x1.49539e3b2d067p-1,
    0x1.47ae147ae147bp-1, 0x1.460cbc7f5cf9ap-1, 0x1.446f86562d9fbp-1,
    0x1.42d6625d51f87p-1, 0x1.4141414141414p-1, 0x1.3fb013fb013fbp-1,
    0x1.3e22cbce4a902p-1, 0x1.3c995a47babe7p-1, 0x1.3b13b13b13b14p-1,
    0x1.3991c2c187f63p-1, 0x1.3813813813814p-1, 0x1.3698df3de0748p-1,
    0x1.3521cfb2b78c1p-1, 0x1.33ae45b57bcb2p-1, 0x1.323e34a2b10bfp-1,
    0x1.30d190130d190p-1, 0x1.2f684bda12f68p-1, 0x1.2e025c04b8097p-1,
    0x1.2c9fb4d812ca0p-1, 0x1.2b404ad012b40p-1, 0x1.29e4129e4129ep-1,
    0x1.288b01288b013p-1, 0x1.27350b8812735p-1, 0x1.25e22708092f1p-1,
    0x1.2492492492492p-1, 0x1.23456789abcdfp-1, 0x1.21fb78121fb78p-1,
    0x1.20b470c67c0d9p-1, 0x1.1f7047dc11f70p-1, 0x1.1e2ef3b3fb874p-1,
    0x1.1cf06ada2811dp-1, 0x1.1bb4a4046ed29p-1, 0x1.1a7b9611a7b96p-1,
    0x1.19453808ca29cp-1, 0x1.1811811811812p-1, 0x1.16e0689427379p-1,
    0x1.15b1e5f75270dp-1, 0x1.1485f0e0acd3bp-1, 0x1.135c81135c811p-1,
    0x1.12358e75d3033p-1, 0x1.1111111111111p-1, 0x1.0fef010fef011p-1,
    0x1.0ecf56be69c90p-1, 0x1.0db20a88f4696p-1, 0x1.0c9714fbcda3bp-1,
    0x1.0b7e6ec259dc8p-1, 0x1.0a6810a6810a7p-1, 0x1.0953f39010954p-1,
    0x1.0842108421084p-1, 0x1.073260a47f7c6p-1, 0x1.0624dd2f1a9fcp-1,
    0x1.05197f7d73404p-1, 0x1.0410410410410p-1, 0x1.03091b51f5e1ap-1,
    0x1.0204081020408p-1, 0x1.0101010101010p-1};

} // namespace __llvm_libc
