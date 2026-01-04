//===-- Common constants for acoshf function --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_FLOAT_CONSTANTS_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_FLOAT_CONSTANTS_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace acoshf_internal {

// Lookup table for (1/f) where f = 1 + n*2^(-7), n = 0..127.
static constexpr double ONE_OVER_F[128] = {
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

// Lookup table for log(f) = log(1 + n*2^(-7)) where n = 0..127.
static constexpr double LOG_F[128] = {
    0x0.0000000000000p+0, 0x1.fe02a6b106788p-8, 0x1.fc0a8b0fc03e3p-7,
    0x1.7b91b07d5b11ap-6, 0x1.f829b0e783300p-6, 0x1.39e87b9febd5fp-5,
    0x1.77458f632dcfcp-5, 0x1.b42dd711971bep-5, 0x1.f0a30c01162a6p-5,
    0x1.16536eea37ae0p-4, 0x1.341d7961bd1d0p-4, 0x1.51b073f06183fp-4,
    0x1.6f0d28ae56b4bp-4, 0x1.8c345d6319b20p-4, 0x1.a926d3a4ad563p-4,
    0x1.c5e548f5bc743p-4, 0x1.e27076e2af2e5p-4, 0x1.fec9131dbeabap-4,
    0x1.0d77e7cd08e59p-3, 0x1.1b72ad52f67a0p-3, 0x1.29552f81ff523p-3,
    0x1.371fc201e8f74p-3, 0x1.44d2b6ccb7d1ep-3, 0x1.526e5e3a1b437p-3,
    0x1.5ff3070a793d3p-3, 0x1.6d60fe719d21cp-3, 0x1.7ab890210d909p-3,
    0x1.87fa06520c910p-3, 0x1.9525a9cf456b4p-3, 0x1.a23bc1fe2b563p-3,
    0x1.af3c94e80bff2p-3, 0x1.bc286742d8cd6p-3, 0x1.c8ff7c79a9a21p-3,
    0x1.d5c216b4fbb91p-3, 0x1.e27076e2af2e5p-3, 0x1.ef0adcbdc5936p-3,
    0x1.fb9186d5e3e2ap-3, 0x1.0402594b4d040p-2, 0x1.0a324e27390e3p-2,
    0x1.1058bf9ae4ad5p-2, 0x1.1675cababa60ep-2, 0x1.1c898c16999fap-2,
    0x1.22941fbcf7965p-2, 0x1.2895a13de86a3p-2, 0x1.2e8e2bae11d30p-2,
    0x1.347dd9a987d54p-2, 0x1.3a64c556945e9p-2, 0x1.404308686a7e3p-2,
    0x1.4618bc21c5ec2p-2, 0x1.4be5f957778a0p-2, 0x1.51aad872df82dp-2,
    0x1.5767717455a6cp-2, 0x1.5d1bdbf5809cap-2, 0x1.62c82f2b9c795p-2,
    0x1.686c81e9b14aep-2, 0x1.6e08eaa2ba1e3p-2, 0x1.739d7f6bbd006p-2,
    0x1.792a55fdd47a2p-2, 0x1.7eaf83b82afc3p-2, 0x1.842d1da1e8b17p-2,
    0x1.89a3386c1425ap-2, 0x1.8f11e873662c7p-2, 0x1.947941c2116fap-2,
    0x1.99d958117e08ap-2, 0x1.9f323ecbf984bp-2, 0x1.a484090e5bb0ap-2,
    0x1.a9cec9a9a0849p-2, 0x1.af1293247786bp-2, 0x1.b44f77bcc8f62p-2,
    0x1.b9858969310fbp-2, 0x1.beb4d9da71b7bp-2, 0x1.c3dd7a7cdad4dp-2,
    0x1.c8ff7c79a9a21p-2, 0x1.ce1af0b85f3ebp-2, 0x1.d32fe7e00ebd5p-2,
    0x1.d83e7258a2f3ep-2, 0x1.dd46a04c1c4a0p-2, 0x1.e24881a7c6c26p-2,
    0x1.e744261d68787p-2, 0x1.ec399d2468cc0p-2, 0x1.f128f5faf06ecp-2,
    0x1.f6123fa7028acp-2, 0x1.faf588f78f31ep-2, 0x1.ffd2e0857f498p-2,
    0x1.02552a5a5d0fep-1, 0x1.04bdf9da926d2p-1, 0x1.0723e5c1cdf40p-1,
    0x1.0986f4f573520p-1, 0x1.0be72e4252a82p-1, 0x1.0e44985d1cc8bp-1,
    0x1.109f39e2d4c96p-1, 0x1.12f719593efbcp-1, 0x1.154c3d2f4d5e9p-1,
    0x1.179eabbd899a0p-1, 0x1.19ee6b467c96ep-1, 0x1.1c3b81f713c24p-1,
    0x1.1e85f5e7040d0p-1, 0x1.20cdcd192ab6dp-1, 0x1.23130d7bebf42p-1,
    0x1.2555bce98f7cbp-1, 0x1.2795e1289b11ap-1, 0x1.29d37fec2b08ap-1,
    0x1.2c0e9ed448e8bp-1, 0x1.2e47436e40268p-1, 0x1.307d7334f10bep-1,
    0x1.32b1339121d71p-1, 0x1.34e289d9ce1d3p-1, 0x1.37117b54747b5p-1,
    0x1.393e0d3562a19p-1, 0x1.3b68449fffc22p-1, 0x1.3d9026a7156fap-1,
    0x1.3fb5b84d16f42p-1, 0x1.41d8fe84672aep-1, 0x1.43f9fe2f9ce67p-1,
    0x1.4618bc21c5ec2p-1, 0x1.48353d1ea88dfp-1, 0x1.4a4f85db03ebbp-1,
    0x1.4c679afccee39p-1, 0x1.4e7d811b75bb0p-1, 0x1.50913cc01686bp-1,
    0x1.52a2d265bc5aap-1, 0x1.54b2467999497p-1, 0x1.56bf9d5b3f399p-1,
    0x1.58cadb5cd7989p-1, 0x1.5ad404c359f2cp-1, 0x1.5cdb1dc6c1764p-1,
    0x1.5ee02a9241675p-1, 0x1.60e32f44788d8p-1};

} // namespace acoshf_internal

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ACOSH_FLOAT_CONSTANTS_H
