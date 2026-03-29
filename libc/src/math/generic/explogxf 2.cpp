//===-- Single-precision general exp/log functions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "explogxf.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// N[Table[Log[2, 1 + x], {x, 0/64, 63/64, 1/64}], 40]
alignas(64) const double LOG_P1_LOG2[LOG_P1_SIZE] = {
    0x0.0000000000000p+0, 0x1.6e79685c2d22ap-6, 0x1.6bad3758efd87p-5,
    0x1.0eb389fa29f9bp-4, 0x1.663f6fac91316p-4, 0x1.bc84240adabbap-4,
    0x1.08c588cda79e4p-3, 0x1.32ae9e278ae1ap-3, 0x1.5c01a39fbd688p-3,
    0x1.84c2bd02f03b3p-3, 0x1.acf5e2db4ec94p-3, 0x1.d49ee4c325970p-3,
    0x1.fbc16b902680ap-3, 0x1.11307dad30b76p-2, 0x1.24407ab0e073ap-2,
    0x1.37124cea4cdedp-2, 0x1.49a784bcd1b8bp-2, 0x1.5c01a39fbd688p-2,
    0x1.6e221cd9d0cdep-2, 0x1.800a563161c54p-2, 0x1.91bba891f1709p-2,
    0x1.a33760a7f6051p-2, 0x1.b47ebf73882a1p-2, 0x1.c592fad295b56p-2,
    0x1.d6753e032ea0fp-2, 0x1.e726aa1e754d2p-2, 0x1.f7a8568cb06cfp-2,
    0x1.03fda8b97997fp-1, 0x1.0c10500d63aa6p-1, 0x1.140c9faa1e544p-1,
    0x1.1bf311e95d00ep-1, 0x1.23c41d42727c8p-1, 0x1.2b803473f7ad1p-1,
    0x1.3327c6ab49ca7p-1, 0x1.3abb3faa02167p-1, 0x1.423b07e986aa9p-1,
    0x1.49a784bcd1b8bp-1, 0x1.510118708a8f9p-1, 0x1.5848226989d34p-1,
    0x1.5f7cff41e09afp-1, 0x1.66a008e4788ccp-1, 0x1.6db196a76194ap-1,
    0x1.74b1fd64e0754p-1, 0x1.7ba18f93502e4p-1, 0x1.82809d5be7073p-1,
    0x1.894f74b06ef8bp-1, 0x1.900e6160002cdp-1, 0x1.96bdad2acb5f6p-1,
    0x1.9d5d9fd5010b3p-1, 0x1.a3ee7f38e181fp-1, 0x1.aa708f58014d3p-1,
    0x1.b0e4126bcc86cp-1, 0x1.b74948f5532dap-1, 0x1.bda071cc67e6ep-1,
    0x1.c3e9ca2e1a055p-1, 0x1.ca258dca93316p-1, 0x1.d053f6d260896p-1,
    0x1.d6753e032ea0fp-1, 0x1.dc899ab3ff56cp-1, 0x1.e29142e0e0140p-1,
    0x1.e88c6b3626a73p-1, 0x1.ee7b471b3a950p-1, 0x1.f45e08bcf0655p-1,
    0x1.fa34e1177c233p-1,
};

// N[Table[1/(1 + x), {x, 0/64, 63/64, 1/64}], 40]
alignas(64) const double LOG_P1_1_OVER[LOG_P1_SIZE] = {
    0x1.0000000000000p+0, 0x1.f81f81f81f820p-1, 0x1.f07c1f07c1f08p-1,
    0x1.e9131abf0b767p-1, 0x1.e1e1e1e1e1e1ep-1, 0x1.dae6076b981dbp-1,
    0x1.d41d41d41d41dp-1, 0x1.cd85689039b0bp-1, 0x1.c71c71c71c71cp-1,
    0x1.c0e070381c0e0p-1, 0x1.bacf914c1bad0p-1, 0x1.b4e81b4e81b4fp-1,
    0x1.af286bca1af28p-1, 0x1.a98ef606a63bep-1, 0x1.a41a41a41a41ap-1,
    0x1.9ec8e951033d9p-1, 0x1.999999999999ap-1, 0x1.948b0fcd6e9e0p-1,
    0x1.8f9c18f9c18fap-1, 0x1.8acb90f6bf3aap-1, 0x1.8618618618618p-1,
    0x1.8181818181818p-1, 0x1.7d05f417d05f4p-1, 0x1.78a4c8178a4c8p-1,
    0x1.745d1745d1746p-1, 0x1.702e05c0b8170p-1, 0x1.6c16c16c16c17p-1,
    0x1.6816816816817p-1, 0x1.642c8590b2164p-1, 0x1.6058160581606p-1,
    0x1.5c9882b931057p-1, 0x1.58ed2308158edp-1, 0x1.5555555555555p-1,
    0x1.51d07eae2f815p-1, 0x1.4e5e0a72f0539p-1, 0x1.4afd6a052bf5bp-1,
    0x1.47ae147ae147bp-1, 0x1.446f86562d9fbp-1, 0x1.4141414141414p-1,
    0x1.3e22cbce4a902p-1, 0x1.3b13b13b13b14p-1, 0x1.3813813813814p-1,
    0x1.3521cfb2b78c1p-1, 0x1.323e34a2b10bfp-1, 0x1.2f684bda12f68p-1,
    0x1.2c9fb4d812ca0p-1, 0x1.29e4129e4129ep-1, 0x1.27350b8812735p-1,
    0x1.2492492492492p-1, 0x1.21fb78121fb78p-1, 0x1.1f7047dc11f70p-1,
    0x1.1cf06ada2811dp-1, 0x1.1a7b9611a7b96p-1, 0x1.1811811811812p-1,
    0x1.15b1e5f75270dp-1, 0x1.135c81135c811p-1, 0x1.1111111111111p-1,
    0x1.0ecf56be69c90p-1, 0x1.0c9714fbcda3bp-1, 0x1.0a6810a6810a7p-1,
    0x1.0842108421084p-1, 0x1.0624dd2f1a9fcp-1, 0x1.0410410410410p-1,
    0x1.0204081020408p-1};

// Taylos series expansion for Log[2, 1 + x] splitted to EVEN AND ODD numbers
// K_LOG2_ODD starts from x^3
alignas(64) const
    double K_LOG2_ODD[4] = {0x1.ec709dc3a03fdp-2, 0x1.2776c50ef9bfep-2,
                            0x1.a61762a7aded9p-3, 0x1.484b13d7c02a9p-3};

alignas(64) const
    double K_LOG2_EVEN[4] = {-0x1.71547652b82fep-1, -0x1.71547652b82fep-2,
                             -0x1.ec709dc3a03fdp-3, -0x1.2776c50ef9bfep-3};

} // namespace LIBC_NAMESPACE_DECL
