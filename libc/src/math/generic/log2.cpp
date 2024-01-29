//===-- Double-precision log2(x) function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log2.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/PolyEval.h"
#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include "common_constants.h"
#include "log_range_reduction.h"

namespace LIBC_NAMESPACE {

// 128-bit precision dyadic floating point numbers.
using Float128 = typename fputil::DyadicFloat<128>;
using MType = typename Float128::MantissaType;
using Sign = fputil::Sign;

namespace {

constexpr fputil::DoubleDouble LOG2_E = {0x1.777d0ffda0d24p-56,
                                         0x1.71547652b82fep0};

// Extra errors from P is from using x^2 to reduce evaluation latency.
constexpr double P_ERR = 0x1.0p-49;

const fputil::DoubleDouble LOG_R1[128] = {
    {0.0, 0.0},
    {0x1.46662d417cedp-62, 0x1.010157588de71p-7},
    {0x1.27c8e8416e71fp-60, 0x1.0205658935847p-6},
    {-0x1.d192d0619fa67p-60, 0x1.8492528c8cabfp-6},
    {0x1.c05cf1d753622p-59, 0x1.0415d89e74444p-5},
    {-0x1.cdd6f7f4a137ep-59, 0x1.466aed42de3eap-5},
    {0x1.a8be97660a23dp-60, 0x1.894aa149fb343p-5},
    {-0x1.e48fb0500efd4p-59, 0x1.ccb73cdddb2ccp-5},
    {-0x1.dd7009902bf32p-58, 0x1.08598b59e3a07p-4},
    {-0x1.7558367a6acf6p-59, 0x1.1973bd1465567p-4},
    {0x1.7a976d3b5b45fp-59, 0x1.3bdf5a7d1ee64p-4},
    {0x1.f38745c5c450ap-58, 0x1.5e95a4d9791cbp-4},
    {-0x1.72566212cdd05p-61, 0x1.700d30aeac0e1p-4},
    {-0x1.478a85704ccb7p-58, 0x1.9335e5d594989p-4},
    {-0x1.0057eed1ca59fp-59, 0x1.b6ac88dad5b1cp-4},
    {0x1.a38cb559a6706p-58, 0x1.c885801bc4b23p-4},
    {-0x1.a2bf991780d3fp-59, 0x1.ec739830a112p-4},
    {-0x1.ac9f4215f9393p-58, 0x1.fe89139dbd566p-4},
    {-0x1.0e63a5f01c691p-58, 0x1.1178e8227e47cp-3},
    {-0x1.c6ef1d9b2ef7ep-59, 0x1.1aa2b7e23f72ap-3},
    {-0x1.499a3f25af95fp-58, 0x1.2d1610c86813ap-3},
    {0x1.7d411a5b944adp-58, 0x1.365fcb0159016p-3},
    {-0x1.0d5604930f135p-58, 0x1.4913d8333b561p-3},
    {-0x1.71a9682395bfdp-61, 0x1.527e5e4a1b58dp-3},
    {-0x1.d34f0f4621bedp-60, 0x1.6574ebe8c133ap-3},
    {-0x1.8de59c21e166cp-57, 0x1.6f0128b756abcp-3},
    {-0x1.1232ce70be781p-57, 0x1.823c16551a3c2p-3},
    {0x1.55aa8b6997a4p-58, 0x1.8beafeb38fe8cp-3},
    {0x1.142c507fb7a3dp-58, 0x1.95a5adcf7017fp-3},
    {0x1.bcafa9de97203p-57, 0x1.a93ed3c8ad9e3p-3},
    {-0x1.6353ab386a94dp-57, 0x1.b31d8575bce3dp-3},
    {0x1.dd355f6a516d7p-60, 0x1.bd087383bd8adp-3},
    {0x1.60629242471a2p-57, 0x1.d1037f2655e7bp-3},
    {0x1.aa11d49f96cb9p-58, 0x1.db13db0d4894p-3},
    {0x1.2276041f43042p-59, 0x1.e530effe71012p-3},
    {-0x1.08ab2ddc708ap-58, 0x1.ef5ade4dcffe6p-3},
    {0x1.f665066f980a2p-57, 0x1.f991c6cb3b379p-3},
    {0x1.cdb16ed4e9138p-56, 0x1.07138604d5862p-2},
    {0x1.162c79d5d11eep-58, 0x1.0c42d676162e3p-2},
    {-0x1.0e63a5f01c691p-57, 0x1.1178e8227e47cp-2},
    {0x1.66fbd28b40935p-56, 0x1.16b5ccbacfb73p-2},
    {-0x1.12aeb84249223p-57, 0x1.1bf99635a6b95p-2},
    {0x1.e0efadd9db02bp-56, 0x1.269621134db92p-2},
    {-0x1.82dad7fd86088p-56, 0x1.2bef07cdc9354p-2},
    {-0x1.3d69909e5c3dcp-56, 0x1.314f1e1d35ce4p-2},
    {-0x1.324f0e883858ep-58, 0x1.36b6776be1117p-2},
    {-0x1.2ad27e50a8ec6p-56, 0x1.3c25277333184p-2},
    {0x1.0dbb243827392p-57, 0x1.419b423d5e8c7p-2},
    {0x1.8fb4c14c56eefp-60, 0x1.4718dc271c41bp-2},
    {-0x1.123615b147a5dp-58, 0x1.4c9e09e172c3cp-2},
    {-0x1.8f7e9b38a6979p-57, 0x1.522ae0738a3d8p-2},
    {-0x1.0908d15f88b63p-57, 0x1.57bf753c8d1fbp-2},
    {-0x1.6541148cbb8a2p-56, 0x1.5d5bddf595f3p-2},
    {0x1.dc18ce51fff99p-57, 0x1.630030b3aac49p-2},
    {0x1.a64eadd740178p-58, 0x1.68ac83e9c6a14p-2},
    {0x1.657c222d868cdp-58, 0x1.6e60ee6af1972p-2},
    {0x1.84a4ee3059583p-56, 0x1.741d876c67bb1p-2},
    {-0x1.c168817443f22p-56, 0x1.79e26687cfb3ep-2},
    {-0x1.219024acd3b77p-58, 0x1.7fafa3bd8151cp-2},
    {-0x1.486666443b153p-56, 0x1.85855776dcbfbp-2},
    {-0x1.70f2f38238303p-56, 0x1.8b639a88b2df5p-2},
    {-0x1.ad4bb98c1f2c5p-56, 0x1.914a8635bf68ap-2},
    {-0x1.89d2816cf838fp-57, 0x1.973a3431356aep-2},
    {0x1.87bcbcfd3e187p-59, 0x1.9d32bea15ed3bp-2},
    {-0x1.ba8062860ae23p-57, 0x1.a33440224fa79p-2},
    {-0x1.ba8062860ae23p-57, 0x1.a33440224fa79p-2},
    {0x1.bcafa9de97203p-56, 0x1.a93ed3c8ad9e3p-2},
    {0x1.9d56c45dd3e86p-56, 0x1.af5295248cddp-2},
    {0x1.494b610665378p-56, 0x1.b56fa04462909p-2},
    {0x1.6fd02999b21e1p-59, 0x1.bb9611b80e2fbp-2},
    {-0x1.bfc00b8f3feaap-56, 0x1.c1c60693fa39ep-2},
    {-0x1.bfc00b8f3feaap-56, 0x1.c1c60693fa39ep-2},
    {0x1.223eadb651b4ap-57, 0x1.c7ff9c74554c9p-2},
    {0x1.0798270b29f39p-56, 0x1.ce42f18064743p-2},
    {0x1.d7f4d3b3d406bp-56, 0x1.d490246defa6bp-2},
    {-0x1.0b5837185a661p-56, 0x1.dae75484c9616p-2},
    {-0x1.ac81cc8a4dfb8p-56, 0x1.e148a1a2726cep-2},
    {-0x1.ac81cc8a4dfb8p-56, 0x1.e148a1a2726cep-2},
    {0x1.57d646a17bc6ap-56, 0x1.e7b42c3ddad73p-2},
    {-0x1.74b71fb5e57e3p-62, 0x1.ee2a156b413e5p-2},
    {-0x1.0d487f5aba5e5p-57, 0x1.f4aa7ee03192dp-2},
    {-0x1.0d487f5aba5e5p-57, 0x1.f4aa7ee03192dp-2},
    {0x1.7e8f05924d259p-57, 0x1.fb358af7a4884p-2},
    {0x1.1713a36138e19p-57, 0x1.00e5ae5b207abp-1},
    {-0x1.17f9e54e78104p-57, 0x1.04360be7603adp-1},
    {-0x1.17f9e54e78104p-57, 0x1.04360be7603adp-1},
    {0x1.2241edf5fd1f7p-57, 0x1.078bf0533c568p-1},
    {0x1.0d710fcfc4e0dp-55, 0x1.0ae76e2d054fap-1},
    {0x1.0d710fcfc4e0dp-55, 0x1.0ae76e2d054fap-1},
    {0x1.3300f002e836ep-55, 0x1.0e4898611cce1p-1},
    {-0x1.91eee7772c7c2p-55, 0x1.11af823c75aa8p-1},
    {-0x1.91eee7772c7c2p-55, 0x1.11af823c75aa8p-1},
    {0x1.342eb628dba17p-56, 0x1.151c3f6f29612p-1},
    {0x1.89df1568ca0bp-55, 0x1.188ee40f23ca6p-1},
    {0x1.89df1568ca0bp-55, 0x1.188ee40f23ca6p-1},
    {0x1.59bddae1ccce2p-56, 0x1.1c07849ae6007p-1},
    {-0x1.2164ff40e9817p-56, 0x1.1f8635fc61659p-1},
    {-0x1.2164ff40e9817p-56, 0x1.1f8635fc61659p-1},
    {-0x1.fcc8dbccc25cbp-57, 0x1.230b0d8bebc98p-1},
    {0x1.e0efadd9db02bp-55, 0x1.269621134db92p-1},
    {0x1.e0efadd9db02bp-55, 0x1.269621134db92p-1},
    {-0x1.6a0c343be95dcp-56, 0x1.2a2786d0ec107p-1},
    {-0x1.b941ee770436bp-56, 0x1.2dbf557b0df43p-1},
    {-0x1.b941ee770436bp-56, 0x1.2dbf557b0df43p-1},
    {0x1.6c3a5f12642c9p-57, 0x1.315da4434068bp-1},
    {0x1.6c3a5f12642c9p-57, 0x1.315da4434068bp-1},
    {-0x1.f01ab6065515cp-56, 0x1.35028ad9d8c86p-1},
    {0x1.21512aa596ea3p-55, 0x1.38ae2171976e7p-1},
    {0x1.21512aa596ea3p-55, 0x1.38ae2171976e7p-1},
    {0x1.1930603d87b6ep-56, 0x1.3c6080c36bfb5p-1},
    {0x1.1930603d87b6ep-56, 0x1.3c6080c36bfb5p-1},
    {0x1.86cf0f38b461ap-57, 0x1.4019c2125ca93p-1},
    {-0x1.84f481051f71ap-56, 0x1.43d9ff2f923c5p-1},
    {-0x1.84f481051f71ap-56, 0x1.43d9ff2f923c5p-1},
    {0x1.2541aca7d5844p-55, 0x1.47a1527e8a2d3p-1},
    {0x1.2541aca7d5844p-55, 0x1.47a1527e8a2d3p-1},
    {0x1.c457b531506f6p-55, 0x1.4b6fd6f970c1fp-1},
    {0x1.c457b531506f6p-55, 0x1.4b6fd6f970c1fp-1},
    {0x1.d749362382a77p-56, 0x1.4f45a835a4e19p-1},
    {0x1.d749362382a77p-56, 0x1.4f45a835a4e19p-1},
    {0x1.988ba4aea614dp-56, 0x1.5322e26867857p-1},
    {0x1.988ba4aea614dp-56, 0x1.5322e26867857p-1},
    {0x1.80bff3303dd48p-55, 0x1.5707a26bb8c66p-1},
    {0x1.80bff3303dd48p-55, 0x1.5707a26bb8c66p-1},
    {-0x1.6714fbcd8135bp-55, 0x1.5af405c3649ep-1},
    {-0x1.6714fbcd8135bp-55, 0x1.5af405c3649ep-1},
    {0x1.1c066d235ee63p-56, 0x1.5ee82aa24192p-1},
    {0.0, 0.0},
};

const LogRR LOG2_TABLE = {
    // -log2(r) with 128-bit precision generated by SageMath with:
    //
    // for i in range(1, 127):
    //   r = 2^-8 * ceil( 2^8 * (1 - 2^(-8)) / (1 + i*2^(-7)) );
    //   s, m, e = RealField(128)(r).log2().sign_mantissa_exponent();
    //   print("{Sign::POS,", e, ", MType({", hex(m % 2^64), ",", hex((m >> 64)
    //   % 2^64),
    //         "})},");
    /* .step_1 = */ {
        {Sign::POS, 0, MType(0)},
        {Sign::POS, -134, MType({0xe8c251630adb856a, 0xb963dd107b993ada})},
        {Sign::POS, -133, MType({0xa41b08fbe05f82d0, 0xba1f7430f9aab1b2})},
        {Sign::POS, -132, MType({0x1f06c085bc1b865d, 0x8c25c7262b57c149})},
        {Sign::POS, -132, MType({0x2e1c07f0438ebac0, 0xbb9ca64ecac6aaef})},
        {Sign::POS, -132, MType({0xaacc0e21d6541224, 0xeb75e8f8ff5ff022})},
        {Sign::POS, -131, MType({0x31514aef39ce6303, 0x8dd9953002a4e866})},
        {Sign::POS, -131, MType({0x50799beaaab2940c, 0xa62b07f3457c4070})},
        {Sign::POS, -131, MType({0xda288fc615a727dc, 0xbeb024b67dda6339})},
        {Sign::POS, -131, MType({0x22dbbaced44516ce, 0xcb0657cd5dbe4f6f})},
        {Sign::POS, -131, MType({0xd939dceecdd9ce05, 0xe3da945b878e27d0})},
        {Sign::POS, -131, MType({0x9596a8e2e84c8f45, 0xfce4aee0e88b2749})},
        {Sign::POS, -130, MType({0x243efd9325954cfe, 0x84bf1c673032495d})},
        {Sign::POS, -130, MType({0x91d79938e7226384, 0x916d6e1559a4b696})},
        {Sign::POS, -130, MType({0x22563c9ed9462091, 0x9e37db2866f2850b})},
        {Sign::POS, -130, MType({0x3a53ca1181015ada, 0xa4a7c31dc6f9a5d5})},
        {Sign::POS, -130, MType({0x3eb8023eed65d601, 0xb19d45fa1be70855})},
        {Sign::POS, -130, MType({0xce5cabbd2d753d9b, 0xb823018e3cfc25f0})},
        {Sign::POS, -130, MType({0x54dbf16fb0695ee3, 0xc544c055fde99333})},
        {Sign::POS, -130, MType({0x5196a85a067c6739, 0xcbe0e589e3f6042d})},
        {Sign::POS, -130, MType({0xf349845e48955078, 0xd930124bea9a2c66})},
        {Sign::POS, -130, MType({0x815ef705cfaef035, 0xdfe33d3fffa66037})},
        {Sign::POS, -130, MType({0x2ba704dcaa76f41d, 0xed61169f220e97f2})},
        {Sign::POS, -130, MType({0x2062f36bc14d0d93, 0xf42be9e9b09b3def})},
        {Sign::POS, -129, MType({0x132880194144b02b, 0x80ecdde7d30ea2ed})},
        {Sign::POS, -129, MType({0x54880de63812fd49, 0x845e706cafd1bf61})},
        {Sign::POS, -129, MType({0xa87c02eaf36e2c29, 0x8b4e029b1f8ac391})},
        {Sign::POS, -129, MType({0x9804237ec8d9431d, 0x8ecc164ea93841ae})},
        {Sign::POS, -129, MType({0x20f81ca95d9e7968, 0x924e69589e6b6268})},
        {Sign::POS, -129, MType({0x124bc6f1acf95dc4, 0x995ff71b8773432d})},
        {Sign::POS, -129, MType({0x5a5e8e21bff3336b, 0x9cef470aacfb7bf9})},
        {Sign::POS, -129, MType({0x4e53fa3329f65894, 0xa08300be1f651473})},
        {Sign::POS, -129, MType({0x2742d7296a39eed6, 0xa7b7dd96762cc3c7})},
        {Sign::POS, -129, MType({0xf359c5544bc5e134, 0xab591735abc724e4})},
        {Sign::POS, -129, MType({0x6b6c874dd96e1d75, 0xaefee78f75707221})},
        {Sign::POS, -129, MType({0x21006678c0a5c390, 0xb2a95a4cc313bb59})},
        {Sign::POS, -129, MType({0x6d40900b25024b32, 0xb6587b432e47501b})},
        {Sign::POS, -129, MType({0x89e2eb553b279b3d, 0xbdc4f8167955698f})},
        {Sign::POS, -129, MType({0xd58525aad392ca50, 0xc1826c8608fe9951})},
        {Sign::POS, -129, MType({0x54dbf16fb0695ee3, 0xc544c055fde99333})},
        {Sign::POS, -129, MType({0x88d5eae3326327bb, 0xc90c004926e9dbfb})},
        {Sign::POS, -129, MType({0x46dfa05bddfded8c, 0xccd83954b6359379})},
        {Sign::POS, -129, MType({0xbfe9dbebf2e8a45e, 0xd47fcb8c0852f0c0})},
        {Sign::POS, -129, MType({0x7b11f1c5160c515c, 0xd85b3fa7a3407fa8})},
        {Sign::POS, -129, MType({0x1339e5677ec44dd0, 0xdc3be2bd8d837f7f})},
        {Sign::POS, -129, MType({0xea2b8c7bb0ee9c8b, 0xe021c2cf17ed9bdb})},
        {Sign::POS, -129, MType({0xaec562332791fe38, 0xe40cee16a2ff21c4})},
        {Sign::POS, -129, MType({0x71682ebacca79cfa, 0xe7fd7308d6895b14})},
        {Sign::POS, -129, MType({0xa5ad5ce9fb5a7bb6, 0xebf36055e1abc61e})},
        {Sign::POS, -129, MType({0x3225190531a852c5, 0xefeec4eac371584e})},
        {Sign::POS, -129, MType({0xda8ad649da21eab0, 0xf3efaff29c559a77})},
        {Sign::POS, -129, MType({0x4c3e2ea7c15c3d1e, 0xf7f630d808fc2ada})},
        {Sign::POS, -129, MType({0xbcb9bfa9852e0d35, 0xfc02574686680cc6})},
        {Sign::POS, -128, MType({0xce032f41d1e774e8, 0x800a1995f0019518})},
        {Sign::POS, -128, MType({0x9b39ffeebc29372a, 0x8215ea5cd3e4c4c7})},
        {Sign::POS, -128, MType({0x87f95f1befb6f806, 0x8424a6335c777e0b})},
        {Sign::POS, -128, MType({0xb987b42e3bb332a1, 0x8636557862acb7ce})},
        {Sign::POS, -128, MType({0x139a7ba83bf2d136, 0x884b00aef726cec5})},
        {Sign::POS, -128, MType({0x50799beaaab2941, 0x8a62b07f3457c407})},
        {Sign::POS, -128, MType({0x8bd744617e9b7d52, 0x8c7d6db7169e0cda})},
        {Sign::POS, -128, MType({0x46ad444333ceb10, 0x8e9b414b5a92a606})},
        {Sign::POS, -128, MType({0xef4c737fba4f5d66, 0x90bc345861bf3d52})},
        {Sign::POS, -128, MType({0xae441c09d761c549, 0x92e050231df57d6f})},
        {Sign::POS, -128, MType({0x6e36aa9ce90a3879, 0x95079e1a0382dc79})},
        {Sign::POS, -128, MType({0xefca1a184e93809, 0x973227d6027ebd8a})},
        {Sign::POS, -128, MType({0xefca1a184e93809, 0x973227d6027ebd8a})},
        {Sign::POS, -128, MType({0x124bc6f1acf95dc4, 0x995ff71b8773432d})},
        {Sign::POS, -128, MType({0x352bea51e58ea9e8, 0x9b9115db83a3dd2d})},
        {Sign::POS, -128, MType({0x266d6cdc959153bc, 0x9dc58e347d37696d})},
        {Sign::POS, -128, MType({0x4527d82c8214ddca, 0x9ffd6a73a78eaf35})},
        {Sign::POS, -128, MType({0x404cabb76d600e3c, 0xa238b5160413106e})},
        {Sign::POS, -128, MType({0x404cabb76d600e3c, 0xa238b5160413106e})},
        {Sign::POS, -128, MType({0xcab7d2ec23f0eef3, 0xa47778c98bcc86a1})},
        {Sign::POS, -128, MType({0x761c48dd859de2d3, 0xa6b9c06e6211646b})},
        {Sign::POS, -128, MType({0x7fd3b7d7e5d148bb, 0xa8ff971810a5e181})},
        {Sign::POS, -128, MType({0xc27c6780d92b4d11, 0xab49080ecda53208})},
        {Sign::POS, -128, MType({0xdb502402c94092cd, 0xad961ed0cb91d406})},
        {Sign::POS, -128, MType({0xdb502402c94092cd, 0xad961ed0cb91d406})},
        {Sign::POS, -128, MType({0x3432ef6b732b6843, 0xafe6e71393eeda29})},
        {Sign::POS, -128, MType({0xbb324da7e046e792, 0xb23b6cc56cc84c99})},
        {Sign::POS, -128, MType({0xb21709ce430c8e24, 0xb493bc0ec9954243})},
        {Sign::POS, -128, MType({0xb21709ce430c8e24, 0xb493bc0ec9954243})},
        {Sign::POS, -128, MType({0xe91ad16ecff10111, 0xb6efe153c7e319f6})},
        {Sign::POS, -128, MType({0xce31e481cd797e79, 0xb94fe935b83e3eb5})},
        {Sign::POS, -128, MType({0xda3e961a96c580fa, 0xbbb3e094b3d228d3})},
        {Sign::POS, -128, MType({0xda3e961a96c580fa, 0xbbb3e094b3d228d3})},
        {Sign::POS, -128, MType({0xf396598aae91499a, 0xbe1bd4913f3fda43})},
        {Sign::POS, -128, MType({0xae4cceb0f621941b, 0xc087d28dfb2febb8})},
        {Sign::POS, -128, MType({0xae4cceb0f621941b, 0xc087d28dfb2febb8})},
        {Sign::POS, -128, MType({0x6c1855c42078f81b, 0xc2f7e831632b6670})},
        {Sign::POS, -128, MType({0x169535fb8bf577c8, 0xc56c23679b4d206e})},
        {Sign::POS, -128, MType({0x169535fb8bf577c8, 0xc56c23679b4d206e})},
        {Sign::POS, -128, MType({0x3b24cecc60217942, 0xc7e492644d64237e})},
        {Sign::POS, -128, MType({0x3dc2687fcf939696, 0xca6143a49626d820})},
        {Sign::POS, -128, MType({0x3dc2687fcf939696, 0xca6143a49626d820})},
        {Sign::POS, -128, MType({0xa62e6add1a901a0, 0xcce245f1031e41fa})},
        {Sign::POS, -128, MType({0x5bb6e23138ad51e1, 0xcf67a85fa1f89a04})},
        {Sign::POS, -128, MType({0x5bb6e23138ad51e1, 0xcf67a85fa1f89a04})},
        {Sign::POS, -128, MType({0x7fc60a5103092bae, 0xd1f17a5621fb01ac})},
        {Sign::POS, -128, MType({0xbfe9dbebf2e8a45e, 0xd47fcb8c0852f0c0})},
        {Sign::POS, -128, MType({0xbfe9dbebf2e8a45e, 0xd47fcb8c0852f0c0})},
        {Sign::POS, -128, MType({0x8e2d7d378127d823, 0xd712ac0cf811659d})},
        {Sign::POS, -128, MType({0x5c1a7f14b168b365, 0xd9aa2c3b0ea3cbc1})},
        {Sign::POS, -128, MType({0x5c1a7f14b168b365, 0xd9aa2c3b0ea3cbc1})},
        {Sign::POS, -128, MType({0xb7579f0f8d3d514b, 0xdc465cd155a90942})},
        {Sign::POS, -128, MType({0xb7579f0f8d3d514b, 0xdc465cd155a90942})},
        {Sign::POS, -128, MType({0xb087205eb55aea85, 0xdee74ee64b0c38d3})},
        {Sign::POS, -128, MType({0x424a2623d60dfb16, 0xe18d13ee805a4de3})},
        {Sign::POS, -128, MType({0x424a2623d60dfb16, 0xe18d13ee805a4de3})},
        {Sign::POS, -128, MType({0x4d3a591ae6854787, 0xe437bdbf5254459c})},
        {Sign::POS, -128, MType({0x4d3a591ae6854787, 0xe437bdbf5254459c})},
        {Sign::POS, -128, MType({0x8dcdb6b24c5c5cdf, 0xe6e75e91b9cca551})},
        {Sign::POS, -128, MType({0x33ac7d9ebba8a53c, 0xe99c090536ece983})},
        {Sign::POS, -128, MType({0x33ac7d9ebba8a53c, 0xe99c090536ece983})},
        {Sign::POS, -128, MType({0xfb2eede4b59d8959, 0xec55d022d80e3d27})},
        {Sign::POS, -128, MType({0xfb2eede4b59d8959, 0xec55d022d80e3d27})},
        {Sign::POS, -128, MType({0x308b454666de8f99, 0xef14c7605d60654c})},
        {Sign::POS, -128, MType({0x308b454666de8f99, 0xef14c7605d60654c})},
        {Sign::POS, -128, MType({0x8383cb0ce23bebd4, 0xf1d902a37aaa5085})},
        {Sign::POS, -128, MType({0x8383cb0ce23bebd4, 0xf1d902a37aaa5085})},
        {Sign::POS, -128, MType({0x64fc87b4a41f7b70, 0xf4a2964538813c67})},
        {Sign::POS, -128, MType({0x64fc87b4a41f7b70, 0xf4a2964538813c67})},
        {Sign::POS, -128, MType({0x3f5d7d82b65c5686, 0xf77197157665f689})},
        {Sign::POS, -128, MType({0x3f5d7d82b65c5686, 0xf77197157665f689})},
        {Sign::POS, -128, MType({0x6476077b9fbd41ae, 0xfa461a5e8f4b759d})},
        {Sign::POS, -128, MType({0x6476077b9fbd41ae, 0xfa461a5e8f4b759d})},
        {Sign::POS, -128, MType({0xe3909ffd0d61778, 0xfd2035e9221ef5d0})},
        {Sign::POS, 0, MType(0)},
    },
    // -log2(r) for the second step, generated by SageMath with:
    //
    // for i in range(-2^6, 2^7 + 1):
    //   r = 2^-16 * round( 2^16 / (1 + i*2^(-14)) );
    //   s, m, e = RealField(128)(r).log2().sign_mantissa_exponent();
    //   print("{Sign::NEG," if s == 1 else "{Sign::POS,", e, ",
    //         MType({", hex(m % 2^64), ",", hex((m >> 64) % 2^64), "})},");
    /* .step_2 = */
    {
        {Sign::NEG, -135, MType({0xb5cfed58337e848a, 0xb906155918954401})},
        {Sign::NEG, -135, MType({0xffaf2ac1b1d20910, 0xb6264958a3c7fa2b})},
        {Sign::NEG, -135, MType({0x52521a3950ea2ed8, 0xb34671e439aa448e})},
        {Sign::NEG, -135, MType({0xf87e1abdee10fd95, 0xb0668efb7ef48ab7})},
        {Sign::NEG, -135, MType({0xfbd43bbcc24c5e43, 0xad86a09e185af0e8})},
        {Sign::NEG, -135, MType({0x2f4f5d48f9796742, 0xaaa6a6cbaa8d57ce})},
        {Sign::NEG, -135, MType({0x3477fd67c1cab6b3, 0xa7c6a183da375c3d})},
        {Sign::NEG, -135, MType({0x7b4d33eb381fe558, 0xa4e690c64c0056f0})},
        {Sign::NEG, -135, MType({0x3ce25e48cb498dea, 0xa2067492a48b5c43})},
        {Sign::NEG, -135, MType({0x70b0fcc9e4330983, 0x9f264ce888773bed})},
        {Sign::NEG, -135, MType({0xbc9e4267d3189b22, 0x9c4619c79c5e80bf})},
        {Sign::NEG, -135, MType({0x5fb3d896326615c4, 0x9965db2f84d7705f})},
        {Sign::NEG, -135, MType({0x178b58311e96d323, 0x9685911fe6740b02})},
        {Sign::NEG, -135, MType({0x6bf8b6cf73d847, 0x93a53b9865c20b2a})},
        {Sign::NEG, -135, MType({0x7019f6e64a580a02, 0x90c4da98a74ae561})},
        {Sign::NEG, -135, MType({0xcb5733cf0eb4191d, 0x8de46e204f93c7f6})},
        {Sign::NEG, -135, MType({0x56148d4fc5e415b6, 0x8b03f62f031d9ab8})},
        {Sign::NEG, -135, MType({0xfe5370f425872623, 0x882372c46664feaf})},
        {Sign::NEG, -135, MType({0x21b72a1457ee70d6, 0x8542e3e01de24ddf})},
        {Sign::NEG, -135, MType({0xabff4f89968bed0b, 0x81aa211f1e332fcf})},
        {Sign::NEG, -136, MType({0x86410a676480a5a7, 0xfd92f0cf88d75f24})},
        {Sign::NEG, -136, MType({0x44280889021970e4, 0xf7d1886b2a876289})},
        {Sign::NEG, -136, MType({0x32eb139d9812090d, 0xf21009106a42bc14})},
        {Sign::NEG, -136, MType({0xbef9dd41e8e42810, 0xec4e72be90cd2d2d})},
        {Sign::NEG, -136, MType({0x689d08ca6c7c3eb1, 0xe68cc574e6e1e5d7})},
        {Sign::NEG, -136, MType({0x1ef259a7f69821d, 0xe0cb0132b5338423})},
        {Sign::NEG, -136, MType({0xe22cea71b7bb8467, 0xdb0925f7446c13a9})},
        {Sign::NEG, -136, MType({0xe5bb27303f542fe, 0xd54733c1dd2d0d04})},
        {Sign::NEG, -136, MType({0x57453c8d5dc64ce1, 0xcf852a91c80f553f})},
        {Sign::NEG, -136, MType({0x6cc7add1fc09ef92, 0xc9c30a664da33d56})},
        {Sign::NEG, -136, MType({0xe678d7280de1c07f, 0xc400d33eb67081a7})},
        {Sign::NEG, -136, MType({0x419bbeb2239bdc39, 0xbe3e851a4af6496d})},
        {Sign::NEG, -136, MType({0xd4676d1d81755809, 0xb87c1ff853ab2631})},
        {Sign::NEG, -136, MType({0xb69dfef7ac2e2890, 0xb2b9a3d818fd1349})},
        {Sign::NEG, -136, MType({0x9f72fa0a8fccabc0, 0xacf710b8e3517548})},
        {Sign::NEG, -136, MType({0xb8bfe6a3addb988e, 0xa7346699fb051978})},
        {Sign::NEG, -136, MType({0x67862c8ec9dcd60d, 0xa171a57aa86c3551})},
        {Sign::NEG, -136, MType({0x9bd3370909e28a6, 0x9baecd5a33d265ee})},
        {Sign::NEG, -136, MType({0xa96bc611b991419b, 0x95ebde37e57aaf84})},
        {Sign::NEG, -136, MType({0xa50bb80f203f0d62, 0x9028d813059f7cdc})},
        {Sign::NEG, -136, MType({0x4d36cd474f65a317, 0x8a65baeadc729ec5})},
        {Sign::NEG, -136, MType({0x779be241ef4874a3, 0x84a286beb21d4b8c})},
        {Sign::NEG, -137, MType({0xe76a962fa65ace3, 0xfdbe771b9d803cea})},
        {Sign::NEG, -137, MType({0xd3d35627464a5267, 0xf237b2aef4e62e5a})},
        {Sign::NEG, -137, MType({0x162ef4b0e838c363, 0xe6b0c035fa8b328c})},
        {Sign::NEG, -137, MType({0x77bb10b976b3b9ca, 0xdb299faf3e7cd74f})},
        {Sign::NEG, -137, MType({0x209853cee70bc58b, 0xcfa2511950b77014})},
        {Sign::NEG, -137, MType({0x63f9b57cbaf2e58d, 0xc41ad472c12614d3})},
        {Sign::NEG, -137, MType({0x4fca1c931bd6e6d6, 0xb89329ba1fa2a0fd})},
        {Sign::NEG, -137, MType({0x26d26e434a53490a, 0xad0b50edfbf5b265})},
        {Sign::NEG, -137, MType({0xc55e079078dc86a0, 0xa1834a0ce5d6a82d})},
        {Sign::NEG, -137, MType({0xf05b9d5bd28f540b, 0x95fb15156ceba1b5})},
        {Sign::NEG, -137, MType({0x8ef87f1a11cdb727, 0x8a72b20620c97d84})},
        {Sign::NEG, -138, MType({0x9d6870114c1183cf, 0xfdd441bb21e7b069})},
        {Sign::NEG, -138, MType({0x63d514fff97e86f3, 0xe6c2c33499ba16c4})},
        {Sign::NEG, -138, MType({0x11a381901eadd883, 0xcfb0e875c7cc5929})},
        {Sign::NEG, -138, MType({0xa9d69d37bc0a5bac, 0xb89eb17bcabe1857})},
        {Sign::NEG, -138, MType({0x2dc97c9ffefd2497, 0xa18c1e43c10c6898})},
        {Sign::NEG, -138, MType({0xdcdc8afcb2ac09a, 0x8a792ecac911cf92})},
        {Sign::NEG, -139, MType({0xdd454eb3a1489470, 0xe6cbc61c020c8446})},
        {Sign::NEG, -139, MType({0x878035864d84b319, 0xb8a476150dfe4470})},
        {Sign::NEG, -139, MType({0x7ce595cc53b8342c, 0x8a7c6d7af1de7942})},
        {Sign::NEG, -140, MType({0x4710b59049899141, 0xb8a7588fd29b1baa})},
        {Sign::NEG, -141, MType({0x5957f633309d74e3, 0xb8a8c9d8be9ae994})},
        {Sign::POS, 0, MType({0x0, 0x0})},
        {Sign::POS, -141, MType({0x8268aba030b1adf6, 0xb8abac81ab576f3b})},
        {Sign::POS, -140, MType({0x1511cba2fb213a10, 0xb8ad1de1ac9ea6a5})},
        {Sign::POS, -139, MType({0x6379fb9fd9bc6235, 0x8a82eb7708262500})},
        {Sign::POS, -139, MType({0xb6fe1bf601ee27d5, 0xb8b000b8c65957cc})},
        {Sign::POS, -139, MType({0x8c6e60693a14e6d0, 0xe6ddcebbd72d3f7f})},
        {Sign::POS, -138, MType({0xe9bcfd0c62eaa2ca, 0x8a862ac30095c084})},
        {Sign::POS, -138, MType({0x73b214209a5234a7, 0xa19dca8e85918b6d})},
        {Sign::POS, -138, MType({0x347d4ca3109fe4db, 0xb8b5c6c35e142a9b})},
        {Sign::POS, -138, MType({0x37a62c48783bb066, 0xcfce1f646dca7745})},
        {Sign::POS, -138, MType({0x794b6437fb56344, 0xe6e6d4749883fbe3})},
        {Sign::POS, -138, MType({0x1cb9a45ed90318e6, 0xfdffe5f6c232f658})},
        {Sign::POS, -137, MType({0xbc118e5dbbef7dbc, 0x8a8ca9f6e7762d0f})},
        {Sign::POS, -137, MType({0xb4c0fb9535907cf8, 0x96198f2e5173e93b})},
        {Sign::POS, -137, MType({0xc051d2c5f00a9bb9, 0xa1a6a2a3113fe246})},
        {Sign::POS, -137, MType({0x553269878c1e5110, 0xad33e4569918a8d5})},
        {Sign::POS, -137, MType({0xbc906750b0ce372c, 0xb8c1544a5b4e2caf})},
        {Sign::POS, -137, MType({0x4c50eaa63be294b6, 0xc44ef27fca41bdd8})},
        {Sign::POS, -137, MType({0xb6cb28db8c065b44, 0xcfdcbef858660da1})},
        {Sign::POS, -137, MType({0x70479336830ceb05, 0xdb6ab9b5783f2fc5})},
        {Sign::POS, -137, MType({0x2a458c831f6aeb49, 0xe6f8e2b89c629b7a})},
        {Sign::POS, -137, MType({0x6489ba5bd391e206, 0xf2873a0337772c8a})},
        {Sign::POS, -137, MType({0x13f6fda510aeec3b, 0xfe15bf96bc35246b})},
        {Sign::POS, -136, MType({0x2f9a0ef9e8250836, 0x84d239ba4eb315a9})},
        {Sign::POS, -136, MType({0x389019e822b70f1e, 0x8a99aacf26f2a8a7})},
        {Sign::POS, -136, MType({0x308beeffa12cf669, 0x9061330aa04f87ae})},
        {Sign::POS, -136, MType({0x9886a71b25a2085d, 0x9628d26d7448a43f})},
        {Sign::POS, -136, MType({0x70ba9cebe0b969c3, 0x9bf088f85c65a56b})},
        {Sign::POS, -136, MType({0xcd855dc705ea2bea, 0xa1b856ac1236e85b})},
        {Sign::POS, -136, MType({0x7736196b11afb331, 0xa7803b894f5580e0})},
        {Sign::POS, -136, MType({0x94c99761b8eab3d8, 0xad483790cd6339fa})},
        {Sign::POS, -136, MType({0x6194b8c040814736, 0xb3104ac3460a9668})},
        {Sign::POS, -136, MType({0xedde8d24c7a999cc, 0xb8d8752172fed130})},
        {Sign::POS, -136, MType({0xea6b01ebde42f1d0, 0xbea0b6ac0dfbde2f})},
        {Sign::POS, -136, MType({0x7ef732b69334cf50, 0xc4690f63d0c66aa1})},
        {Sign::POS, -136, MType({0x2ba86275fcfc2d72, 0xca317f49752bddae})},
        {Sign::POS, -136, MType({0xb56ea44e185bf99f, 0xcffa065db50258f6})},
        {Sign::POS, -136, MType({0x1d5c3bbeb6902bfe, 0xd5c2a4a14a28b920})},
        {Sign::POS, -136, MType({0xa2f2bb9e156b0f37, 0xdb8b5a14ee86965f})},
        {Sign::POS, -136, MType({0xd166eb8da06ab5ef, 0xe15426b95c0c4506})},
        {Sign::POS, -136, MType({0x97dc7bae4219de0f, 0xe71d0a8f4cb2d60f})},
        {Sign::POS, -136, MType({0x6c9a8e7698f416c4, 0xece605977a7c17a8})},
        {Sign::POS, -136, MType({0x7b3a20aa5289695e, 0xf2af17d29f7295c0})},
        {Sign::POS, -136, MType({0xddcf578ee2c2897b, 0xf878414175a99a93})},
        {Sign::POS, -136, MType({0xe10ebd96c3ec30ec, 0xfe4181e4b73d2f37})},
        {Sign::POS, -135, MType({0xa9b7baecb34ba577, 0x82056cde8f290e13})},
        {Sign::POS, -135, MType({0x2da910dc61c182da, 0x8430f56d5e1edfd1})},
        {Sign::POS, -135, MType({0xfaca09dc7e0ba8b5, 0x8715b5a8f27bed90})},
        {Sign::POS, -135, MType({0xd723876173c0947, 0x89fa818019a2cace})},
        {Sign::POS, -135, MType({0x4e6651df154e8f8c, 0x8cdf58f330b64515})},
        {Sign::POS, -135, MType({0xee54b77d3bc34b6d, 0x8fc43c0294dd8af3})},
        {Sign::POS, -135, MType({0xad07dde9b5f92cce, 0x92a92aaea3442c3d})},
        {Sign::POS, -135, MType({0x261aacf944b638f0, 0x958e24f7b91a1a53})},
        {Sign::POS, -135, MType({0x232f5d64a85b219d, 0x98732ade3393a868})},
        {Sign::POS, -135, MType({0xf3a958bb706093fc, 0x9b583c626fe98bc9})},
        {Sign::POS, -135, MType({0xc9eaa059e7b0333a, 0x9e3d5984cb58dc25})},
        {Sign::POS, -135, MType({0x1e154029663243c0, 0xa1228245a32313cf})},
        {Sign::POS, -135, MType({0x16515200e283d006, 0xa407b6a5548e1006})},
        {Sign::POS, -135, MType({0xf498168a3337ca4f, 0xa6ecf6a43ce4113d})},
        {Sign::POS, -135, MType({0x8a04a89f0548a10f, 0xa9d24242b973bb63})},
        {Sign::POS, -135, MType({0xafaad01f25772805, 0xacb7998127901623})},
        {Sign::POS, -135, MType({0xc4f47950543fe0b8, 0xaf9cfc5fe4908d31})},
        {Sign::POS, -135, MType({0x338655e677d0d3ec, 0xb2826adf4dd0f08e})},
        {Sign::POS, -135, MType({0xf8ac2ce19d009541, 0xb567e4ffc0b174cc})},
        {Sign::POS, -135, MType({0x344d5e7dd7b2f465, 0xb84d6ac19a96b35c})},
        {Sign::POS, -135, MType({0xbd6a217fb4598ec7, 0xbb32fc2538e9aaca})},
        {Sign::POS, -135, MType({0xbc21ff368f562b75, 0xbe18992af917bf0e})},
        {Sign::POS, -135, MType({0x4944139ccbf2cb9a, 0xc0fe41d33892b9cc})},
        {Sign::POS, -135, MType({0x1369970c8b67e6b5, 0xc3e3f61e54d0ca9c})},
        {Sign::POS, -135, MType({0x99b370e2d04a530, 0xc6c9b60cab4c8752})},
        {Sign::POS, -135, MType({0xb81c3d48aff589f, 0xc9af819e9984ec44})},
        {Sign::POS, -135, MType({0x9f22b80993be311b, 0xcc9558d47cfd5c90})},
        {Sign::POS, -135, MType({0xac29209c8d8985ae, 0xcf7b3baeb33da265})},
        {Sign::POS, -135, MType({0x3cbb6a520292351d, 0xd2612a2d99d1ef47})},
        {Sign::POS, -135, MType({0x43de9ae40507ef24, 0xd54724518e4adc56})},
        {Sign::POS, -135, MType({0x69677b902ea4df3a, 0xd82d2a1aee3d6a97})},
        {Sign::POS, -135, MType({0xdb7a3aff74967bd5, 0xdb133b8a17430339})},
        {Sign::POS, -135, MType({0x25990c82a0066ac6, 0xddf9589f66f977de})},
        {Sign::POS, -135, MType({0xd424aacf4babf55, 0xe0df815b3b0302dd})},
        {Sign::POS, -135, MType({0xf8e3e7eb5a7bdebb, 0xe30c278d9936c595})},
        {Sign::POS, -135, MType({0x5ef8bf5adf5deebe, 0xe5f264adb62d5810})},
        {Sign::POS, -135, MType({0x331d19965368fc82, 0xe8d8ad75590bdf92})},
        {Sign::POS, -135, MType({0x901c30c427e358b8, 0xebbf01e4df85219e})},
        {Sign::POS, -135, MType({0xaeac7e9857253b06, 0xeea561fca7504dc1})},
        {Sign::POS, -135, MType({0xe2113e5893ab5b40, 0xf18bcdbd0e28fdd7})},
        {Sign::POS, -135, MType({0x9a4efc80ae977826, 0xf472452671cf3654})},
        {Sign::POS, -135, MType({0x6bf3ba8319332c9f, 0xf758c83930076689})},
        {Sign::POS, -135, MType({0x1d732d302e75018b, 0xfa3f56f5a69a68ed})},
        {Sign::POS, -135, MType({0xba179c5dbcceec01, 0xfd25f15c33558362})},
        {Sign::POS, -134, MType({0x5543f53b8ad85039, 0x80064bb69a0533c0})},
        {Sign::POS, -134, MType({0xe971a5565b93cb67, 0x8179a4948347996b})},
        {Sign::POS, -134, MType({0x5b399644ba714691, 0x82ed0348045f379d})},
        {Sign::POS, -134, MType({0x5079f1e0ec4b8496, 0x846067d14c3b8982})},
        {Sign::POS, -134, MType({0x6aba4990a32e8873, 0x85d3d23089ce40b0})},
        {Sign::POS, -134, MType({0xe16770c3a404291c, 0x87474265ec0b4548})},
        {Sign::POS, -134, MType({0x1edb7ffb1d6b3eab, 0x88bab871a1e8b61c})},
        {Sign::POS, -134, MType({0x603243e1ba7c7865, 0x8a2e3453da5ee8cd})},
        {Sign::POS, -134, MType({0x57ea5c03ea4621dd, 0x8ba1b60cc46869f6})},
        {Sign::POS, -134, MType({0xd3534cbf43bd7fd8, 0x8d153d9c8f01fd4a})},
        {Sign::POS, -134, MType({0x62c8c8075dc91cd5, 0x8e88cb03692a9dbc})},
        {Sign::POS, -134, MType({0x4bb70a5e3db7b85, 0x8ffc5e4181e37d9e})},
        {Sign::POS, -134, MType({0xd3875ba32159547a, 0x916ff757083006c7})},
        {Sign::POS, -134, MType({0x5c94c80e7a8f66b1, 0x9286adfca91ba28d})},
        {Sign::POS, -134, MType({0x52d313c47b4f91db, 0x93fa514ba0517623})},
        {Sign::POS, -134, MType({0x80829e9f3957a4c3, 0x956dfa72866fc57d})},
        {Sign::POS, -134, MType({0x1cd4917972015ae7, 0x96e1a9718a824be5})},
        {Sign::POS, -134, MType({0x1af23c29ef3032da, 0x98555e48db96fcd2})},
        {Sign::POS, -134, MType({0xe7f7bf240be67b80, 0x99c918f8a8be040e})},
        {Sign::POS, -134, MType({0x2bbe3cd4f7d868fa, 0x9b3cd9812109c5dc})},
        {Sign::POS, -134, MType({0x8c75d6a4c5ae460d, 0x9cb09fe2738edf14})},
        {Sign::POS, -134, MType({0x750fb989c9a06186, 0x9e246c1ccf642550})},
        {Sign::POS, -134, MType({0xde787e244901bdf9, 0x9f983e3063a2a709})},
        {Sign::POS, -134, MType({0x1ba3205ff729efa4, 0xa10c161d5f65abc0})},
        {Sign::POS, -134, MType({0xa864d2a038fb19cd, 0xa27ff3e3f1cab41b})},
        {Sign::POS, -134, MType({0xfb21f083a5fec56d, 0xa3f3d78449f17a11})},
        {Sign::POS, -134, MType({0x594c5552bcc377f5, 0xa567c0fe96fbf109})},
        {Sign::POS, -134, MType({0xaeb35a353fc5a503, 0xa6dbb053080e45fc})},
        {Sign::POS, -134, MType({0x67a5c05130c0f330, 0xa84fa581cc4edf9f})},
        {Sign::POS, -134, MType({0x4de5cafde1caf46f, 0xa9c3a08b12e65e81})},
        {Sign::POS, -134, MType({0x686fce3d160e88fd, 0xab37a16f0aff9d32})},
        {Sign::POS, -134, MType({0xde1375b3af6749a6, 0xacaba82de3c7b066})},
        {Sign::POS, -134, MType({0x243569048ac4affe, 0xadc2b114c632da56})},
        {Sign::POS, -134, MType({0xd6796227dcd39551, 0xaf36c21319b80ea2})},
        {Sign::POS, -134, MType({0xabc9265386172074, 0xb0aad8eccfb38d51})},
        {Sign::POS, -134, MType({0xcaac9f17896f2ce, 0xb21ef5a2175ac65e})},
        {Sign::POS, -134, MType({0x1c65a3c7f828972b, 0xb39318331fe56492})},
        {Sign::POS, -134, MType({0xabdc66446a4286d9, 0xb50740a0188d4daa})},
        {Sign::POS, -134, MType({0x2f3bbe8e8d72abec, 0xb67b6ee9308ea27b})},
        {Sign::POS, -134, MType({0xb67dbdd7f03d168c, 0xb7efa30e9727bf11})},
    },
    // -log2(r) for the third step, generated by SageMath with:
    //
    // for i in range(-80, 81):
    //   r = 2^-21 * round( 2^21 / (1 + i*2^(-21)) );
    //   s, m, e = RealField(128)(r).log2().sign_mantissa_exponent();
    //   print("{Sign::NEG," if (s == 1) else "{Sign::POS,", e, ",
    //         MType({", hex(m % 2^64), ",", hex((m >> 64) % 2^64), "})},");
    /* .step_3 = */
    {
        {Sign::NEG, -142, MType({0x26f2c63c0827ccbb, 0xe6d3a96b978fc16e})},
        {Sign::NEG, -142, MType({0x4b56fe667c8ec091, 0xe3f107a9fbfc50ca})},
        {Sign::NEG, -142, MType({0x647d76181aec10fc, 0xe10e65d14b937265})},
        {Sign::NEG, -142, MType({0x99e8f4d5379eca79, 0xde2bc3e18653b4f5})},
        {Sign::NEG, -142, MType({0xf07da89990c20623, 0xdb4921daac3ba730})},
        {Sign::NEG, -142, MType({0x4a8121848531851a, 0xd8667fbcbd49d7cd})},
        {Sign::NEG, -142, MType({0x679a4d854ae13619, 0xd583dd87b97cd580})},
        {Sign::NEG, -142, MType({0xe4d174072487a514, 0xd2a13b3ba0d32eff})},
        {Sign::NEG, -142, MType({0x3c90319d969b54be, 0xcfbe98d8734b7301})},
        {Sign::NEG, -142, MType({0xc6a173b09ba301e6, 0xccdbf65e30e43039})},
        {Sign::NEG, -142, MType({0xb8317428d7d8d06b, 0xc9f953ccd99bf55e})},
        {Sign::NEG, -142, MType({0x23cdb51bcc2061cd, 0xc716b1246d715125})},
        {Sign::NEG, -142, MType({0xf964fc78084fd515, 0xc4340e64ec62d241})},
        {Sign::NEG, -142, MType({0x6474fb15ccbb015, 0xc1516b8e566f076a})},
        {Sign::NEG, -142, MType({0xf525ef6d0b75b1c3, 0xbe6ec8a0ab947f51})},
        {Sign::NEG, -142, MType({0x4e13532df7ee8da7, 0xbb8c259bebd1c8ae})},
        {Sign::NEG, -142, MType({0x76832500d72a9027, 0xb8a9828017257233})},
        {Sign::NEG, -142, MType({0xb14a3d285e592ba0, 0xb5c6df4d2d8e0a95})},
        {Sign::NEG, -142, MType({0x1e9e9dc9711f6e20, 0xb2e43c032f0a2089})},
        {Sign::NEG, -142, MType({0xbc176e974f255fac, 0xb00198a21b9842c1})},
        {Sign::NEG, -142, MType({0x64acf87fc0f648e6, 0xad1ef529f336fff3})},
        {Sign::NEG, -142, MType({0xd0b8a1574433e1f8, 0xaa3c519ab5e4e6d1})},
        {Sign::NEG, -142, MType({0x95f4e785371c69a9, 0xa759adf463a08610})},
        {Sign::NEG, -142, MType({0x277d5db00363a46f, 0xa4770a36fc686c63})},
        {Sign::NEG, -142, MType({0xd5cea669485ec36c, 0xa1946662803b287c})},
        {Sign::NEG, -142, MType({0xcec66fda04833322, 0x9eb1c276ef174910})},
        {Sign::NEG, -142, MType({0x1da36f6ebe3851db, 0x9bcf1e7448fb5cd2})},
        {Sign::NEG, -142, MType({0xab055d83abfc0d82, 0x98ec7a5a8de5f273})},
        {Sign::NEG, -142, MType({0x3cecf110dbda68e9, 0x9609d629bdd598a8})},
        {Sign::NEG, -142, MType({0x76bbdb565a37e84b, 0x932731e1d8c8de22})},
        {Sign::NEG, -142, MType({0xd934c38857eee4f3, 0x90448d82debe5194})},
        {Sign::NEG, -142, MType({0xc27b427b4fbfc7db, 0x8d61e90ccfb481b1})},
        {Sign::NEG, -142, MType({0x6e13de502b142b39, 0x8a7f447faba9fd2b})},
        {Sign::NEG, -142, MType({0xf4e406206614e2ba, 0x879c9fdb729d52b3})},
        {Sign::NEG, -142, MType({0x4d320daa3312ea6c, 0x84b9fb20248d10fd})},
        {Sign::NEG, -142, MType({0x4aa528fc9d433c1a, 0x81d7564dc177c6b9})},
        {Sign::NEG, -143, MType({0x3c8ad047559b1622, 0xfde962c892b80533})},
        {Sign::NEG, -143, MType({0xacf765a8fc5bcc31, 0xf82418c77870a69f})},
        {Sign::NEG, -143, MType({0xbe238832edd27f20, 0xf25ece9834168f1a})},
        {Sign::NEG, -143, MType({0x2644bfca329b708, 0xec99843ac5a6dc07})},
        {Sign::NEG, -143, MType({0xc6d05a788e614744, 0xe6d439af2d1eaac6})},
        {Sign::NEG, -143, MType({0x133fe9cc57a8c1d0, 0xe10eeef56a7b18bc})},
        {Sign::NEG, -143, MType({0xaa4cb429195fb5dd, 0xdb49a40d7db94348})},
        {Sign::NEG, -143, MType({0x951ef239abbb959, 0xd58458f766d647ce})},
        {Sign::NEG, -143, MType({0x686c430c89143d35, 0xcfbf0db325cf43ad})},
        {Sign::NEG, -143, MType({0xba79c248afd42c12, 0xc9f9c240baa15447})},
        {Sign::NEG, -143, MType({0xad19e0a92f115327, 0xc43476a0254996fd})},
        {Sign::NEG, -143, MType({0xa8ad6ac3b0c99520, 0xbe6f2ad165c5292f})},
        {Sign::NEG, -143, MType({0xd0567d4a9cc5e6a1, 0xb8a9ded47c11283d})},
        {Sign::NEG, -143, MType({0x1f87c654b231443, 0xb2e492a9682ab188})},
        {Sign::NEG, -143, MType({0xd6380b08358051bc, 0xad1f46502a0ee26d})},
        {Sign::NEG, -143, MType({0xa07b024d26d391f6, 0xa759f9c8c1bad84e})},
        {Sign::NEG, -143, MType({0x6ee868cb69e3a7d8, 0xa194ad132f2bb089})},
        {Sign::NEG, -143, MType({0xa6869eff6682f73, 0x9bcf602f725e887d})},
        {Sign::NEG, -143, MType({0xf6a44d559ccf3f61, 0x960a131d8b507d87})},
        {Sign::NEG, -143, MType({0x72066e1d30a8e210, 0x9044c5dd79fead08})},
        {Sign::NEG, -143, MType({0x75ba3245b1b856af, 0x8a7f786f3e66345c})},
        {Sign::NEG, -143, MType({0xb5ac020473ab198f, 0x84ba2ad2d88430e1})},
        {Sign::NEG, -144, MType({0x41127e3a88eb6741, 0xfde9ba1090ab7feb})},
        {Sign::NEG, -144, MType({0xbf80787522aca1c4, 0xf25f1e1f1baffdea})},
        {Sign::NEG, -144, MType({0xaf00688b14fa3adc, 0xe6d481d15210167b})},
        {Sign::NEG, -144, MType({0x4d72837c8ab4d1e5, 0xdb49e52733c60457})},
        {Sign::NEG, -144, MType({0x4e38ac27bb252090, 0xcfbf4820c0cc0236})},
        {Sign::NEG, -144, MType({0xda3661f9292f59e8, 0xc434aabdf91c4ad0})},
        {Sign::NEG, -144, MType({0x8fd0af9bdfd21488, 0xb8aa0cfedcb118de})},
        {Sign::NEG, -144, MType({0x82ee19a9abf0bfa5, 0xad1f6ee36b84a716})},
        {Sign::NEG, -144, MType({0x3cf68d5b5369a251, 0xa194d06ba591302f})},
        {Sign::NEG, -144, MType({0xbcd34f38c977647e, 0x960a31978ad0eede})},
        {Sign::NEG, -144, MType({0x76eee9c9605e2143, 0x8a7f92671b3e1dda})},
        {Sign::NEG, -145, MType({0xaa6a3887f0c803ab, 0xfde9e5b4ada5efae})},
        {Sign::NEG, -145, MType({0x6e25927e582ac191, 0xe6d4a5e27b136f13})},
        {Sign::NEG, -145, MType({0xe2ebcac2f3a8e9eb, 0xcfbf65579eb92f4a})},
        {Sign::NEG, -145, MType({0x9d9acc22d5690751, 0xb8aa2414188ba5bb})},
        {Sign::NEG, -145, MType({0x1e12604b6d4132ef, 0xa194e217e87f47cb})},
        {Sign::NEG, -145, MType({0xcf340d2acb9b92a9, 0x8a7f9f630e888add})},
        {Sign::NEG, -146, MType({0xdc5e49fbde3c520, 0xe6d4b7eb1537c8ae})},
        {Sign::NEG, -146, MType({0xc074c9557c01188, 0xb8aa2f9eb95b9332})},
        {Sign::NEG, -146, MType({0xf0f82818ff9b654f, 0x8a7fa5e109656009})},
        {Sign::NEG, -147, MType({0xd4cd612078bbe9b0, 0xb8aa35640a7c33eb})},
        {Sign::NEG, -148, MType({0xf08cf68f42e09fa0, 0xb8aa3846b33aaecf})},
        {Sign::POS, 0, MType({0x0, 0x0})},
        {Sign::POS, -148, MType({0x68bd0facdf0ddaaf, 0xb8aa3e0c0513f9b1})},
        {Sign::POS, -147, MType({0x192af653dd41575b, 0xb8aa40eeae2ec9b3})},
        {Sign::POS, -146, MType({0x3b5c89842e540a51, 0x8a7fb2dd018e4892})},
        {Sign::POS, -146, MType({0x34ad8ebdd8b2750c, 0xb8aa46b400c0bee3})},
        {Sign::POS, -146, MType({0x70b12bd698e5be74, 0xe6d4dbfc54c5dd1b})},
        {Sign::POS, -145, MType({0x8c7e424efbd90e1, 0x8a7fb95afeda5c46})},
        {Sign::POS, -145, MType({0x31b8eba774a1de77, 0xa19505707dd23344})},
        {Sign::POS, -145, MType({0xee400e8c68838733, 0xb8aa523ea755fe32})},
        {Sign::POS, -145, MType({0xe71fa0b5603bc2f, 0xcfbf9fc57b7147be})},
        {Sign::POS, -145, MType({0x7763c919d8ac65f1, 0xe6d4ee04fa2f9a92})},
        {Sign::POS, -145, MType({0x232b270bb6046ec1, 0xfdea3cfd239c815e})},
        {Sign::POS, -144, MType({0x106f39197e068972, 0x8a7fc656fbe1c368})},
        {Sign::POS, -144, MType({0x4a4a6f4012941bd9, 0x960a6e8bbb581acc})},
        {Sign::POS, -144, MType({0x5bb34c1120b3e54b, 0xa195171cd0370c34})},
        {Sign::POS, -144, MType({0x6bb6731392a3147a, 0xad1fc00a3a845cf9})},
        {Sign::POS, -144, MType({0x2be1268dcee3c8fc, 0xb8aa6953fa45d275})},
        {Sign::POS, -144, MType({0xd84158d5d50251a9, 0xc43512fa0f813201})},
        {Sign::POS, -144, MType({0x3765bda15d0ef0fa, 0xcfbfbcfc7a3c40fa})},
        {Sign::POS, -144, MType({0x9a5ddb55f9cc27d9, 0xdb4a675b3a7cc4b9})},
        {Sign::POS, -144, MType({0xdcba1c593d918775, 0xe6d512165048829b})},
        {Sign::POS, -144, MType({0x648be060e1e30a95, 0xf25fbd2dbba53ffd})},
        {Sign::POS, -144, MType({0x22658dc2f1bcf6e8, 0xfdea68a17c98c23b})},
        {Sign::POS, -143, MType({0x48ad5162fb4a236e, 0x84ba8a38c9946759})},
        {Sign::POS, -143, MType({0xdb7fe3789405ce3a, 0x8a7fe04effad9560})},
        {Sign::POS, -143, MType({0x91b56e2e4f2e5ed8, 0x90453693609acde3})},
        {Sign::POS, -143, MType({0xf8998880c3bb4d76, 0x960a8d05ec5ef390})},
        {Sign::POS, -143, MType({0xe2b878052f67efee, 0x9bcfe3a6a2fce918})},
        {Sign::POS, -143, MType({0x67df399193f707c0, 0xa1953a758477912b})},
        {Sign::POS, -143, MType({0xe51b89e4d5d095e1, 0xa75a917290d1ce78})},
        {Sign::POS, -143, MType({0xfcbbee4edbf9f47d, 0xad1fe89dc80e83b1})},
        {Sign::POS, -143, MType({0x964fbd58b168371b, 0xb2e53ff72a309387})},
        {Sign::POS, -143, MType({0xdea7276ca7acd135, 0xb8aa977eb73ae0aa})},
        {Sign::POS, -143, MType({0x47d33f7e7afc83a6, 0xbe6fef346f304dcd})},
        {Sign::POS, -143, MType({0x892603b377909123, 0xc43547185213bda0})},
        {Sign::POS, -143, MType({0x9f32660aa06239fb, 0xc9fa9f2a5fe812d6})},
        {Sign::POS, -143, MType({0xcbcc5504d7407f6c, 0xcfbff76a98b03021})},
        {Sign::POS, -143, MType({0x9608c44d06402ebe, 0xd5854fd8fc6ef834})},
        {Sign::POS, -143, MType({0xca3db5604a863477, 0xdb4aa8758b274dc1})},
        {Sign::POS, -143, MType({0x7a024036206c37d6, 0xe110014044dc137c})},
        {Sign::POS, -143, MType({0xfc2e9be890ff7ee3, 0xe6d55a3929902c17})},
        {Sign::POS, -143, MType({0xecdc275c60da1b53, 0xec9ab36039467a47})},
        {Sign::POS, -143, MType({0x2d6571e94056607f, 0xf2600cb57401e0c0})},
        {Sign::POS, -143, MType({0xe4664401fd1ca2a7, 0xf8256638d9c54234})},
        {Sign::POS, -143, MType({0x7dbba7dcb50b3fd7, 0xfdeabfea6a93815a})},
        {Sign::POS, -142, MType({0xd541f90d853c794b, 0x81d80ce51337c072})},
        {Sign::POS, -142, MType({0xb08f65392ce8b75b, 0x84bab9ec06ae11c5})},
        {Sign::POS, -142, MType({0x6e969a29f8462436, 0x879d670a0fae2600})},
        {Sign::POS, -142, MType({0xcfc8cbcaa2bf130c, 0x8a80143f2e396e7d})},
        {Sign::POS, -142, MType({0xb737e48c19421e68, 0x8d62c18b62515c98})},
        {Sign::POS, -142, MType({0x2a9689b997c50c0b, 0x90456eeeabf761ac})},
        {Sign::POS, -142, MType({0x52381fccc774d66b, 0x93281c690b2cef13})},
        {Sign::POS, -142, MType({0x7910cec1dd92dc10, 0x960ac9fa7ff37629})},
        {Sign::POS, -142, MType({0xcb5866bbaff34cb, 0x98ed77a30a4c684a})},
        {Sign::POS, -142, MType({0x9d5c02c80c702d11, 0x9bd02562aa3936d0})},
        {Sign::POS, -142, MType({0xdddad0536b56e775, 0x9eb2d3395fbb5318})},
        {Sign::POS, -142, MType({0xa3a9505d7f71247a, 0xa19581272ad42e7e})},
        {Sign::POS, -142, MType({0xe6dfbd5d210830d7, 0xa4782f2c0b853a5d})},
        {Sign::POS, -142, MType({0xc2372f447bdcfa45, 0xa75add4801cfe812})},
        {Sign::POS, -142, MType({0x73099fd532c14b05, 0xaa3d8b7b0db5a8f9})},
        {Sign::POS, -142, MType({0x5951eef483de2c37, 0xad2039c52f37ee6e})},
        {Sign::POS, -142, MType({0xf7abe6ff6da76f1e, 0xb002e826665829cd})},
        {Sign::POS, -142, MType({0xf354411ed47c5d7b, 0xb2e5969eb317cc74})},
        {Sign::POS, -142, MType({0x1428a99ba8f5911f, 0xb5c8452e157847c0})},
        {Sign::POS, -142, MType({0x44a7c4330edff2c8, 0xb8aaf3d48d7b0d0c})},
        {Sign::POS, -142, MType({0x91f1306a84e4e07b, 0xbb8da2921b218db6})},
        {Sign::POS, -142, MType({0x2bc58de40cdf7b6a, 0xbe705166be6d3b1c})},
        {Sign::POS, -142, MType({0x648680b254df1d99, 0xc1530052775f869a})},
        {Sign::POS, -142, MType({0xb136b5ace0d6f74d, 0xc435af5545f9e18e})},
        {Sign::POS, -142, MType({0xa979e6c434fad480, 0xc7185e6f2a3dbd56})},
        {Sign::POS, -142, MType({0x794df5600c90a5a, 0xc9fb0da0242c8b50})},
        {Sign::POS, -142, MType({0xa86d80814ac18cf1, 0xccddbce833c7bcd8})},
        {Sign::POS, -142, MType({0x8b8ac57a9cca2d56, 0xcfc06c475910c34e})},
        {Sign::POS, -142, MType({0xd314c7e03140001f, 0xd2a31bbd9409100f})},
        {Sign::POS, -142, MType({0xc3d4c40e20b5ec89, 0xd585cb4ae4b2147a})},
        {Sign::POS, -142, MType({0xc5351d729060644e, 0xd8687aef4b0d41ed})},
        {Sign::POS, -142, MType({0x614162e1e12e445d, 0xdb4b2aaac71c09c7})},
        {Sign::POS, -142, MType({0x44a652eadf8ede85, 0xde2dda7d58dfdd66})},
        {Sign::POS, -142, MType({0x3eb1e02af3e52c3c, 0xe1108a67005a2e29})},
        {Sign::POS, -142, MType({0x415335a253a82aa2, 0xe3f33a67bd8c6d6f})},
        {Sign::POS, -142, MType({0x611abb0833305fe1, 0xe6d5ea7f90780c97})},
    },
    // -log2(r) for the fourth step, generated by SageMath with:
    //
    // for i in range(-65, 65):
    //   r = 2^-28 * round( 2^28 / (1 + i*2^(-28)) );
    //   s, m, e = RealField(128)(r).log2().sign_mantissa_exponent();
    //   print("{Sign::NEG," if (s == 1) else "{Sign::POS,", e, ",
    //         MType({", hex(m % 2^64), ",", hex((m >> 64) % 2^64), "})},");
    /* .step_4 = */
    {
        {Sign::NEG, -149, MType({0xef1bffe565ce0a46, 0xbb8ce2990b5d0b90})},
        {Sign::NEG, -149, MType({0xbea3244560ca3d99, 0xb8aa39b807a576e4})},
        {Sign::NEG, -149, MType({0x8b91f71ceefa31a2, 0xb5c790d6d5c354df})},
        {Sign::NEG, -149, MType({0x9096e3d684001c0e, 0xb2e4e7f575b6a57b})},
        {Sign::NEG, -149, MType({0x86054c794367f36, 0xb0023f13e77f68b3})},
        {Sign::NEG, -149, MType({0x2d9cb33094afe4de, 0xad1f96322b1d9e80})},
        {Sign::NEG, -149, MType({0x3afa673cfb3698f3, 0xaa3ced50409146dd})},
        {Sign::NEG, -149, MType({0x6b27d8033e4c6450, 0xa75a446e27da61c4})},
        {Sign::NEG, -149, MType({0xf8d36b84d52a477b, 0xa4779b8be0f8ef2f})},
        {Sign::NEG, -149, MType({0x1eab86ae37c03565, 0xa194f2a96becef1a})},
        {Sign::NEG, -149, MType({0x175e8d56deb4ce2c, 0x9eb249c6c8b6617d})},
        {Sign::NEG, -149, MType({0x1d9ae241436519da, 0x9bcfa0e3f7554653})},
        {Sign::NEG, -149, MType({0x6c0ee71adfe44325, 0x98ecf800f7c99d96})},
        {Sign::NEG, -149, MType({0x3d68fc7c2efb522f, 0x960a4f1dca136741})},
        {Sign::NEG, -149, MType({0xcc5781e8ac28e749, 0x9327a63a6e32a34d})},
        {Sign::NEG, -149, MType({0x5388d5ced3a0f5af, 0x9044fd56e42751b6})},
        {Sign::NEG, -149, MType({0xdab5588224c7e4a, 0x8d6254732bf17275})},
        {Sign::NEG, -149, MType({0x356d5d5915c94a70, 0x8a7fab8f45910584})},
        {Sign::NEG, -149, MType({0x57d48712c69a6a7, 0x879d02ab31060ade})},
        {Sign::NEG, -149, MType({0xb88970eae5341d60, 0x84ba59c6ee50827c})},
        {Sign::NEG, -149, MType({0x89402fcbbfe331bb, 0x81d7b0e27d706c5a})},
        {Sign::NEG, -150, MType({0x649fba0879ca348b, 0xfdea0ffbbccb90e3})},
        {Sign::NEG, -150, MType({0xdccd9edfbab6f777, 0xf824be3222612d78})},
        {Sign::NEG, -150, MType({0xf066b9aa4636478e, 0xf25f6c682ba1ae69})},
        {Sign::NEG, -150, MType({0x14c7b3cb21578781, 0xec9a1a9dd88d13ab})},
        {Sign::NEG, -150, MType({0xbf4d347b528f56e1, 0xe6d4c8d329235d30})},
        {Sign::NEG, -150, MType({0x6553e0c9e1b70799, 0xe10f77081d648aef})},
        {Sign::NEG, -150, MType({0x7c385b9bd80c1375, 0xdb4a253cb5509cdb})},
        {Sign::NEG, -150, MType({0x795745ac402f919d, 0xd584d370f0e792e9})},
        {Sign::NEG, -150, MType({0xd20d3d8c2625ac1b, 0xcfbf81a4d0296d0d})},
        {Sign::NEG, -150, MType({0xfbb6dfa297551554, 0xc9fa2fd853162b3c})},
        {Sign::NEG, -150, MType({0x6bb0c62ca2867d91, 0xc434de0b79adcd6b})},
        {Sign::NEG, -150, MType({0x9757893d57e40877, 0xbe6f8c3e43f0538d})},
        {Sign::NEG, -150, MType({0xf407bebdc8f8c28e, 0xb8aa3a70b1ddbd97})},
        {Sign::NEG, -150, MType({0xf71dfa6d08b016be, 0xb2e4e8a2c3760b7e})},
        {Sign::NEG, -150, MType({0x15f6cde02b5543ce, 0xad1f96d478b93d37})},
        {Sign::NEG, -150, MType({0xc5eec8824692d1e9, 0xa75a4505d1a752b4})},
        {Sign::NEG, -150, MType({0x7c6277947172081a, 0xa194f336ce404bec})},
        {Sign::NEG, -150, MType({0xaeae662dc45a61ce, 0x9bcfa1676e8428d2})},
        {Sign::NEG, -150, MType({0xd22f1d3b59110455, 0x960a4f97b272e95b})},
        {Sign::NEG, -150, MType({0x5c4123804ab83462, 0x9044fdc79a0c8d7c})},
        {Sign::NEG, -150, MType({0xc240fd95b5cecb89, 0x8a7fabf725511528})},
        {Sign::NEG, -150, MType({0x798b2deab82fadc4, 0x84ba5a2654408055})},
        {Sign::NEG, -151, MType({0xeef86988e2227ddb, 0xfdea10aa4db59ded})},
        {Sign::NEG, -151, MType({0x62e1207c0209b090, 0xf25f6d073a400203})},
        {Sign::NEG, -151, MType({0x3989789113ec7bee, 0xe6d4c9636e202cd4})},
        {Sign::NEG, -151, MType({0x5daa65565e562909, 0xdb4a25bee9561e49})},
        {Sign::NEG, -151, MType({0xb9fcd6062a84acbd, 0xcfbf8219abe1d64b})},
        {Sign::NEG, -151, MType({0x3939b586c46792b3, 0xc434de73b5c354c4})},
        {Sign::NEG, -151, MType({0xc619ea6a7a9ee85e, 0xb8aa3acd06fa999b})},
        {Sign::NEG, -151, MType({0x4b5656ef9e7a27fd, 0xad1f97259f87a4bb})},
        {Sign::NEG, -151, MType({0xb3a7d90083f7239c, 0xa194f37d7f6a760b})},
        {Sign::NEG, -151, MType({0xe9c74a3381c0f016, 0x960a4fd4a6a30d75})},
        {Sign::NEG, -151, MType({0xd86d7fcaf12ed012, 0x8a7fac2b15316ae2})},
        {Sign::NEG, -152, MType({0xd4a6956a5c863e0f, 0xfdea1101962b1c76})},
        {Sign::NEG, -152, MType({0x1462ef192f547877, 0xe6d4c9ab909eeed1})},
        {Sign::NEG, -152, MType({0x45819d2f1d72eb8b, 0xcfbf825419be4ca6})},
        {Sign::NEG, -152, MType({0x3d742790eedbe719, 0xb8aa3afb318935c8})},
        {Sign::NEG, -152, MType({0xd1ac0d7b70d74492, 0xa194f3a0d7ffaa08})},
        {Sign::NEG, -152, MType({0xd79ac58375f83d0c, 0x8a7fac450d21a939})},
        {Sign::NEG, -153, MType({0x49637b2bac367e87, 0xe6d4c9cfa1de665a})},
        {Sign::NEG, -153, MType({0x1cc4b5eedcc78b35, 0xb8aa3b1246d08f69})},
        {Sign::NEG, -153, MType({0xd43bf48a42745836, 0x8a7fac520919cd43})},
        {Sign::NEG, -154, MType({0x3557bdcf592619eb, 0xb8aa3b1dd1743f1c})},
        {Sign::NEG, -155, MType({0x6bdc2e83d3ebb0c4, 0xb8aa3b2396c617ae})},
        {Sign::POS, 0, MType({0x0, 0x0})},
        {Sign::POS, -155, MType({0x2d5b40050e44e8ab, 0xb8aa3b2f2169ca44})},
        {Sign::POS, -154, MType({0xb8560371b8f04afe, 0xb8aa3b34e6bba447})},
        {Sign::POS, -153, MType({0xc79a43ccc70459cc, 0x8a7fac6c010a1f14})},
        {Sign::POS, -153, MType({0x22c25632f519f77f, 0xb8aa3b40715f59c0})},
        {Sign::POS, -153, MType({0x42c10a314e35fb9e, 0xe6d4ca17c45d8282})},
        {Sign::POS, -152, MType({0xbe5a212ed7b949e4, 0x8a7fac78fd024cdb})},
        {Sign::POS, -152, MType({0x12dcf94ef5c5b918, 0xa194f3e7892a4fde})},
        {Sign::POS, -152, MType({0x49781013e57110ce, 0xb8aa3b5786a6ca76})},
        {Sign::POS, -152, MType({0x8cba70c085c12cb3, 0xcfbf82c8f577bcd2})},
        {Sign::POS, -152, MType({0x7332f3fb09328b8, 0xe6d4ca3bd59d2721})},
        {Sign::POS, -152, MType({0xe37168243a9d8b14, 0xfdea11b02717098f})},
        {Sign::POS, -151, MType({0xa602205479b93722, 0x8a7fac92f4f2b226})},
        {Sign::POS, -151, MType({0xb5bd735852c0d583, 0x960a504e8f041bc3})},
        {Sign::POS, -151, MType({0x363248630b0d812d, 0xa194f40ae1bfc1b6})},
        {Sign::POS, -151, MType({0x3ca83f0e02b823c0, 0xad1f97c7ed25a415})},
        {Sign::POS, -151, MType({0xde66fb46974bc4fd, 0xb8aa3b85b135c2f7})},
        {Sign::POS, -151, MType({0x30b6254e23c69fc2, 0xc434df442df01e75})},
        {Sign::POS, -151, MType({0x48dd69ba009b370c, 0xcfbf83036354b6a4})},
        {Sign::POS, -151, MType({0x3c24797383b16af5, 0xdb4a26c351638b9c})},
        {Sign::POS, -151, MType({0x1fd309b800678db7, 0xe6d4ca83f81c9d74})},
        {Sign::POS, -151, MType({0x930d418c79378a3, 0xf25f6e45577fec43})},
        {Sign::POS, -151, MType({0xd85967b2783a12c, 0xfdea12076f8d7820})},
        {Sign::POS, -150, MType({0x210c898c360016ed, 0x84ba5ae52022a091})},
        {Sign::POS, -150, MType({0x5e19883eef2605ab, 0x8a7facc6e4d3a3b0})},
        {Sign::POS, -150, MType({0x488dacc6629300ae, 0x9044fea905d9c579})},
        {Sign::POS, -150, MType({0x6b0cdebd3264e3e3, 0x960a508b833505f7})},
        {Sign::POS, -150, MType({0x503b07e7ff788dc2, 0x9bcfa26e5ce56536})},
        {Sign::POS, -150, MType({0x82bc1435696a69d1, 0xa194f45192eae341})},
        {Sign::POS, -150, MType({0x8d33f1be0e96fb1f, 0xa75a463525458024})},
        {Sign::POS, -150, MType({0xfa4690c48c1b66c9, 0xad1f981913f53bea})},
        {Sign::POS, -150, MType({0x5497e3b57dd5fe75, 0xb2e4e9fd5efa16a0})},
        {Sign::POS, -150, MType({0x26cbdf277e66cad5, 0xb8aa3be206541050})},
        {Sign::POS, -150, MType({0xfb8679db27301625, 0xbe6f8dc70a032905})},
        {Sign::POS, -150, MType({0x5d6bacbb1056f6aa, 0xc434dfac6a0760cd})},
        {Sign::POS, -150, MType({0xd71f72dbd0c3d936, 0xc9fa31922660b7b1})},
        {Sign::POS, -150, MType({0xf345c97bfe230ba2, 0xcfbf83783f0f2dbe})},
        {Sign::POS, -150, MType({0x3c82b0042ce54751, 0xd584d55eb412c300})},
        {Sign::POS, -150, MType({0x3d7a2806f0403bae, 0xdb4a2745856b7781})},
        {Sign::POS, -150, MType({0x80d03540da2f18ae, 0xe10f792cb3194b4d})},
        {Sign::POS, -150, MType({0x9128dd987b73194f, 0xe6d4cb143d1c3e70})},
        {Sign::POS, -150, MType({0xf928291e63940e14, 0xec9a1cfc237450f5})},
        {Sign::POS, -150, MType({0x4372220d20e0e78a, 0xf25f6ee4662182e9})},
        {Sign::POS, -150, MType({0xfaaad4c9407040c7, 0xf824c0cd0523d455})},
        {Sign::POS, -150, MType({0xa9764fe14e20e9e4, 0xfdea12b6007b4547})},
        {Sign::POS, -149, MType({0xed3c5206ea4d3942, 0x81d7b24fac13eae4})},
        {Sign::POS, -149, MType({0xc2af218aea6da27, 0x84ba5b448614c2f4})},
        {Sign::POS, -149, MType({0xf6d912ac383aaeba, 0x879d04398e402ad6})},
        {Sign::POS, -149, MType({0x7298bf5cca8b3d95, 0x8a7fad2ec4962293})},
        {Sign::POS, -149, MType({0x44bc04daa8808214, 0x8d6256242916aa2f})},
        {Sign::POS, -149, MType({0x3294f0eb14683198, 0x9044ff19bbc1c1b0})},
        {Sign::POS, -149, MType({0x17592684ff600c3, 0x9327a80f7c97691c})},
        {Sign::POS, -149, MType({0x76aff9419c43e8b9, 0x960a51056b97a078})},
        {Sign::POS, -149, MType({0x5796367b39d26c63, 0x98ecf9fb88c267cb})},
        {Sign::POS, -149, MType({0x697a5c2e6888ddaa, 0x9bcfa2f1d417bf1a})},
        {Sign::POS, -149, MType({0x71ae7d8967b5a2b7, 0x9eb24be84d97a66b})},
        {Sign::POS, -149, MType({0x3584aecf760e7b39, 0xa194f4def5421dc4})},
        {Sign::POS, -149, MType({0x7a4f0558d1b0c59e, 0xa4779dd5cb17252a})},
        {Sign::POS, -149, MType({0x55f9792b821c455, 0xa75a46cccf16bca4})},
        {Sign::POS, -149, MType({0x9c087cff664ee311, 0xaa3cefc40140e436})},
        {Sign::POS, -149, MType({0x39bce36188dfc04, 0xad1f98bb61959be8})},
        {Sign::POS, -149, MType({0x16ba4e30a9d9d21, 0xb00241b2f014e3be})},
        {Sign::POS, -149, MType({0x5aca1bc777a54d5e, 0xb2e4eaaaacbebbbe})},
        {Sign::POS, -149, MType({0xd5094eb99a35d1f0, 0xb5c793a2979323ee})},
        {Sign::POS, -149, MType({0x357b5aa4ac49738d, 0xb8aa3c9ab0921c55})},
    }};

// > P = fpminimax(log2(1 + x)/x, 3, [|128...|], [-0x1.0002143p-29 , 0x1p-29]);
// > P;
// > dirtyinfnorm(log2(1 + x)/x - P, [-0x1.0002143p-29 , 0x1p-29]);
// 0x1.27ad5...p-121
const Float128 BIG_COEFFS[4]{
    {Sign::NEG, -129, MType({0x3eccf6940d66bbcc, 0xb8aa3b295c2b21e3})},
    {Sign::POS, -129, MType({0xee39a6d649394bb1, 0xf6384ee1d01febc9})},
    {Sign::NEG, -128, MType({0xbe87fed067ea2ad5, 0xb8aa3b295c17f0bb})},
    {Sign::POS, -127, MType({0xbe87fed0691d3e3f, 0xb8aa3b295c17f0bb})},
};

// Reuse the output of the fast pass range reduction.
// -2^-8 <= m_x < 2^-7
double log2_accurate(int e_x, int index, double m_x) {

  Float128 sum(static_cast<float>(e_x));
  sum = fputil::quick_add(sum, LOG2_TABLE.step_1[index]);

  Float128 v_f128 = log_range_reduction(m_x, LOG2_TABLE, sum);

  // Polynomial approximation
  Float128 p = fputil::quick_mul(v_f128, BIG_COEFFS[0]);
  p = fputil::quick_mul(v_f128, fputil::quick_add(p, BIG_COEFFS[1]));
  p = fputil::quick_mul(v_f128, fputil::quick_add(p, BIG_COEFFS[2]));
  p = fputil::quick_mul(v_f128, fputil::quick_add(p, BIG_COEFFS[3]));

  Float128 r = fputil::quick_add(sum, p);

  return static_cast<double>(r);
}

} // namespace

LLVM_LIBC_FUNCTION(double, log2, (double x)) {
  using FPBits_t = typename fputil::FPBits<double>;
  using Sign = fputil::Sign;
  FPBits_t xbits(x);
  uint64_t x_u = xbits.uintval();

  int x_e = -FPBits_t::EXP_BIAS;

  if (LIBC_UNLIKELY(xbits == FPBits_t::one())) {
    // log2(1.0) = +0.0
    return 0.0;
  }

  if (LIBC_UNLIKELY(xbits.uintval() < FPBits_t::min_normal().uintval() ||
                    xbits.uintval() > FPBits_t::max_normal().uintval())) {
    if (xbits.is_zero()) {
      // return -Inf and raise FE_DIVBYZERO.
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_DIVBYZERO);
      return FPBits_t::inf(Sign::NEG).get_val();
    }
    if (xbits.is_neg() && !xbits.is_nan()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits_t::build_quiet_nan().get_val();
    }
    if (xbits.is_inf_or_nan()) {
      return x;
    }
    // Normalize denormal inputs.
    xbits = FPBits_t(x * 0x1.0p52);
    x_e -= 52;
    x_u = xbits.uintval();
  }

  // log2(x) = log2(2^x_e * x_m)
  //         = x_e + log2(x_m)
  // Range reduction for log2(x_m):
  // For each x_m, we would like to find r such that:
  //   -2^-8 <= r * x_m - 1 < 2^-7
  int shifted = static_cast<int>(x_u >> 45);
  int index = shifted & 0x7F;
  double r = RD[index];

  // Add unbiased exponent. Add an extra 1 if the 8 leading fractional bits are
  // all 1's.
  x_e += static_cast<int>((x_u + (1ULL << 45)) >> 52);
  double e_x = static_cast<double>(x_e);

  // Set m = 1.mantissa.
  uint64_t x_m = (x_u & 0x000F'FFFF'FFFF'FFFFULL) | 0x3FF0'0000'0000'0000ULL;
  double m = FPBits_t(x_m).get_val();

  double u, u_sq, err;
  fputil::DoubleDouble r1;

  // Perform exact range reduction
#ifdef LIBC_TARGET_CPU_HAS_FMA
  u = fputil::multiply_add(r, m, -1.0); // exact
#else
  uint64_t c_m = x_m & 0x3FFF'E000'0000'0000ULL;
  double c = FPBits_t(c_m).get_val();
  u = fputil::multiply_add(r, m - c, CD[index]); // exact
#endif // LIBC_TARGET_CPU_HAS_FMA

  // Exact sum:
  //   r1.hi + r1.lo = e_x * log(2)_hi - log(r)_hi + u
  r1 = fputil::exact_add(LOG_R1[index].hi, u);

  // Error of u_sq = ulp(u^2);
  u_sq = u * u;
  // Total error is bounded by ~ C * ulp(u^2).
  err = u_sq * P_ERR;
  // Degree-7 minimax polynomial
  double p0 = fputil::multiply_add(u, LOG_COEFFS[1], LOG_COEFFS[0]);
  double p1 = fputil::multiply_add(u, LOG_COEFFS[3], LOG_COEFFS[2]);
  double p2 = fputil::multiply_add(u, LOG_COEFFS[5], LOG_COEFFS[4]);
  double p = fputil::polyeval(u_sq, LOG_R1[index].lo, p0, p1, p2);

  r1.lo += p;

  // Quick double-double multiplication:
  //   r2.hi + r2.lo ~ r1 * log2(e),
  // with error bounded by:
  //   4*ulp( ulp(r2.hi) )
  fputil::DoubleDouble r2 = fputil::quick_mult(r1, LOG2_E);
  fputil::DoubleDouble r3 = fputil::exact_add(e_x, r2.hi);
  r3.lo += r2.lo;

  // Overall, if we choose sufficiently large constant C, the total error is
  // bounded by (C * ulp(u^2)).

  // Lower bound from the result
  double left = r3.hi + (r3.lo - err);
  // Upper bound from the result
  double right = r3.hi + (r3.lo + err);

  // Ziv's test if fast pass is accurate enough.
  if (left == right)
    return left;

  return log2_accurate(x_e, index, u);
}

} // namespace LIBC_NAMESPACE
