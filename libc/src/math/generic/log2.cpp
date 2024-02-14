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
#include "src/__support/integer_literals.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include "common_constants.h"
#include "log_range_reduction.h"

namespace LIBC_NAMESPACE {

// 128-bit precision dyadic floating point numbers.
using Float128 = typename fputil::DyadicFloat<128>;
using Sign = fputil::Sign;
using LIBC_NAMESPACE::operator""_u128;

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
    //   print("{Sign::POS,", e, ", hex(m), "_u128},");
    /* .step_1 = */ {
        {Sign::POS, 0, 0_u128},
        {Sign::POS, -134, 0xb963dd107b993adae8c251630adb856a_u128},
        {Sign::POS, -133, 0xba1f7430f9aab1b2a41b08fbe05f82d0_u128},
        {Sign::POS, -132, 0x8c25c7262b57c1491f06c085bc1b865d_u128},
        {Sign::POS, -132, 0xbb9ca64ecac6aaef2e1c07f0438ebac0_u128},
        {Sign::POS, -132, 0xeb75e8f8ff5ff022aacc0e21d6541224_u128},
        {Sign::POS, -131, 0x8dd9953002a4e86631514aef39ce6303_u128},
        {Sign::POS, -131, 0xa62b07f3457c407050799beaaab2940c_u128},
        {Sign::POS, -131, 0xbeb024b67dda6339da288fc615a727dc_u128},
        {Sign::POS, -131, 0xcb0657cd5dbe4f6f22dbbaced44516ce_u128},
        {Sign::POS, -131, 0xe3da945b878e27d0d939dceecdd9ce05_u128},
        {Sign::POS, -131, 0xfce4aee0e88b27499596a8e2e84c8f45_u128},
        {Sign::POS, -130, 0x84bf1c673032495d243efd9325954cfe_u128},
        {Sign::POS, -130, 0x916d6e1559a4b69691d79938e7226384_u128},
        {Sign::POS, -130, 0x9e37db2866f2850b22563c9ed9462091_u128},
        {Sign::POS, -130, 0xa4a7c31dc6f9a5d53a53ca1181015ada_u128},
        {Sign::POS, -130, 0xb19d45fa1be708553eb8023eed65d601_u128},
        {Sign::POS, -130, 0xb823018e3cfc25f0ce5cabbd2d753d9b_u128},
        {Sign::POS, -130, 0xc544c055fde9933354dbf16fb0695ee3_u128},
        {Sign::POS, -130, 0xcbe0e589e3f6042d5196a85a067c6739_u128},
        {Sign::POS, -130, 0xd930124bea9a2c66f349845e48955078_u128},
        {Sign::POS, -130, 0xdfe33d3fffa66037815ef705cfaef035_u128},
        {Sign::POS, -130, 0xed61169f220e97f22ba704dcaa76f41d_u128},
        {Sign::POS, -130, 0xf42be9e9b09b3def2062f36bc14d0d93_u128},
        {Sign::POS, -129, 0x80ecdde7d30ea2ed132880194144b02b_u128},
        {Sign::POS, -129, 0x845e706cafd1bf6154880de63812fd49_u128},
        {Sign::POS, -129, 0x8b4e029b1f8ac391a87c02eaf36e2c29_u128},
        {Sign::POS, -129, 0x8ecc164ea93841ae9804237ec8d9431d_u128},
        {Sign::POS, -129, 0x924e69589e6b626820f81ca95d9e7968_u128},
        {Sign::POS, -129, 0x995ff71b8773432d124bc6f1acf95dc4_u128},
        {Sign::POS, -129, 0x9cef470aacfb7bf95a5e8e21bff3336b_u128},
        {Sign::POS, -129, 0xa08300be1f6514734e53fa3329f65894_u128},
        {Sign::POS, -129, 0xa7b7dd96762cc3c72742d7296a39eed6_u128},
        {Sign::POS, -129, 0xab591735abc724e4f359c5544bc5e134_u128},
        {Sign::POS, -129, 0xaefee78f757072216b6c874dd96e1d75_u128},
        {Sign::POS, -129, 0xb2a95a4cc313bb5921006678c0a5c390_u128},
        {Sign::POS, -129, 0xb6587b432e47501b6d40900b25024b32_u128},
        {Sign::POS, -129, 0xbdc4f8167955698f89e2eb553b279b3d_u128},
        {Sign::POS, -129, 0xc1826c8608fe9951d58525aad392ca50_u128},
        {Sign::POS, -129, 0xc544c055fde9933354dbf16fb0695ee3_u128},
        {Sign::POS, -129, 0xc90c004926e9dbfb88d5eae3326327bb_u128},
        {Sign::POS, -129, 0xccd83954b635937946dfa05bddfded8c_u128},
        {Sign::POS, -129, 0xd47fcb8c0852f0c0bfe9dbebf2e8a45e_u128},
        {Sign::POS, -129, 0xd85b3fa7a3407fa87b11f1c5160c515c_u128},
        {Sign::POS, -129, 0xdc3be2bd8d837f7f1339e5677ec44dd0_u128},
        {Sign::POS, -129, 0xe021c2cf17ed9bdbea2b8c7bb0ee9c8b_u128},
        {Sign::POS, -129, 0xe40cee16a2ff21c4aec562332791fe38_u128},
        {Sign::POS, -129, 0xe7fd7308d6895b1471682ebacca79cfa_u128},
        {Sign::POS, -129, 0xebf36055e1abc61ea5ad5ce9fb5a7bb6_u128},
        {Sign::POS, -129, 0xefeec4eac371584e3225190531a852c5_u128},
        {Sign::POS, -129, 0xf3efaff29c559a77da8ad649da21eab0_u128},
        {Sign::POS, -129, 0xf7f630d808fc2ada4c3e2ea7c15c3d1e_u128},
        {Sign::POS, -129, 0xfc02574686680cc6bcb9bfa9852e0d35_u128},
        {Sign::POS, -128, 0x800a1995f0019518ce032f41d1e774e8_u128},
        {Sign::POS, -128, 0x8215ea5cd3e4c4c79b39ffeebc29372a_u128},
        {Sign::POS, -128, 0x8424a6335c777e0b87f95f1befb6f806_u128},
        {Sign::POS, -128, 0x8636557862acb7ceb987b42e3bb332a1_u128},
        {Sign::POS, -128, 0x884b00aef726cec5139a7ba83bf2d136_u128},
        {Sign::POS, -128, 0x8a62b07f3457c407050799beaaab2941_u128},
        {Sign::POS, -128, 0x8c7d6db7169e0cda8bd744617e9b7d52_u128},
        {Sign::POS, -128, 0x8e9b414b5a92a606046ad444333ceb10_u128},
        {Sign::POS, -128, 0x90bc345861bf3d52ef4c737fba4f5d66_u128},
        {Sign::POS, -128, 0x92e050231df57d6fae441c09d761c549_u128},
        {Sign::POS, -128, 0x95079e1a0382dc796e36aa9ce90a3879_u128},
        {Sign::POS, -128, 0x973227d6027ebd8a0efca1a184e93809_u128},
        {Sign::POS, -128, 0x973227d6027ebd8a0efca1a184e93809_u128},
        {Sign::POS, -128, 0x995ff71b8773432d124bc6f1acf95dc4_u128},
        {Sign::POS, -128, 0x9b9115db83a3dd2d352bea51e58ea9e8_u128},
        {Sign::POS, -128, 0x9dc58e347d37696d266d6cdc959153bc_u128},
        {Sign::POS, -128, 0x9ffd6a73a78eaf354527d82c8214ddca_u128},
        {Sign::POS, -128, 0xa238b5160413106e404cabb76d600e3c_u128},
        {Sign::POS, -128, 0xa238b5160413106e404cabb76d600e3c_u128},
        {Sign::POS, -128, 0xa47778c98bcc86a1cab7d2ec23f0eef3_u128},
        {Sign::POS, -128, 0xa6b9c06e6211646b761c48dd859de2d3_u128},
        {Sign::POS, -128, 0xa8ff971810a5e1817fd3b7d7e5d148bb_u128},
        {Sign::POS, -128, 0xab49080ecda53208c27c6780d92b4d11_u128},
        {Sign::POS, -128, 0xad961ed0cb91d406db502402c94092cd_u128},
        {Sign::POS, -128, 0xad961ed0cb91d406db502402c94092cd_u128},
        {Sign::POS, -128, 0xafe6e71393eeda293432ef6b732b6843_u128},
        {Sign::POS, -128, 0xb23b6cc56cc84c99bb324da7e046e792_u128},
        {Sign::POS, -128, 0xb493bc0ec9954243b21709ce430c8e24_u128},
        {Sign::POS, -128, 0xb493bc0ec9954243b21709ce430c8e24_u128},
        {Sign::POS, -128, 0xb6efe153c7e319f6e91ad16ecff10111_u128},
        {Sign::POS, -128, 0xb94fe935b83e3eb5ce31e481cd797e79_u128},
        {Sign::POS, -128, 0xbbb3e094b3d228d3da3e961a96c580fa_u128},
        {Sign::POS, -128, 0xbbb3e094b3d228d3da3e961a96c580fa_u128},
        {Sign::POS, -128, 0xbe1bd4913f3fda43f396598aae91499a_u128},
        {Sign::POS, -128, 0xc087d28dfb2febb8ae4cceb0f621941b_u128},
        {Sign::POS, -128, 0xc087d28dfb2febb8ae4cceb0f621941b_u128},
        {Sign::POS, -128, 0xc2f7e831632b66706c1855c42078f81b_u128},
        {Sign::POS, -128, 0xc56c23679b4d206e169535fb8bf577c8_u128},
        {Sign::POS, -128, 0xc56c23679b4d206e169535fb8bf577c8_u128},
        {Sign::POS, -128, 0xc7e492644d64237e3b24cecc60217942_u128},
        {Sign::POS, -128, 0xca6143a49626d8203dc2687fcf939696_u128},
        {Sign::POS, -128, 0xca6143a49626d8203dc2687fcf939696_u128},
        {Sign::POS, -128, 0xcce245f1031e41fa0a62e6add1a901a0_u128},
        {Sign::POS, -128, 0xcf67a85fa1f89a045bb6e23138ad51e1_u128},
        {Sign::POS, -128, 0xcf67a85fa1f89a045bb6e23138ad51e1_u128},
        {Sign::POS, -128, 0xd1f17a5621fb01ac7fc60a5103092bae_u128},
        {Sign::POS, -128, 0xd47fcb8c0852f0c0bfe9dbebf2e8a45e_u128},
        {Sign::POS, -128, 0xd47fcb8c0852f0c0bfe9dbebf2e8a45e_u128},
        {Sign::POS, -128, 0xd712ac0cf811659d8e2d7d378127d823_u128},
        {Sign::POS, -128, 0xd9aa2c3b0ea3cbc15c1a7f14b168b365_u128},
        {Sign::POS, -128, 0xd9aa2c3b0ea3cbc15c1a7f14b168b365_u128},
        {Sign::POS, -128, 0xdc465cd155a90942b7579f0f8d3d514b_u128},
        {Sign::POS, -128, 0xdc465cd155a90942b7579f0f8d3d514b_u128},
        {Sign::POS, -128, 0xdee74ee64b0c38d3b087205eb55aea85_u128},
        {Sign::POS, -128, 0xe18d13ee805a4de3424a2623d60dfb16_u128},
        {Sign::POS, -128, 0xe18d13ee805a4de3424a2623d60dfb16_u128},
        {Sign::POS, -128, 0xe437bdbf5254459c4d3a591ae6854787_u128},
        {Sign::POS, -128, 0xe437bdbf5254459c4d3a591ae6854787_u128},
        {Sign::POS, -128, 0xe6e75e91b9cca5518dcdb6b24c5c5cdf_u128},
        {Sign::POS, -128, 0xe99c090536ece98333ac7d9ebba8a53c_u128},
        {Sign::POS, -128, 0xe99c090536ece98333ac7d9ebba8a53c_u128},
        {Sign::POS, -128, 0xec55d022d80e3d27fb2eede4b59d8959_u128},
        {Sign::POS, -128, 0xec55d022d80e3d27fb2eede4b59d8959_u128},
        {Sign::POS, -128, 0xef14c7605d60654c308b454666de8f99_u128},
        {Sign::POS, -128, 0xef14c7605d60654c308b454666de8f99_u128},
        {Sign::POS, -128, 0xf1d902a37aaa50858383cb0ce23bebd4_u128},
        {Sign::POS, -128, 0xf1d902a37aaa50858383cb0ce23bebd4_u128},
        {Sign::POS, -128, 0xf4a2964538813c6764fc87b4a41f7b70_u128},
        {Sign::POS, -128, 0xf4a2964538813c6764fc87b4a41f7b70_u128},
        {Sign::POS, -128, 0xf77197157665f6893f5d7d82b65c5686_u128},
        {Sign::POS, -128, 0xf77197157665f6893f5d7d82b65c5686_u128},
        {Sign::POS, -128, 0xfa461a5e8f4b759d6476077b9fbd41ae_u128},
        {Sign::POS, -128, 0xfa461a5e8f4b759d6476077b9fbd41ae_u128},
        {Sign::POS, -128, 0xfd2035e9221ef5d00e3909ffd0d61778_u128},
        {Sign::POS, 0, 0_u128},
    },
    // -log2(r) for the second step, generated by SageMath with:
    //
    // for i in range(-2^6, 2^7 + 1):
    //   r = 2^-16 * round( 2^16 / (1 + i*2^(-14)) );
    //   s, m, e = RealField(128)(r).log2().sign_mantissa_exponent();
    //   print("{Sign::NEG," if s == 1 else "{Sign::POS,", e, ",
    //         hex(m), "_u128},");
    /* .step_2 = */
    {
        {Sign::NEG, -135, 0xb906155918954401b5cfed58337e848a_u128},
        {Sign::NEG, -135, 0xb6264958a3c7fa2bffaf2ac1b1d20910_u128},
        {Sign::NEG, -135, 0xb34671e439aa448e52521a3950ea2ed8_u128},
        {Sign::NEG, -135, 0xb0668efb7ef48ab7f87e1abdee10fd95_u128},
        {Sign::NEG, -135, 0xad86a09e185af0e8fbd43bbcc24c5e43_u128},
        {Sign::NEG, -135, 0xaaa6a6cbaa8d57ce2f4f5d48f9796742_u128},
        {Sign::NEG, -135, 0xa7c6a183da375c3d3477fd67c1cab6b3_u128},
        {Sign::NEG, -135, 0xa4e690c64c0056f07b4d33eb381fe558_u128},
        {Sign::NEG, -135, 0xa2067492a48b5c433ce25e48cb498dea_u128},
        {Sign::NEG, -135, 0x9f264ce888773bed70b0fcc9e4330983_u128},
        {Sign::NEG, -135, 0x9c4619c79c5e80bfbc9e4267d3189b22_u128},
        {Sign::NEG, -135, 0x9965db2f84d7705f5fb3d896326615c4_u128},
        {Sign::NEG, -135, 0x9685911fe6740b02178b58311e96d323_u128},
        {Sign::NEG, -135, 0x93a53b9865c20b2a006bf8b6cf73d847_u128},
        {Sign::NEG, -135, 0x90c4da98a74ae5617019f6e64a580a02_u128},
        {Sign::NEG, -135, 0x8de46e204f93c7f6cb5733cf0eb4191d_u128},
        {Sign::NEG, -135, 0x8b03f62f031d9ab856148d4fc5e415b6_u128},
        {Sign::NEG, -135, 0x882372c46664feaffe5370f425872623_u128},
        {Sign::NEG, -135, 0x8542e3e01de24ddf21b72a1457ee70d6_u128},
        {Sign::NEG, -135, 0x81aa211f1e332fcfabff4f89968bed0b_u128},
        {Sign::NEG, -136, 0xfd92f0cf88d75f2486410a676480a5a7_u128},
        {Sign::NEG, -136, 0xf7d1886b2a87628944280889021970e4_u128},
        {Sign::NEG, -136, 0xf21009106a42bc1432eb139d9812090d_u128},
        {Sign::NEG, -136, 0xec4e72be90cd2d2dbef9dd41e8e42810_u128},
        {Sign::NEG, -136, 0xe68cc574e6e1e5d7689d08ca6c7c3eb1_u128},
        {Sign::NEG, -136, 0xe0cb0132b533842301ef259a7f69821d_u128},
        {Sign::NEG, -136, 0xdb0925f7446c13a9e22cea71b7bb8467_u128},
        {Sign::NEG, -136, 0xd54733c1dd2d0d040e5bb27303f542fe_u128},
        {Sign::NEG, -136, 0xcf852a91c80f553f57453c8d5dc64ce1_u128},
        {Sign::NEG, -136, 0xc9c30a664da33d566cc7add1fc09ef92_u128},
        {Sign::NEG, -136, 0xc400d33eb67081a7e678d7280de1c07f_u128},
        {Sign::NEG, -136, 0xbe3e851a4af6496d419bbeb2239bdc39_u128},
        {Sign::NEG, -136, 0xb87c1ff853ab2631d4676d1d81755809_u128},
        {Sign::NEG, -136, 0xb2b9a3d818fd1349b69dfef7ac2e2890_u128},
        {Sign::NEG, -136, 0xacf710b8e35175489f72fa0a8fccabc0_u128},
        {Sign::NEG, -136, 0xa7346699fb051978b8bfe6a3addb988e_u128},
        {Sign::NEG, -136, 0xa171a57aa86c355167862c8ec9dcd60d_u128},
        {Sign::NEG, -136, 0x9baecd5a33d265ee09bd3370909e28a6_u128},
        {Sign::NEG, -136, 0x95ebde37e57aaf84a96bc611b991419b_u128},
        {Sign::NEG, -136, 0x9028d813059f7cdca50bb80f203f0d62_u128},
        {Sign::NEG, -136, 0x8a65baeadc729ec54d36cd474f65a317_u128},
        {Sign::NEG, -136, 0x84a286beb21d4b8c779be241ef4874a3_u128},
        {Sign::NEG, -137, 0xfdbe771b9d803cea0e76a962fa65ace3_u128},
        {Sign::NEG, -137, 0xf237b2aef4e62e5ad3d35627464a5267_u128},
        {Sign::NEG, -137, 0xe6b0c035fa8b328c162ef4b0e838c363_u128},
        {Sign::NEG, -137, 0xdb299faf3e7cd74f77bb10b976b3b9ca_u128},
        {Sign::NEG, -137, 0xcfa2511950b77014209853cee70bc58b_u128},
        {Sign::NEG, -137, 0xc41ad472c12614d363f9b57cbaf2e58d_u128},
        {Sign::NEG, -137, 0xb89329ba1fa2a0fd4fca1c931bd6e6d6_u128},
        {Sign::NEG, -137, 0xad0b50edfbf5b26526d26e434a53490a_u128},
        {Sign::NEG, -137, 0xa1834a0ce5d6a82dc55e079078dc86a0_u128},
        {Sign::NEG, -137, 0x95fb15156ceba1b5f05b9d5bd28f540b_u128},
        {Sign::NEG, -137, 0x8a72b20620c97d848ef87f1a11cdb727_u128},
        {Sign::NEG, -138, 0xfdd441bb21e7b0699d6870114c1183cf_u128},
        {Sign::NEG, -138, 0xe6c2c33499ba16c463d514fff97e86f3_u128},
        {Sign::NEG, -138, 0xcfb0e875c7cc592911a381901eadd883_u128},
        {Sign::NEG, -138, 0xb89eb17bcabe1857a9d69d37bc0a5bac_u128},
        {Sign::NEG, -138, 0xa18c1e43c10c68982dc97c9ffefd2497_u128},
        {Sign::NEG, -138, 0x8a792ecac911cf920dcdc8afcb2ac09a_u128},
        {Sign::NEG, -139, 0xe6cbc61c020c8446dd454eb3a1489470_u128},
        {Sign::NEG, -139, 0xb8a476150dfe4470878035864d84b319_u128},
        {Sign::NEG, -139, 0x8a7c6d7af1de79427ce595cc53b8342c_u128},
        {Sign::NEG, -140, 0xb8a7588fd29b1baa4710b59049899141_u128},
        {Sign::NEG, -141, 0xb8a8c9d8be9ae9945957f633309d74e3_u128},
        {Sign::POS, 0, 0_u128},
        {Sign::POS, -141, 0xb8abac81ab576f3b8268aba030b1adf6_u128},
        {Sign::POS, -140, 0xb8ad1de1ac9ea6a51511cba2fb213a10_u128},
        {Sign::POS, -139, 0x8a82eb77082625006379fb9fd9bc6235_u128},
        {Sign::POS, -139, 0xb8b000b8c65957ccb6fe1bf601ee27d5_u128},
        {Sign::POS, -139, 0xe6ddcebbd72d3f7f8c6e60693a14e6d0_u128},
        {Sign::POS, -138, 0x8a862ac30095c084e9bcfd0c62eaa2ca_u128},
        {Sign::POS, -138, 0xa19dca8e85918b6d73b214209a5234a7_u128},
        {Sign::POS, -138, 0xb8b5c6c35e142a9b347d4ca3109fe4db_u128},
        {Sign::POS, -138, 0xcfce1f646dca774537a62c48783bb066_u128},
        {Sign::POS, -138, 0xe6e6d4749883fbe30794b6437fb56344_u128},
        {Sign::POS, -138, 0xfdffe5f6c232f6581cb9a45ed90318e6_u128},
        {Sign::POS, -137, 0x8a8ca9f6e7762d0fbc118e5dbbef7dbc_u128},
        {Sign::POS, -137, 0x96198f2e5173e93bb4c0fb9535907cf8_u128},
        {Sign::POS, -137, 0xa1a6a2a3113fe246c051d2c5f00a9bb9_u128},
        {Sign::POS, -137, 0xad33e4569918a8d5553269878c1e5110_u128},
        {Sign::POS, -137, 0xb8c1544a5b4e2cafbc906750b0ce372c_u128},
        {Sign::POS, -137, 0xc44ef27fca41bdd84c50eaa63be294b6_u128},
        {Sign::POS, -137, 0xcfdcbef858660da1b6cb28db8c065b44_u128},
        {Sign::POS, -137, 0xdb6ab9b5783f2fc570479336830ceb05_u128},
        {Sign::POS, -137, 0xe6f8e2b89c629b7a2a458c831f6aeb49_u128},
        {Sign::POS, -137, 0xf2873a0337772c8a6489ba5bd391e206_u128},
        {Sign::POS, -137, 0xfe15bf96bc35246b13f6fda510aeec3b_u128},
        {Sign::POS, -136, 0x84d239ba4eb315a92f9a0ef9e8250836_u128},
        {Sign::POS, -136, 0x8a99aacf26f2a8a7389019e822b70f1e_u128},
        {Sign::POS, -136, 0x9061330aa04f87ae308beeffa12cf669_u128},
        {Sign::POS, -136, 0x9628d26d7448a43f9886a71b25a2085d_u128},
        {Sign::POS, -136, 0x9bf088f85c65a56b70ba9cebe0b969c3_u128},
        {Sign::POS, -136, 0xa1b856ac1236e85bcd855dc705ea2bea_u128},
        {Sign::POS, -136, 0xa7803b894f5580e07736196b11afb331_u128},
        {Sign::POS, -136, 0xad483790cd6339fa94c99761b8eab3d8_u128},
        {Sign::POS, -136, 0xb3104ac3460a96686194b8c040814736_u128},
        {Sign::POS, -136, 0xb8d8752172fed130edde8d24c7a999cc_u128},
        {Sign::POS, -136, 0xbea0b6ac0dfbde2fea6b01ebde42f1d0_u128},
        {Sign::POS, -136, 0xc4690f63d0c66aa17ef732b69334cf50_u128},
        {Sign::POS, -136, 0xca317f49752bddae2ba86275fcfc2d72_u128},
        {Sign::POS, -136, 0xcffa065db50258f6b56ea44e185bf99f_u128},
        {Sign::POS, -136, 0xd5c2a4a14a28b9201d5c3bbeb6902bfe_u128},
        {Sign::POS, -136, 0xdb8b5a14ee86965fa2f2bb9e156b0f37_u128},
        {Sign::POS, -136, 0xe15426b95c0c4506d166eb8da06ab5ef_u128},
        {Sign::POS, -136, 0xe71d0a8f4cb2d60f97dc7bae4219de0f_u128},
        {Sign::POS, -136, 0xece605977a7c17a86c9a8e7698f416c4_u128},
        {Sign::POS, -136, 0xf2af17d29f7295c07b3a20aa5289695e_u128},
        {Sign::POS, -136, 0xf878414175a99a93ddcf578ee2c2897b_u128},
        {Sign::POS, -136, 0xfe4181e4b73d2f37e10ebd96c3ec30ec_u128},
        {Sign::POS, -135, 0x82056cde8f290e13a9b7baecb34ba577_u128},
        {Sign::POS, -135, 0x8430f56d5e1edfd12da910dc61c182da_u128},
        {Sign::POS, -135, 0x8715b5a8f27bed90faca09dc7e0ba8b5_u128},
        {Sign::POS, -135, 0x89fa818019a2cace0d723876173c0947_u128},
        {Sign::POS, -135, 0x8cdf58f330b645154e6651df154e8f8c_u128},
        {Sign::POS, -135, 0x8fc43c0294dd8af3ee54b77d3bc34b6d_u128},
        {Sign::POS, -135, 0x92a92aaea3442c3dad07dde9b5f92cce_u128},
        {Sign::POS, -135, 0x958e24f7b91a1a53261aacf944b638f0_u128},
        {Sign::POS, -135, 0x98732ade3393a868232f5d64a85b219d_u128},
        {Sign::POS, -135, 0x9b583c626fe98bc9f3a958bb706093fc_u128},
        {Sign::POS, -135, 0x9e3d5984cb58dc25c9eaa059e7b0333a_u128},
        {Sign::POS, -135, 0xa1228245a32313cf1e154029663243c0_u128},
        {Sign::POS, -135, 0xa407b6a5548e100616515200e283d006_u128},
        {Sign::POS, -135, 0xa6ecf6a43ce4113df498168a3337ca4f_u128},
        {Sign::POS, -135, 0xa9d24242b973bb638a04a89f0548a10f_u128},
        {Sign::POS, -135, 0xacb7998127901623afaad01f25772805_u128},
        {Sign::POS, -135, 0xaf9cfc5fe4908d31c4f47950543fe0b8_u128},
        {Sign::POS, -135, 0xb2826adf4dd0f08e338655e677d0d3ec_u128},
        {Sign::POS, -135, 0xb567e4ffc0b174ccf8ac2ce19d009541_u128},
        {Sign::POS, -135, 0xb84d6ac19a96b35c344d5e7dd7b2f465_u128},
        {Sign::POS, -135, 0xbb32fc2538e9aacabd6a217fb4598ec7_u128},
        {Sign::POS, -135, 0xbe18992af917bf0ebc21ff368f562b75_u128},
        {Sign::POS, -135, 0xc0fe41d33892b9cc4944139ccbf2cb9a_u128},
        {Sign::POS, -135, 0xc3e3f61e54d0ca9c1369970c8b67e6b5_u128},
        {Sign::POS, -135, 0xc6c9b60cab4c8752099b370e2d04a530_u128},
        {Sign::POS, -135, 0xc9af819e9984ec440b81c3d48aff589f_u128},
        {Sign::POS, -135, 0xcc9558d47cfd5c909f22b80993be311b_u128},
        {Sign::POS, -135, 0xcf7b3baeb33da265ac29209c8d8985ae_u128},
        {Sign::POS, -135, 0xd2612a2d99d1ef473cbb6a520292351d_u128},
        {Sign::POS, -135, 0xd54724518e4adc5643de9ae40507ef24_u128},
        {Sign::POS, -135, 0xd82d2a1aee3d6a9769677b902ea4df3a_u128},
        {Sign::POS, -135, 0xdb133b8a17430339db7a3aff74967bd5_u128},
        {Sign::POS, -135, 0xddf9589f66f977de25990c82a0066ac6_u128},
        {Sign::POS, -135, 0xe0df815b3b0302dd0d424aacf4babf55_u128},
        {Sign::POS, -135, 0xe30c278d9936c595f8e3e7eb5a7bdebb_u128},
        {Sign::POS, -135, 0xe5f264adb62d58105ef8bf5adf5deebe_u128},
        {Sign::POS, -135, 0xe8d8ad75590bdf92331d19965368fc82_u128},
        {Sign::POS, -135, 0xebbf01e4df85219e901c30c427e358b8_u128},
        {Sign::POS, -135, 0xeea561fca7504dc1aeac7e9857253b06_u128},
        {Sign::POS, -135, 0xf18bcdbd0e28fdd7e2113e5893ab5b40_u128},
        {Sign::POS, -135, 0xf472452671cf36549a4efc80ae977826_u128},
        {Sign::POS, -135, 0xf758c839300766896bf3ba8319332c9f_u128},
        {Sign::POS, -135, 0xfa3f56f5a69a68ed1d732d302e75018b_u128},
        {Sign::POS, -135, 0xfd25f15c33558362ba179c5dbcceec01_u128},
        {Sign::POS, -134, 0x80064bb69a0533c05543f53b8ad85039_u128},
        {Sign::POS, -134, 0x8179a4948347996be971a5565b93cb67_u128},
        {Sign::POS, -134, 0x82ed0348045f379d5b399644ba714691_u128},
        {Sign::POS, -134, 0x846067d14c3b89825079f1e0ec4b8496_u128},
        {Sign::POS, -134, 0x85d3d23089ce40b06aba4990a32e8873_u128},
        {Sign::POS, -134, 0x87474265ec0b4548e16770c3a404291c_u128},
        {Sign::POS, -134, 0x88bab871a1e8b61c1edb7ffb1d6b3eab_u128},
        {Sign::POS, -134, 0x8a2e3453da5ee8cd603243e1ba7c7865_u128},
        {Sign::POS, -134, 0x8ba1b60cc46869f657ea5c03ea4621dd_u128},
        {Sign::POS, -134, 0x8d153d9c8f01fd4ad3534cbf43bd7fd8_u128},
        {Sign::POS, -134, 0x8e88cb03692a9dbc62c8c8075dc91cd5_u128},
        {Sign::POS, -134, 0x8ffc5e4181e37d9e04bb70a5e3db7b85_u128},
        {Sign::POS, -134, 0x916ff757083006c7d3875ba32159547a_u128},
        {Sign::POS, -134, 0x9286adfca91ba28d5c94c80e7a8f66b1_u128},
        {Sign::POS, -134, 0x93fa514ba051762352d313c47b4f91db_u128},
        {Sign::POS, -134, 0x956dfa72866fc57d80829e9f3957a4c3_u128},
        {Sign::POS, -134, 0x96e1a9718a824be51cd4917972015ae7_u128},
        {Sign::POS, -134, 0x98555e48db96fcd21af23c29ef3032da_u128},
        {Sign::POS, -134, 0x99c918f8a8be040ee7f7bf240be67b80_u128},
        {Sign::POS, -134, 0x9b3cd9812109c5dc2bbe3cd4f7d868fa_u128},
        {Sign::POS, -134, 0x9cb09fe2738edf148c75d6a4c5ae460d_u128},
        {Sign::POS, -134, 0x9e246c1ccf642550750fb989c9a06186_u128},
        {Sign::POS, -134, 0x9f983e3063a2a709de787e244901bdf9_u128},
        {Sign::POS, -134, 0xa10c161d5f65abc01ba3205ff729efa4_u128},
        {Sign::POS, -134, 0xa27ff3e3f1cab41ba864d2a038fb19cd_u128},
        {Sign::POS, -134, 0xa3f3d78449f17a11fb21f083a5fec56d_u128},
        {Sign::POS, -134, 0xa567c0fe96fbf109594c5552bcc377f5_u128},
        {Sign::POS, -134, 0xa6dbb053080e45fcaeb35a353fc5a503_u128},
        {Sign::POS, -134, 0xa84fa581cc4edf9f67a5c05130c0f330_u128},
        {Sign::POS, -134, 0xa9c3a08b12e65e814de5cafde1caf46f_u128},
        {Sign::POS, -134, 0xab37a16f0aff9d32686fce3d160e88fd_u128},
        {Sign::POS, -134, 0xacaba82de3c7b066de1375b3af6749a6_u128},
        {Sign::POS, -134, 0xadc2b114c632da56243569048ac4affe_u128},
        {Sign::POS, -134, 0xaf36c21319b80ea2d6796227dcd39551_u128},
        {Sign::POS, -134, 0xb0aad8eccfb38d51abc9265386172074_u128},
        {Sign::POS, -134, 0xb21ef5a2175ac65e0caac9f17896f2ce_u128},
        {Sign::POS, -134, 0xb39318331fe564921c65a3c7f828972b_u128},
        {Sign::POS, -134, 0xb50740a0188d4daaabdc66446a4286d9_u128},
        {Sign::POS, -134, 0xb67b6ee9308ea27b2f3bbe8e8d72abec_u128},
        {Sign::POS, -134, 0xb7efa30e9727bf11b67dbdd7f03d168c_u128},
    },
    // -log2(r) for the third step, generated by SageMath with:
    //
    // for i in range(-80, 81):
    //   r = 2^-21 * round( 2^21 / (1 + i*2^(-21)) );
    //   s, m, e = RealField(128)(r).log2().sign_mantissa_exponent();
    //   print("{Sign::NEG," if (s == 1) else "{Sign::POS,", e, ",
    //         hex(m), "_u128},");
    /* .step_3 = */
    {
        {Sign::NEG, -142, 0xe6d3a96b978fc16e26f2c63c0827ccbb_u128},
        {Sign::NEG, -142, 0xe3f107a9fbfc50ca4b56fe667c8ec091_u128},
        {Sign::NEG, -142, 0xe10e65d14b937265647d76181aec10fc_u128},
        {Sign::NEG, -142, 0xde2bc3e18653b4f599e8f4d5379eca79_u128},
        {Sign::NEG, -142, 0xdb4921daac3ba730f07da89990c20623_u128},
        {Sign::NEG, -142, 0xd8667fbcbd49d7cd4a8121848531851a_u128},
        {Sign::NEG, -142, 0xd583dd87b97cd580679a4d854ae13619_u128},
        {Sign::NEG, -142, 0xd2a13b3ba0d32effe4d174072487a514_u128},
        {Sign::NEG, -142, 0xcfbe98d8734b73013c90319d969b54be_u128},
        {Sign::NEG, -142, 0xccdbf65e30e43039c6a173b09ba301e6_u128},
        {Sign::NEG, -142, 0xc9f953ccd99bf55eb8317428d7d8d06b_u128},
        {Sign::NEG, -142, 0xc716b1246d71512523cdb51bcc2061cd_u128},
        {Sign::NEG, -142, 0xc4340e64ec62d241f964fc78084fd515_u128},
        {Sign::NEG, -142, 0xc1516b8e566f076a06474fb15ccbb015_u128},
        {Sign::NEG, -142, 0xbe6ec8a0ab947f51f525ef6d0b75b1c3_u128},
        {Sign::NEG, -142, 0xbb8c259bebd1c8ae4e13532df7ee8da7_u128},
        {Sign::NEG, -142, 0xb8a982801725723376832500d72a9027_u128},
        {Sign::NEG, -142, 0xb5c6df4d2d8e0a95b14a3d285e592ba0_u128},
        {Sign::NEG, -142, 0xb2e43c032f0a20891e9e9dc9711f6e20_u128},
        {Sign::NEG, -142, 0xb00198a21b9842c1bc176e974f255fac_u128},
        {Sign::NEG, -142, 0xad1ef529f336fff364acf87fc0f648e6_u128},
        {Sign::NEG, -142, 0xaa3c519ab5e4e6d1d0b8a1574433e1f8_u128},
        {Sign::NEG, -142, 0xa759adf463a0861095f4e785371c69a9_u128},
        {Sign::NEG, -142, 0xa4770a36fc686c63277d5db00363a46f_u128},
        {Sign::NEG, -142, 0xa1946662803b287cd5cea669485ec36c_u128},
        {Sign::NEG, -142, 0x9eb1c276ef174910cec66fda04833322_u128},
        {Sign::NEG, -142, 0x9bcf1e7448fb5cd21da36f6ebe3851db_u128},
        {Sign::NEG, -142, 0x98ec7a5a8de5f273ab055d83abfc0d82_u128},
        {Sign::NEG, -142, 0x9609d629bdd598a83cecf110dbda68e9_u128},
        {Sign::NEG, -142, 0x932731e1d8c8de2276bbdb565a37e84b_u128},
        {Sign::NEG, -142, 0x90448d82debe5194d934c38857eee4f3_u128},
        {Sign::NEG, -142, 0x8d61e90ccfb481b1c27b427b4fbfc7db_u128},
        {Sign::NEG, -142, 0x8a7f447faba9fd2b6e13de502b142b39_u128},
        {Sign::NEG, -142, 0x879c9fdb729d52b3f4e406206614e2ba_u128},
        {Sign::NEG, -142, 0x84b9fb20248d10fd4d320daa3312ea6c_u128},
        {Sign::NEG, -142, 0x81d7564dc177c6b94aa528fc9d433c1a_u128},
        {Sign::NEG, -143, 0xfde962c892b805333c8ad047559b1622_u128},
        {Sign::NEG, -143, 0xf82418c77870a69facf765a8fc5bcc31_u128},
        {Sign::NEG, -143, 0xf25ece9834168f1abe238832edd27f20_u128},
        {Sign::NEG, -143, 0xec99843ac5a6dc0702644bfca329b708_u128},
        {Sign::NEG, -143, 0xe6d439af2d1eaac6c6d05a788e614744_u128},
        {Sign::NEG, -143, 0xe10eeef56a7b18bc133fe9cc57a8c1d0_u128},
        {Sign::NEG, -143, 0xdb49a40d7db94348aa4cb429195fb5dd_u128},
        {Sign::NEG, -143, 0xd58458f766d647ce0951ef239abbb959_u128},
        {Sign::NEG, -143, 0xcfbf0db325cf43ad686c430c89143d35_u128},
        {Sign::NEG, -143, 0xc9f9c240baa15447ba79c248afd42c12_u128},
        {Sign::NEG, -143, 0xc43476a0254996fdad19e0a92f115327_u128},
        {Sign::NEG, -143, 0xbe6f2ad165c5292fa8ad6ac3b0c99520_u128},
        {Sign::NEG, -143, 0xb8a9ded47c11283dd0567d4a9cc5e6a1_u128},
        {Sign::NEG, -143, 0xb2e492a9682ab18801f87c654b231443_u128},
        {Sign::NEG, -143, 0xad1f46502a0ee26dd6380b08358051bc_u128},
        {Sign::NEG, -143, 0xa759f9c8c1bad84ea07b024d26d391f6_u128},
        {Sign::NEG, -143, 0xa194ad132f2bb0896ee868cb69e3a7d8_u128},
        {Sign::NEG, -143, 0x9bcf602f725e887d0a6869eff6682f73_u128},
        {Sign::NEG, -143, 0x960a131d8b507d87f6a44d559ccf3f61_u128},
        {Sign::NEG, -143, 0x9044c5dd79fead0872066e1d30a8e210_u128},
        {Sign::NEG, -143, 0x8a7f786f3e66345c75ba3245b1b856af_u128},
        {Sign::NEG, -143, 0x84ba2ad2d88430e1b5ac020473ab198f_u128},
        {Sign::NEG, -144, 0xfde9ba1090ab7feb41127e3a88eb6741_u128},
        {Sign::NEG, -144, 0xf25f1e1f1baffdeabf80787522aca1c4_u128},
        {Sign::NEG, -144, 0xe6d481d15210167baf00688b14fa3adc_u128},
        {Sign::NEG, -144, 0xdb49e52733c604574d72837c8ab4d1e5_u128},
        {Sign::NEG, -144, 0xcfbf4820c0cc02364e38ac27bb252090_u128},
        {Sign::NEG, -144, 0xc434aabdf91c4ad0da3661f9292f59e8_u128},
        {Sign::NEG, -144, 0xb8aa0cfedcb118de8fd0af9bdfd21488_u128},
        {Sign::NEG, -144, 0xad1f6ee36b84a71682ee19a9abf0bfa5_u128},
        {Sign::NEG, -144, 0xa194d06ba591302f3cf68d5b5369a251_u128},
        {Sign::NEG, -144, 0x960a31978ad0eedebcd34f38c977647e_u128},
        {Sign::NEG, -144, 0x8a7f92671b3e1dda76eee9c9605e2143_u128},
        {Sign::NEG, -145, 0xfde9e5b4ada5efaeaa6a3887f0c803ab_u128},
        {Sign::NEG, -145, 0xe6d4a5e27b136f136e25927e582ac191_u128},
        {Sign::NEG, -145, 0xcfbf65579eb92f4ae2ebcac2f3a8e9eb_u128},
        {Sign::NEG, -145, 0xb8aa2414188ba5bb9d9acc22d5690751_u128},
        {Sign::NEG, -145, 0xa194e217e87f47cb1e12604b6d4132ef_u128},
        {Sign::NEG, -145, 0x8a7f9f630e888addcf340d2acb9b92a9_u128},
        {Sign::NEG, -146, 0xe6d4b7eb1537c8ae0dc5e49fbde3c520_u128},
        {Sign::NEG, -146, 0xb8aa2f9eb95b93320c074c9557c01188_u128},
        {Sign::NEG, -146, 0x8a7fa5e109656009f0f82818ff9b654f_u128},
        {Sign::NEG, -147, 0xb8aa35640a7c33ebd4cd612078bbe9b0_u128},
        {Sign::NEG, -148, 0xb8aa3846b33aaecff08cf68f42e09fa0_u128},
        {Sign::POS, 0, 0_u128},
        {Sign::POS, -148, 0xb8aa3e0c0513f9b168bd0facdf0ddaaf_u128},
        {Sign::POS, -147, 0xb8aa40eeae2ec9b3192af653dd41575b_u128},
        {Sign::POS, -146, 0x8a7fb2dd018e48923b5c89842e540a51_u128},
        {Sign::POS, -146, 0xb8aa46b400c0bee334ad8ebdd8b2750c_u128},
        {Sign::POS, -146, 0xe6d4dbfc54c5dd1b70b12bd698e5be74_u128},
        {Sign::POS, -145, 0x8a7fb95afeda5c4608c7e424efbd90e1_u128},
        {Sign::POS, -145, 0xa19505707dd2334431b8eba774a1de77_u128},
        {Sign::POS, -145, 0xb8aa523ea755fe32ee400e8c68838733_u128},
        {Sign::POS, -145, 0xcfbf9fc57b7147be0e71fa0b5603bc2f_u128},
        {Sign::POS, -145, 0xe6d4ee04fa2f9a927763c919d8ac65f1_u128},
        {Sign::POS, -145, 0xfdea3cfd239c815e232b270bb6046ec1_u128},
        {Sign::POS, -144, 0x8a7fc656fbe1c368106f39197e068972_u128},
        {Sign::POS, -144, 0x960a6e8bbb581acc4a4a6f4012941bd9_u128},
        {Sign::POS, -144, 0xa195171cd0370c345bb34c1120b3e54b_u128},
        {Sign::POS, -144, 0xad1fc00a3a845cf96bb6731392a3147a_u128},
        {Sign::POS, -144, 0xb8aa6953fa45d2752be1268dcee3c8fc_u128},
        {Sign::POS, -144, 0xc43512fa0f813201d84158d5d50251a9_u128},
        {Sign::POS, -144, 0xcfbfbcfc7a3c40fa3765bda15d0ef0fa_u128},
        {Sign::POS, -144, 0xdb4a675b3a7cc4b99a5ddb55f9cc27d9_u128},
        {Sign::POS, -144, 0xe6d512165048829bdcba1c593d918775_u128},
        {Sign::POS, -144, 0xf25fbd2dbba53ffd648be060e1e30a95_u128},
        {Sign::POS, -144, 0xfdea68a17c98c23b22658dc2f1bcf6e8_u128},
        {Sign::POS, -143, 0x84ba8a38c994675948ad5162fb4a236e_u128},
        {Sign::POS, -143, 0x8a7fe04effad9560db7fe3789405ce3a_u128},
        {Sign::POS, -143, 0x90453693609acde391b56e2e4f2e5ed8_u128},
        {Sign::POS, -143, 0x960a8d05ec5ef390f8998880c3bb4d76_u128},
        {Sign::POS, -143, 0x9bcfe3a6a2fce918e2b878052f67efee_u128},
        {Sign::POS, -143, 0xa1953a758477912b67df399193f707c0_u128},
        {Sign::POS, -143, 0xa75a917290d1ce78e51b89e4d5d095e1_u128},
        {Sign::POS, -143, 0xad1fe89dc80e83b1fcbbee4edbf9f47d_u128},
        {Sign::POS, -143, 0xb2e53ff72a309387964fbd58b168371b_u128},
        {Sign::POS, -143, 0xb8aa977eb73ae0aadea7276ca7acd135_u128},
        {Sign::POS, -143, 0xbe6fef346f304dcd47d33f7e7afc83a6_u128},
        {Sign::POS, -143, 0xc43547185213bda0892603b377909123_u128},
        {Sign::POS, -143, 0xc9fa9f2a5fe812d69f32660aa06239fb_u128},
        {Sign::POS, -143, 0xcfbff76a98b03021cbcc5504d7407f6c_u128},
        {Sign::POS, -143, 0xd5854fd8fc6ef8349608c44d06402ebe_u128},
        {Sign::POS, -143, 0xdb4aa8758b274dc1ca3db5604a863477_u128},
        {Sign::POS, -143, 0xe110014044dc137c7a024036206c37d6_u128},
        {Sign::POS, -143, 0xe6d55a3929902c17fc2e9be890ff7ee3_u128},
        {Sign::POS, -143, 0xec9ab36039467a47ecdc275c60da1b53_u128},
        {Sign::POS, -143, 0xf2600cb57401e0c02d6571e94056607f_u128},
        {Sign::POS, -143, 0xf8256638d9c54234e4664401fd1ca2a7_u128},
        {Sign::POS, -143, 0xfdeabfea6a93815a7dbba7dcb50b3fd7_u128},
        {Sign::POS, -142, 0x81d80ce51337c072d541f90d853c794b_u128},
        {Sign::POS, -142, 0x84bab9ec06ae11c5b08f65392ce8b75b_u128},
        {Sign::POS, -142, 0x879d670a0fae26006e969a29f8462436_u128},
        {Sign::POS, -142, 0x8a80143f2e396e7dcfc8cbcaa2bf130c_u128},
        {Sign::POS, -142, 0x8d62c18b62515c98b737e48c19421e68_u128},
        {Sign::POS, -142, 0x90456eeeabf761ac2a9689b997c50c0b_u128},
        {Sign::POS, -142, 0x93281c690b2cef1352381fccc774d66b_u128},
        {Sign::POS, -142, 0x960ac9fa7ff376297910cec1dd92dc10_u128},
        {Sign::POS, -142, 0x98ed77a30a4c684a0cb5866bbaff34cb_u128},
        {Sign::POS, -142, 0x9bd02562aa3936d09d5c02c80c702d11_u128},
        {Sign::POS, -142, 0x9eb2d3395fbb5318dddad0536b56e775_u128},
        {Sign::POS, -142, 0xa19581272ad42e7ea3a9505d7f71247a_u128},
        {Sign::POS, -142, 0xa4782f2c0b853a5de6dfbd5d210830d7_u128},
        {Sign::POS, -142, 0xa75add4801cfe812c2372f447bdcfa45_u128},
        {Sign::POS, -142, 0xaa3d8b7b0db5a8f973099fd532c14b05_u128},
        {Sign::POS, -142, 0xad2039c52f37ee6e5951eef483de2c37_u128},
        {Sign::POS, -142, 0xb002e826665829cdf7abe6ff6da76f1e_u128},
        {Sign::POS, -142, 0xb2e5969eb317cc74f354411ed47c5d7b_u128},
        {Sign::POS, -142, 0xb5c8452e157847c01428a99ba8f5911f_u128},
        {Sign::POS, -142, 0xb8aaf3d48d7b0d0c44a7c4330edff2c8_u128},
        {Sign::POS, -142, 0xbb8da2921b218db691f1306a84e4e07b_u128},
        {Sign::POS, -142, 0xbe705166be6d3b1c2bc58de40cdf7b6a_u128},
        {Sign::POS, -142, 0xc1530052775f869a648680b254df1d99_u128},
        {Sign::POS, -142, 0xc435af5545f9e18eb136b5ace0d6f74d_u128},
        {Sign::POS, -142, 0xc7185e6f2a3dbd56a979e6c434fad480_u128},
        {Sign::POS, -142, 0xc9fb0da0242c8b500794df5600c90a5a_u128},
        {Sign::POS, -142, 0xccddbce833c7bcd8a86d80814ac18cf1_u128},
        {Sign::POS, -142, 0xcfc06c475910c34e8b8ac57a9cca2d56_u128},
        {Sign::POS, -142, 0xd2a31bbd9409100fd314c7e03140001f_u128},
        {Sign::POS, -142, 0xd585cb4ae4b2147ac3d4c40e20b5ec89_u128},
        {Sign::POS, -142, 0xd8687aef4b0d41edc5351d729060644e_u128},
        {Sign::POS, -142, 0xdb4b2aaac71c09c7614162e1e12e445d_u128},
        {Sign::POS, -142, 0xde2dda7d58dfdd6644a652eadf8ede85_u128},
        {Sign::POS, -142, 0xe1108a67005a2e293eb1e02af3e52c3c_u128},
        {Sign::POS, -142, 0xe3f33a67bd8c6d6f415335a253a82aa2_u128},
        {Sign::POS, -142, 0xe6d5ea7f90780c97611abb0833305fe1_u128},
    },
    // -log2(r) for the fourth step, generated by SageMath with:
    //
    // for i in range(-65, 65):
    //   r = 2^-28 * round( 2^28 / (1 + i*2^(-28)) );
    //   s, m, e = RealField(128)(r).log2().sign_mantissa_exponent();
    //   print("{Sign::NEG," if (s == 1) else "{Sign::POS,", e, ",
    //         hex(m), "_u128},");
    /* .step_4 = */
    {
        {Sign::NEG, -149, 0xbb8ce2990b5d0b90ef1bffe565ce0a46_u128},
        {Sign::NEG, -149, 0xb8aa39b807a576e4bea3244560ca3d99_u128},
        {Sign::NEG, -149, 0xb5c790d6d5c354df8b91f71ceefa31a2_u128},
        {Sign::NEG, -149, 0xb2e4e7f575b6a57b9096e3d684001c0e_u128},
        {Sign::NEG, -149, 0xb0023f13e77f68b3086054c794367f36_u128},
        {Sign::NEG, -149, 0xad1f96322b1d9e802d9cb33094afe4de_u128},
        {Sign::NEG, -149, 0xaa3ced50409146dd3afa673cfb3698f3_u128},
        {Sign::NEG, -149, 0xa75a446e27da61c46b27d8033e4c6450_u128},
        {Sign::NEG, -149, 0xa4779b8be0f8ef2ff8d36b84d52a477b_u128},
        {Sign::NEG, -149, 0xa194f2a96becef1a1eab86ae37c03565_u128},
        {Sign::NEG, -149, 0x9eb249c6c8b6617d175e8d56deb4ce2c_u128},
        {Sign::NEG, -149, 0x9bcfa0e3f75546531d9ae241436519da_u128},
        {Sign::NEG, -149, 0x98ecf800f7c99d966c0ee71adfe44325_u128},
        {Sign::NEG, -149, 0x960a4f1dca1367413d68fc7c2efb522f_u128},
        {Sign::NEG, -149, 0x9327a63a6e32a34dcc5781e8ac28e749_u128},
        {Sign::NEG, -149, 0x9044fd56e42751b65388d5ced3a0f5af_u128},
        {Sign::NEG, -149, 0x8d6254732bf172750dab5588224c7e4a_u128},
        {Sign::NEG, -149, 0x8a7fab8f45910584356d5d5915c94a70_u128},
        {Sign::NEG, -149, 0x879d02ab31060ade057d48712c69a6a7_u128},
        {Sign::NEG, -149, 0x84ba59c6ee50827cb88970eae5341d60_u128},
        {Sign::NEG, -149, 0x81d7b0e27d706c5a89402fcbbfe331bb_u128},
        {Sign::NEG, -150, 0xfdea0ffbbccb90e3649fba0879ca348b_u128},
        {Sign::NEG, -150, 0xf824be3222612d78dccd9edfbab6f777_u128},
        {Sign::NEG, -150, 0xf25f6c682ba1ae69f066b9aa4636478e_u128},
        {Sign::NEG, -150, 0xec9a1a9dd88d13ab14c7b3cb21578781_u128},
        {Sign::NEG, -150, 0xe6d4c8d329235d30bf4d347b528f56e1_u128},
        {Sign::NEG, -150, 0xe10f77081d648aef6553e0c9e1b70799_u128},
        {Sign::NEG, -150, 0xdb4a253cb5509cdb7c385b9bd80c1375_u128},
        {Sign::NEG, -150, 0xd584d370f0e792e9795745ac402f919d_u128},
        {Sign::NEG, -150, 0xcfbf81a4d0296d0dd20d3d8c2625ac1b_u128},
        {Sign::NEG, -150, 0xc9fa2fd853162b3cfbb6dfa297551554_u128},
        {Sign::NEG, -150, 0xc434de0b79adcd6b6bb0c62ca2867d91_u128},
        {Sign::NEG, -150, 0xbe6f8c3e43f0538d9757893d57e40877_u128},
        {Sign::NEG, -150, 0xb8aa3a70b1ddbd97f407bebdc8f8c28e_u128},
        {Sign::NEG, -150, 0xb2e4e8a2c3760b7ef71dfa6d08b016be_u128},
        {Sign::NEG, -150, 0xad1f96d478b93d3715f6cde02b5543ce_u128},
        {Sign::NEG, -150, 0xa75a4505d1a752b4c5eec8824692d1e9_u128},
        {Sign::NEG, -150, 0xa194f336ce404bec7c6277947172081a_u128},
        {Sign::NEG, -150, 0x9bcfa1676e8428d2aeae662dc45a61ce_u128},
        {Sign::NEG, -150, 0x960a4f97b272e95bd22f1d3b59110455_u128},
        {Sign::NEG, -150, 0x9044fdc79a0c8d7c5c4123804ab83462_u128},
        {Sign::NEG, -150, 0x8a7fabf725511528c240fd95b5cecb89_u128},
        {Sign::NEG, -150, 0x84ba5a2654408055798b2deab82fadc4_u128},
        {Sign::NEG, -151, 0xfdea10aa4db59dedeef86988e2227ddb_u128},
        {Sign::NEG, -151, 0xf25f6d073a40020362e1207c0209b090_u128},
        {Sign::NEG, -151, 0xe6d4c9636e202cd43989789113ec7bee_u128},
        {Sign::NEG, -151, 0xdb4a25bee9561e495daa65565e562909_u128},
        {Sign::NEG, -151, 0xcfbf8219abe1d64bb9fcd6062a84acbd_u128},
        {Sign::NEG, -151, 0xc434de73b5c354c43939b586c46792b3_u128},
        {Sign::NEG, -151, 0xb8aa3acd06fa999bc619ea6a7a9ee85e_u128},
        {Sign::NEG, -151, 0xad1f97259f87a4bb4b5656ef9e7a27fd_u128},
        {Sign::NEG, -151, 0xa194f37d7f6a760bb3a7d90083f7239c_u128},
        {Sign::NEG, -151, 0x960a4fd4a6a30d75e9c74a3381c0f016_u128},
        {Sign::NEG, -151, 0x8a7fac2b15316ae2d86d7fcaf12ed012_u128},
        {Sign::NEG, -152, 0xfdea1101962b1c76d4a6956a5c863e0f_u128},
        {Sign::NEG, -152, 0xe6d4c9ab909eeed11462ef192f547877_u128},
        {Sign::NEG, -152, 0xcfbf825419be4ca645819d2f1d72eb8b_u128},
        {Sign::NEG, -152, 0xb8aa3afb318935c83d742790eedbe719_u128},
        {Sign::NEG, -152, 0xa194f3a0d7ffaa08d1ac0d7b70d74492_u128},
        {Sign::NEG, -152, 0x8a7fac450d21a939d79ac58375f83d0c_u128},
        {Sign::NEG, -153, 0xe6d4c9cfa1de665a49637b2bac367e87_u128},
        {Sign::NEG, -153, 0xb8aa3b1246d08f691cc4b5eedcc78b35_u128},
        {Sign::NEG, -153, 0x8a7fac520919cd43d43bf48a42745836_u128},
        {Sign::NEG, -154, 0xb8aa3b1dd1743f1c3557bdcf592619eb_u128},
        {Sign::NEG, -155, 0xb8aa3b2396c617ae6bdc2e83d3ebb0c4_u128},
        {Sign::POS, 0, 0_u128},
        {Sign::POS, -155, 0xb8aa3b2f2169ca442d5b40050e44e8ab_u128},
        {Sign::POS, -154, 0xb8aa3b34e6bba447b8560371b8f04afe_u128},
        {Sign::POS, -153, 0x8a7fac6c010a1f14c79a43ccc70459cc_u128},
        {Sign::POS, -153, 0xb8aa3b40715f59c022c25632f519f77f_u128},
        {Sign::POS, -153, 0xe6d4ca17c45d828242c10a314e35fb9e_u128},
        {Sign::POS, -152, 0x8a7fac78fd024cdbbe5a212ed7b949e4_u128},
        {Sign::POS, -152, 0xa194f3e7892a4fde12dcf94ef5c5b918_u128},
        {Sign::POS, -152, 0xb8aa3b5786a6ca7649781013e57110ce_u128},
        {Sign::POS, -152, 0xcfbf82c8f577bcd28cba70c085c12cb3_u128},
        {Sign::POS, -152, 0xe6d4ca3bd59d272107332f3fb09328b8_u128},
        {Sign::POS, -152, 0xfdea11b02717098fe37168243a9d8b14_u128},
        {Sign::POS, -151, 0x8a7fac92f4f2b226a602205479b93722_u128},
        {Sign::POS, -151, 0x960a504e8f041bc3b5bd735852c0d583_u128},
        {Sign::POS, -151, 0xa194f40ae1bfc1b6363248630b0d812d_u128},
        {Sign::POS, -151, 0xad1f97c7ed25a4153ca83f0e02b823c0_u128},
        {Sign::POS, -151, 0xb8aa3b85b135c2f7de66fb46974bc4fd_u128},
        {Sign::POS, -151, 0xc434df442df01e7530b6254e23c69fc2_u128},
        {Sign::POS, -151, 0xcfbf83036354b6a448dd69ba009b370c_u128},
        {Sign::POS, -151, 0xdb4a26c351638b9c3c24797383b16af5_u128},
        {Sign::POS, -151, 0xe6d4ca83f81c9d741fd309b800678db7_u128},
        {Sign::POS, -151, 0xf25f6e45577fec430930d418c79378a3_u128},
        {Sign::POS, -151, 0xfdea12076f8d78200d85967b2783a12c_u128},
        {Sign::POS, -150, 0x84ba5ae52022a091210c898c360016ed_u128},
        {Sign::POS, -150, 0x8a7facc6e4d3a3b05e19883eef2605ab_u128},
        {Sign::POS, -150, 0x9044fea905d9c579488dacc6629300ae_u128},
        {Sign::POS, -150, 0x960a508b833505f76b0cdebd3264e3e3_u128},
        {Sign::POS, -150, 0x9bcfa26e5ce56536503b07e7ff788dc2_u128},
        {Sign::POS, -150, 0xa194f45192eae34182bc1435696a69d1_u128},
        {Sign::POS, -150, 0xa75a4635254580248d33f1be0e96fb1f_u128},
        {Sign::POS, -150, 0xad1f981913f53beafa4690c48c1b66c9_u128},
        {Sign::POS, -150, 0xb2e4e9fd5efa16a05497e3b57dd5fe75_u128},
        {Sign::POS, -150, 0xb8aa3be20654105026cbdf277e66cad5_u128},
        {Sign::POS, -150, 0xbe6f8dc70a032905fb8679db27301625_u128},
        {Sign::POS, -150, 0xc434dfac6a0760cd5d6bacbb1056f6aa_u128},
        {Sign::POS, -150, 0xc9fa31922660b7b1d71f72dbd0c3d936_u128},
        {Sign::POS, -150, 0xcfbf83783f0f2dbef345c97bfe230ba2_u128},
        {Sign::POS, -150, 0xd584d55eb412c3003c82b0042ce54751_u128},
        {Sign::POS, -150, 0xdb4a2745856b77813d7a2806f0403bae_u128},
        {Sign::POS, -150, 0xe10f792cb3194b4d80d03540da2f18ae_u128},
        {Sign::POS, -150, 0xe6d4cb143d1c3e709128dd987b73194f_u128},
        {Sign::POS, -150, 0xec9a1cfc237450f5f928291e63940e14_u128},
        {Sign::POS, -150, 0xf25f6ee4662182e94372220d20e0e78a_u128},
        {Sign::POS, -150, 0xf824c0cd0523d455faaad4c9407040c7_u128},
        {Sign::POS, -150, 0xfdea12b6007b4547a9764fe14e20e9e4_u128},
        {Sign::POS, -149, 0x81d7b24fac13eae4ed3c5206ea4d3942_u128},
        {Sign::POS, -149, 0x84ba5b448614c2f40c2af218aea6da27_u128},
        {Sign::POS, -149, 0x879d04398e402ad6f6d912ac383aaeba_u128},
        {Sign::POS, -149, 0x8a7fad2ec49622937298bf5cca8b3d95_u128},
        {Sign::POS, -149, 0x8d6256242916aa2f44bc04daa8808214_u128},
        {Sign::POS, -149, 0x9044ff19bbc1c1b03294f0eb14683198_u128},
        {Sign::POS, -149, 0x9327a80f7c97691c017592684ff600c3_u128},
        {Sign::POS, -149, 0x960a51056b97a07876aff9419c43e8b9_u128},
        {Sign::POS, -149, 0x98ecf9fb88c267cb5796367b39d26c63_u128},
        {Sign::POS, -149, 0x9bcfa2f1d417bf1a697a5c2e6888ddaa_u128},
        {Sign::POS, -149, 0x9eb24be84d97a66b71ae7d8967b5a2b7_u128},
        {Sign::POS, -149, 0xa194f4def5421dc43584aecf760e7b39_u128},
        {Sign::POS, -149, 0xa4779dd5cb17252a7a4f0558d1b0c59e_u128},
        {Sign::POS, -149, 0xa75a46cccf16bca4055f9792b821c455_u128},
        {Sign::POS, -149, 0xaa3cefc40140e4369c087cff664ee311_u128},
        {Sign::POS, -149, 0xad1f98bb61959be8039bce36188dfc04_u128},
        {Sign::POS, -149, 0xb00241b2f014e3be016ba4e30a9d9d21_u128},
        {Sign::POS, -149, 0xb2e4eaaaacbebbbe5aca1bc777a54d5e_u128},
        {Sign::POS, -149, 0xb5c793a2979323eed5094eb99a35d1f0_u128},
        {Sign::POS, -149, 0xb8aa3c9ab0921c55357b5aa4ac49738d_u128},
    }};

// > P = fpminimax(log2(1 + x)/x, 3, [|128...|], [-0x1.0002143p-29 , 0x1p-29]);
// > P;
// > dirtyinfnorm(log2(1 + x)/x - P, [-0x1.0002143p-29 , 0x1p-29]);
// 0x1.27ad5...p-121
const Float128 BIG_COEFFS[4]{
    {Sign::NEG, -129, 0xb8aa3b295c2b21e33eccf6940d66bbcc_u128},
    {Sign::POS, -129, 0xf6384ee1d01febc9ee39a6d649394bb1_u128},
    {Sign::NEG, -128, 0xb8aa3b295c17f0bbbe87fed067ea2ad5_u128},
    {Sign::POS, -127, 0xb8aa3b295c17f0bbbe87fed0691d3e3f_u128},
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
      return FPBits_t::quiet_nan().get_val();
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
