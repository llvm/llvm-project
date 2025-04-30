/* ============================================================
Copyright (c) 2002-2015 Advanced Micro Devices, Inc.

All rights reserved.

Redistribution and  use in source and binary  forms, with or
without  modification,  are   permitted  provided  that  the
following conditions are met:

+ Redistributions  of source  code  must  retain  the  above
  copyright  notice,   this  list  of   conditions  and  the
  following disclaimer.

+ Redistributions  in binary  form must reproduce  the above
  copyright  notice,   this  list  of   conditions  and  the
  following  disclaimer in  the  documentation and/or  other
  materials provided with the distribution.

+ Neither the  name of Advanced Micro Devices,  Inc. nor the
  names  of  its contributors  may  be  used  to endorse  or
  promote  products  derived   from  this  software  without
  specific prior written permission.

THIS  SOFTWARE  IS PROVIDED  BY  THE  COPYRIGHT HOLDERS  AND
CONTRIBUTORS "AS IS" AND  ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING,  BUT NOT  LIMITED TO,  THE IMPLIED  WARRANTIES OF
MERCHANTABILITY  AND FITNESS  FOR A  PARTICULAR  PURPOSE ARE
DISCLAIMED.  IN  NO  EVENT  SHALL  ADVANCED  MICRO  DEVICES,
INC.  OR CONTRIBUTORS  BE LIABLE  FOR ANY  DIRECT, INDIRECT,
INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL  DAMAGES
(INCLUDING,  BUT NOT LIMITED  TO, PROCUREMENT  OF SUBSTITUTE
GOODS  OR  SERVICES;  LOSS  OF  USE, DATA,  OR  PROFITS;  OR
BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON  ANY THEORY OF
LIABILITY,  WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
(INCLUDING NEGLIGENCE  OR OTHERWISE) ARISING IN  ANY WAY OUT
OF  THE  USE  OF  THIS  SOFTWARE, EVEN  IF  ADVISED  OF  THE
POSSIBILITY OF SUCH DAMAGE.

It is  licensee's responsibility  to comply with  any export
regulations applicable in licensee's jurisdiction.
============================================================ */

#include "libm_amd.h"
#include "libm_util_amd.h"

#define USE_VALF_WITH_FLAGS
#define USE_NAN_WITH_FLAGS
#define USE_SCALEDOUBLE_1
#define USE_SCALEDOWNDOUBLE
#define USE_HANDLE_ERRORF
#include "libm_inlines_amd.h"
#undef USE_VALF_WITH_FLAGS
#undef USE_NAN_WITH_FLAGS
#undef USE_SCALEDOUBLE_1
#undef USE_SCALEDOWNDOUBLE
#undef USE_HANDLE_ERRORF

#include "libm_errno_amd.h"

float FN_PROTOTYPE(mth_i_atan2)(float fy, float fx)
{
  /* Array atan_jby256 contains precomputed values of atan(j/256),
     for j = 16, 17, ..., 256. */

  static const double atan_jby256[241] = {
      6.24188099959573430842e-02,  /* 0x3faff55bb72cfde9 */
      6.63088949198234745008e-02,  /* 0x3fb0f99ea71d52a6 */
      7.01969710718705064423e-02,  /* 0x3fb1f86dbf082d58 */
      7.40829225490337306415e-02,  /* 0x3fb2f719318a4a9a */
      7.79666338315423007588e-02,  /* 0x3fb3f59f0e7c559d */
      8.18479898030765457007e-02,  /* 0x3fb4f3fd677292fb */
      8.57268757707448092464e-02,  /* 0x3fb5f2324fd2d7b2 */
      8.96031774848717321724e-02,  /* 0x3fb6f03bdcea4b0c */
      9.34767811585894559112e-02,  /* 0x3fb7ee182602f10e */
      9.73475734872236708739e-02,  /* 0x3fb8ebc54478fb28 */
      1.01215441667466668485e-01,  /* 0x3fb9e94153cfdcf1 */
      1.05080273416329528224e-01,  /* 0x3fbae68a71c722b8 */
      1.08941956989865793015e-01,  /* 0x3fbbe39ebe6f07c3 */
      1.12800381201659388752e-01,  /* 0x3fbce07c5c3cca32 */
      1.16655435441069349478e-01,  /* 0x3fbddd21701eba6e */
      1.20507009691224548087e-01,  /* 0x3fbed98c2190043a */
      1.24354994546761424279e-01,  /* 0x3fbfd5ba9aac2f6d */
      1.28199281231298117811e-01,  /* 0x3fc068d584212b3d */
      1.32039761614638734288e-01,  /* 0x3fc0e6adccf40881 */
      1.35876328229701304195e-01,  /* 0x3fc1646541060850 */
      1.39708874289163620386e-01,  /* 0x3fc1e1fafb043726 */
      1.43537293701821222491e-01,  /* 0x3fc25f6e171a535c */
      1.47361481088651630200e-01,  /* 0x3fc2dcbdb2fba1ff */
      1.51181331798580037562e-01,  /* 0x3fc359e8edeb99a3 */
      1.54996741923940972718e-01,  /* 0x3fc3d6eee8c6626c */
      1.58807608315631065832e-01,  /* 0x3fc453cec6092a9e */
      1.62613828597948567589e-01,  /* 0x3fc4d087a9da4f17 */
      1.66415301183114927586e-01,  /* 0x3fc54d18ba11570a */
      1.70211925285474380276e-01,  /* 0x3fc5c9811e3ec269 */
      1.74003600935367680469e-01,  /* 0x3fc645bfffb3aa73 */
      1.77790228992676047071e-01,  /* 0x3fc6c1d4898933d8 */
      1.81571711160032150945e-01,  /* 0x3fc73dbde8a7d201 */
      1.85347949995694760705e-01,  /* 0x3fc7b97b4bce5b02 */
      1.89118848926083965578e-01,  /* 0x3fc8350be398ebc7 */
      1.92884312257974643856e-01,  /* 0x3fc8b06ee2879c28 */
      1.96644245190344985064e-01,  /* 0x3fc92ba37d050271 */
      2.00398553825878511514e-01,  /* 0x3fc9a6a8e96c8626 */
      2.04147145182116990236e-01,  /* 0x3fca217e601081a5 */
      2.07889927202262986272e-01,  /* 0x3fca9c231b403279 */
      2.11626808765629753628e-01,  /* 0x3fcb1696574d780b */
      2.15357699697738047551e-01,  /* 0x3fcb90d7529260a2 */
      2.19082510780057748701e-01,  /* 0x3fcc0ae54d768466 */
      2.22801153759394493514e-01,  /* 0x3fcc84bf8a742e6d */
      2.26513541356919617664e-01,  /* 0x3fccfe654e1d5395 */
      2.30219587276843717927e-01,  /* 0x3fcd77d5df205736 */
      2.33919206214733416127e-01,  /* 0x3fcdf110864c9d9d */
      2.37612313865471241892e-01,  /* 0x3fce6a148e96ec4d */
      2.41298826930858800743e-01,  /* 0x3fcee2e1451d980c */
      2.44978663126864143473e-01,  /* 0x3fcf5b75f92c80dd */
      2.48651741190513253521e-01,  /* 0x3fcfd3d1fc40dbe4 */
      2.52317980886427151166e-01,  /* 0x3fd025fa510665b5 */
      2.55977303013005474952e-01,  /* 0x3fd061eea03d6290 */
      2.59629629408257511791e-01,  /* 0x3fd09dc597d86362 */
      2.63274882955282396590e-01,  /* 0x3fd0d97ee509acb3 */
      2.66912987587400396539e-01,  /* 0x3fd1151a362431c9 */
      2.70543868292936529052e-01,  /* 0x3fd150973a9ce546 */
      2.74167451119658789338e-01,  /* 0x3fd18bf5a30bf178 */
      2.77783663178873208022e-01,  /* 0x3fd1c735212dd883 */
      2.81392432649178403370e-01,  /* 0x3fd2025567e47c95 */
      2.84993688779881237938e-01,  /* 0x3fd23d562b381041 */
      2.88587361894077354396e-01,  /* 0x3fd278372057ef45 */
      2.92173383391398755471e-01,  /* 0x3fd2b2f7fd9b5fe2 */
      2.95751685750431536626e-01,  /* 0x3fd2ed987a823cfe */
      2.99322202530807379706e-01,  /* 0x3fd328184fb58951 */
      3.02884868374971361060e-01,  /* 0x3fd362773707ebcb */
      3.06439619009630070945e-01,  /* 0x3fd39cb4eb76157b */
      3.09986391246883430384e-01,  /* 0x3fd3d6d129271134 */
      3.13525122985043869228e-01,  /* 0x3fd410cbad6c7d32 */
      3.17055753209146973237e-01,  /* 0x3fd44aa436c2af09 */
      3.20578221991156986359e-01,  /* 0x3fd4845a84d0c21b */
      3.24092470489871664618e-01,  /* 0x3fd4bdee586890e6 */
      3.27598440950530811477e-01,  /* 0x3fd4f75f73869978 */
      3.31096076704132047386e-01,  /* 0x3fd530ad9951cd49 */
      3.34585322166458920545e-01,  /* 0x3fd569d88e1b4cd7 */
      3.38066122836825466713e-01,  /* 0x3fd5a2e0175e0f4e */
      3.41538425296541714449e-01,  /* 0x3fd5dbc3fbbe768d */
      3.45002177207105076295e-01,  /* 0x3fd614840309cfe1 */
      3.48457327308122011278e-01,  /* 0x3fd64d1ff635c1c5 */
      3.51903825414964732676e-01,  /* 0x3fd685979f5fa6fd */
      3.55341622416168290144e-01,  /* 0x3fd6bdeac9cbd76c */
      3.58770670270572189509e-01,  /* 0x3fd6f61941e4def0 */
      3.62190922004212156882e-01,  /* 0x3fd72e22d53aa2a9 */
      3.65602331706966821034e-01,  /* 0x3fd7660752817501 */
      3.69004854528964421068e-01,  /* 0x3fd79dc6899118d1 */
      3.72398446676754202311e-01,  /* 0x3fd7d5604b63b3f7 */
      3.75783065409248884237e-01,  /* 0x3fd80cd46a14b1d0 */
      3.79158669033441808605e-01,  /* 0x3fd84422b8df95d7 */
      3.82525216899905096124e-01,  /* 0x3fd87b4b0c1ebedb */
      3.85882669398073752109e-01,  /* 0x3fd8b24d394a1b25 */
      3.89230987951320717144e-01,  /* 0x3fd8e92916f5cde8 */
      3.92570135011828580396e-01,  /* 0x3fd91fde7cd0c662 */
      3.95900074055262896078e-01,  /* 0x3fd9566d43a34907 */
      3.99220769575252543149e-01,  /* 0x3fd98cd5454d6b18 */
      4.02532187077682512832e-01,  /* 0x3fd9c3165cc58107 */
      4.05834293074804064450e-01,  /* 0x3fd9f93066168001 */
      4.09127055079168300278e-01,  /* 0x3fda2f233e5e530b */
      4.12410441597387267265e-01,  /* 0x3fda64eec3cc23fc */
      4.15684422123729413467e-01,  /* 0x3fda9a92d59e98cf */
      4.18948967133552840902e-01,  /* 0x3fdad00f5422058b */
      4.22204048076583571270e-01,  /* 0x3fdb056420ae9343 */
      4.25449637370042266227e-01,  /* 0x3fdb3a911da65c6c */
      4.28685708391625730496e-01,  /* 0x3fdb6f962e737efb */
      4.31912235472348193799e-01,  /* 0x3fdba473378624a5 */
      4.35129193889246812521e-01,  /* 0x3fdbd9281e528191 */
      4.38336559857957774877e-01,  /* 0x3fdc0db4c94ec9ef */
      4.41534310525166673322e-01,  /* 0x3fdc42191ff11eb6 */
      4.44722423960939305942e-01,  /* 0x3fdc76550aad71f8 */
      4.47900879150937292206e-01,  /* 0x3fdcaa6872f3631b */
      4.51069655988523443568e-01,  /* 0x3fdcde53432c1350 */
      4.54228735266762495559e-01,  /* 0x3fdd121566b7f2ad */
      4.57378098670320809571e-01,  /* 0x3fdd45aec9ec862b */
      4.60517728767271039558e-01,  /* 0x3fdd791f5a1226f4 */
      4.63647609000806093515e-01,  /* 0x3fddac670561bb4f */
      4.66767723680866497560e-01,  /* 0x3fdddf85bb026974 */
      4.69878057975686880265e-01,  /* 0x3fde127b6b0744af */
      4.72978597903265574054e-01,  /* 0x3fde4548066cf51a */
      4.76069330322761219421e-01,  /* 0x3fde77eb7f175a34 */
      4.79150242925822533735e-01,  /* 0x3fdeaa65c7cf28c4 */
      4.82221324227853687105e-01,  /* 0x3fdedcb6d43f8434 */
      4.85282563559221225002e-01,  /* 0x3fdf0ede98f393cf */
      4.88333951056405479729e-01,  /* 0x3fdf40dd0b541417 */
      4.91375477653101910835e-01,  /* 0x3fdf72b221a4e495 */
      4.94407135071275316562e-01,  /* 0x3fdfa45dd3029258 */
      4.97428915812172245392e-01,  /* 0x3fdfd5e0175fdf83 */
      5.00440813147294050189e-01,  /* 0x3fe0039c73c1a40b */
      5.03442821109336358099e-01,  /* 0x3fe01c341e82422d */
      5.06434934483096732549e-01,  /* 0x3fe034b709250488 */
      5.09417148796356245022e-01,  /* 0x3fe04d25314342e5 */
      5.12389460310737621107e-01,  /* 0x3fe0657e94db30cf */
      5.15351866012543347040e-01,  /* 0x3fe07dc3324e9b38 */
      5.18304363603577900044e-01,  /* 0x3fe095f30861a58f */
      5.21246951491958210312e-01,  /* 0x3fe0ae0e1639866c */
      5.24179628782913242802e-01,  /* 0x3fe0c6145b5b43da */
      5.27102395269579471204e-01,  /* 0x3fe0de05d7aa6f7c */
      5.30015251423793132268e-01,  /* 0x3fe0f5e28b67e295 */
      5.32918198386882147055e-01,  /* 0x3fe10daa77307a0d */
      5.35811237960463593311e-01,  /* 0x3fe1255d9bfbd2a8 */
      5.38694372597246617929e-01,  /* 0x3fe13cfbfb1b056e */
      5.41567605391844897333e-01,  /* 0x3fe1548596376469 */
      5.44430940071603086672e-01,  /* 0x3fe16bfa6f5137e1 */
      5.47284380987436924748e-01,  /* 0x3fe1835a88be7c13 */
      5.50127933104692989907e-01,  /* 0x3fe19aa5e5299f99 */
      5.52961601994028217888e-01,  /* 0x3fe1b1dc87904284 */
      5.55785393822313511514e-01,  /* 0x3fe1c8fe7341f64f */
      5.58599315343562330405e-01,  /* 0x3fe1e00babdefeb3 */
      5.61403373889889367732e-01,  /* 0x3fe1f7043557138a */
      5.64197577362497537656e-01,  /* 0x3fe20de813e823b1 */
      5.66981934222700489912e-01,  /* 0x3fe224b74c1d192a */
      5.69756453482978431069e-01,  /* 0x3fe23b71e2cc9e6a */
      5.72521144698072359525e-01,  /* 0x3fe25217dd17e501 */
      5.75276017956117824426e-01,  /* 0x3fe268a940696da6 */
      5.78021083869819540801e-01,  /* 0x3fe27f261273d1b3 */
      5.80756353567670302596e-01,  /* 0x3fe2958e59308e30 */
      5.83481838685214859730e-01,  /* 0x3fe2abe21aded073 */
      5.86197551356360535557e-01,  /* 0x3fe2c2215e024465 */
      5.88903504204738026395e-01,  /* 0x3fe2d84c2961e48b */
      5.91599710335111383941e-01,  /* 0x3fe2ee628406cbca */
      5.94286183324841177367e-01,  /* 0x3fe30464753b090a */
      5.96962937215401501234e-01,  /* 0x3fe31a52048874be */
      5.99629986503951384336e-01,  /* 0x3fe3302b39b78856 */
      6.02287346134964152178e-01,  /* 0x3fe345f01cce37bb */
      6.04935031491913965951e-01,  /* 0x3fe35ba0b60eccce */
      6.07573058389022313541e-01,  /* 0x3fe3713d0df6c503 */
      6.10201443063065118722e-01,  /* 0x3fe386c52d3db11e */
      6.12820202165241245673e-01,  /* 0x3fe39c391cd41719 */
      6.15429352753104952356e-01,  /* 0x3fe3b198e5e2564a */
      6.18028912282561737612e-01,  /* 0x3fe3c6e491c78dc4 */
      6.20618898599929469384e-01,  /* 0x3fe3dc1c2a188504 */
      6.23199329934065904268e-01,  /* 0x3fe3f13fb89e96f4 */
      6.25770224888563042498e-01,  /* 0x3fe4064f47569f48 */
      6.28331602434009650615e-01,  /* 0x3fe41b4ae06fea41 */
      6.30883481900321840818e-01,  /* 0x3fe430328e4b26d5 */
      6.33425882969144482537e-01,  /* 0x3fe445065b795b55 */
      6.35958825666321447834e-01,  /* 0x3fe459c652badc7f */
      6.38482330354437466191e-01,  /* 0x3fe46e727efe4715 */
      6.40996417725432032775e-01,  /* 0x3fe4830aeb5f7bfd */
      6.43501108793284370968e-01,  /* 0x3fe4978fa3269ee1 */
      6.45996424886771558604e-01,  /* 0x3fe4ac00b1c71762 */
      6.48482387642300484032e-01,  /* 0x3fe4c05e22de94e4 */
      6.50959018996812410762e-01,  /* 0x3fe4d4a8023414e8 */
      6.53426341180761927063e-01,  /* 0x3fe4e8de5bb6ec04 */
      6.55884376711170835605e-01,  /* 0x3fe4fd013b7dd17e */
      6.58333148384755983962e-01,  /* 0x3fe51110adc5ed81 */
      6.60772679271132590273e-01,  /* 0x3fe5250cbef1e9fa */
      6.63202992706093175102e-01,  /* 0x3fe538f57b89061e */
      6.65624112284960989250e-01,  /* 0x3fe54ccaf0362c8f */
      6.68036061856020157990e-01,  /* 0x3fe5608d29c70c34 */
      6.70438865514021320458e-01,  /* 0x3fe5743c352b33b9 */
      6.72832547593763097282e-01,  /* 0x3fe587d81f732fba */
      6.75217132663749830535e-01,  /* 0x3fe59b60f5cfab9d */
      6.77592645519925151909e-01,  /* 0x3fe5aed6c5909517 */
      6.79959111179481823228e-01,  /* 0x3fe5c2399c244260 */
      6.82316554874748071313e-01,  /* 0x3fe5d58987169b18 */
      6.84665002047148862907e-01,  /* 0x3fe5e8c6941043cf */
      6.87004478341244895212e-01,  /* 0x3fe5fbf0d0d5cc49 */
      6.89335009598845749323e-01,  /* 0x3fe60f084b46e05e */
      6.91656621853199760075e-01,  /* 0x3fe6220d115d7b8d */
      6.93969341323259825138e-01,  /* 0x3fe634ff312d1f3b */
      6.96273194408023488045e-01,  /* 0x3fe647deb8e20b8f */
      6.98568207680949848637e-01,  /* 0x3fe65aabb6c07b02 */
      7.00854407884450081312e-01,  /* 0x3fe66d663923e086 */
      7.03131821924453670469e-01,  /* 0x3fe6800e4e7e2857 */
      7.05400476865049030906e-01,  /* 0x3fe692a40556fb6a */
      7.07660399923197958039e-01,  /* 0x3fe6a5276c4b0575 */
      7.09911618463524796141e-01,  /* 0x3fe6b798920b3d98 */
      7.12154159993178659249e-01,  /* 0x3fe6c9f7855c3198 */
      7.14388052156768926793e-01,  /* 0x3fe6dc44551553ae */
      7.16613322731374569052e-01,  /* 0x3fe6ee7f10204aef */
      7.18829999621624415873e-01,  /* 0x3fe700a7c5784633 */
      7.21038110854851588272e-01,  /* 0x3fe712be84295198 */
      7.23237684576317874097e-01,  /* 0x3fe724c35b4fae7b */
      7.25428749044510712274e-01,  /* 0x3fe736b65a172dff */
      7.27611332626510676214e-01,  /* 0x3fe748978fba8e0f */
      7.29785463793429123314e-01,  /* 0x3fe75a670b82d8d8 */
      7.31951171115916565668e-01,  /* 0x3fe76c24dcc6c6c0 */
      7.34108483259739652560e-01,  /* 0x3fe77dd112ea22c7 */
      7.36257428981428097003e-01,  /* 0x3fe78f6bbd5d315e */
      7.38398037123989547936e-01,  /* 0x3fe7a0f4eb9c19a2 */
      7.40530336612692630105e-01,  /* 0x3fe7b26cad2e50fd */
      7.42654356450917929600e-01,  /* 0x3fe7c3d311a6092b */
      7.44770125716075148681e-01,  /* 0x3fe7d528289fa093 */
      7.46877673555587429099e-01,  /* 0x3fe7e66c01c114fd */
      7.48977029182941400620e-01,  /* 0x3fe7f79eacb97898 */
      7.51068221873802288613e-01,  /* 0x3fe808c03940694a */
      7.53151280962194302759e-01,  /* 0x3fe819d0b7158a4c */
      7.55226235836744863583e-01,  /* 0x3fe82ad036000005 */
      7.57293115936992444759e-01,  /* 0x3fe83bbec5cdee22 */
      7.59351950749757920178e-01,  /* 0x3fe84c9c7653f7ea */
      7.61402769805578416573e-01,  /* 0x3fe85d69576cc2c5 */
      7.63445602675201784315e-01,  /* 0x3fe86e2578f87ae5 */
      7.65480478966144461950e-01,  /* 0x3fe87ed0eadc5a2a */
      7.67507428319308182552e-01,  /* 0x3fe88f6bbd023118 */
      7.69526480405658186434e-01,  /* 0x3fe89ff5ff57f1f7 */
      7.71537664922959498526e-01,  /* 0x3fe8b06fc1cf3dfe */
      7.73541011592573490852e-01,  /* 0x3fe8c0d9145cf49d */
      7.75536550156311621507e-01,  /* 0x3fe8d13206f8c4ca */
      7.77524310373347682379e-01,  /* 0x3fe8e17aa99cc05d */
      7.79504322017186335181e-01,  /* 0x3fe8f1b30c44f167 */
      7.81476614872688268854e-01,  /* 0x3fe901db3eeef187 */
      7.83441218733151756304e-01,  /* 0x3fe911f35199833b */
      7.85398163397448278999e-01}; /* 0x3fe921fb54442d18 */

  /* Some constants. */

  static double pi = 3.1415926535897932e+00, /* 0x400921fb54442d18 */
      piby2 = 1.5707963267948966e+00,        /* 0x3ff921fb54442d18 */
      piby4 = 7.8539816339744831e-01,        /* 0x3fe921fb54442d18 */
      three_piby4 = 2.3561944901923449e+00;  /* 0x4002d97c7f3321d2 */

  double u, v, vbyu, q, s, uu, r;
  unsigned int swap_vu, index, xzero, yzero, xnan, ynan, xinf, yinf;
  int xexp, yexp, diffexp;

  double x = fx;
  double y = fy;

  /* Find properties of arguments x and y. */

  __UINT8_T ux, aux, xneg, uy, auy, yneg;

  GET_BITS_DP64(x, ux);
  GET_BITS_DP64(y, uy);
  aux = ux & ~SIGNBIT_DP64;
  auy = uy & ~SIGNBIT_DP64;
  xexp = (int)((ux & EXPBITS_DP64) >> EXPSHIFTBITS_DP64);
  yexp = (int)((uy & EXPBITS_DP64) >> EXPSHIFTBITS_DP64);
  xneg = ux & SIGNBIT_DP64;
  yneg = uy & SIGNBIT_DP64;
  xzero = (aux == 0);
  yzero = (auy == 0);
  xnan = (aux > PINFBITPATT_DP64);
  ynan = (auy > PINFBITPATT_DP64);
  xinf = (aux == PINFBITPATT_DP64);
  yinf = (auy == PINFBITPATT_DP64);

  diffexp = yexp - xexp;

  /* Special cases */

  if (xnan)
    return fx + fx; /* Raise invalid if it's a signalling NaN */
  else if (ynan)
    return (float)(y + y); /* Raise invalid if it's a signalling NaN */
  else if (yzero) {        /* Zero y gives +-0 for positive x
                              and +-pi for negative x */
    if (xneg) {
      if (yneg)
        return valf_with_flags((float)-pi, AMD_F_INEXACT);
      else
        return valf_with_flags((float)pi, AMD_F_INEXACT);
    } else
      return (float)y;
  } else if (xzero) { /* Zero x gives +- pi/2
                         depending on sign of y */
    if (yneg)
      return valf_with_flags((float)-piby2, AMD_F_INEXACT);
    else
      valf_with_flags((float)piby2, AMD_F_INEXACT);
  }

  if (diffexp > 26) { /* abs(y)/abs(x) > 2^26 => arctan(x/y)
                         is insignificant compared to piby2 */
    if (yneg)
      return valf_with_flags((float)-piby2, AMD_F_INEXACT);
    else
      return valf_with_flags((float)piby2, AMD_F_INEXACT);
  } else if (diffexp < -13 &&
             (!xneg)) { /* x positive and dominant over y by a factor of 2^13.
                           In this case atan(y/x) is y/x to machine accuracy. */

    if (diffexp < -150) /* Result underflows */
    {
      if (yneg)
        return valf_with_flags(-0.0F, AMD_F_INEXACT | AMD_F_UNDERFLOW);
      else
        return valf_with_flags(0.0F, AMD_F_INEXACT | AMD_F_UNDERFLOW);
    } else {
      if (diffexp < -126) {
        /* Result will likely be denormalized */
        y = scaleDouble_1(y, 100);
        y /= x;
        /* Now y is 2^100 times the true result. Scale it back down. */
        GET_BITS_DP64(y, uy);
        scaleDownDouble(uy, 100, &uy);
        PUT_BITS_DP64(uy, y);
        if ((uy & EXPBITS_DP64) == 0)
          return valf_with_flags((float)y, AMD_F_INEXACT | AMD_F_UNDERFLOW);
        else
          return (float)y;
      } else
        return (float)(y / x);
    }
  } else if (diffexp < -26 &&
             xneg) { /* abs(x)/abs(y) > 2^56 and x < 0 => arctan(y/x)
                        is insignificant compared to pi */
    if (yneg)
      return valf_with_flags((float)-pi, AMD_F_INEXACT);
    else
      return valf_with_flags((float)pi, AMD_F_INEXACT);
  } else if (yinf && xinf) { /* If abs(x) and abs(y) are both infinity
                                return +-pi/4 or +- 3pi/4 according to
                                signs.  */
    if (xneg) {
      if (yneg)
        return valf_with_flags((float)-three_piby4, AMD_F_INEXACT);
      else
        return valf_with_flags((float)three_piby4, AMD_F_INEXACT);
    } else {
      if (yneg)
        return valf_with_flags((float)-piby4, AMD_F_INEXACT);
      else
        return valf_with_flags((float)piby4, AMD_F_INEXACT);
    }
  }

  /* General case: take absolute values of arguments */

  u = x;
  v = y;
  if (xneg)
    u = -x;
  if (yneg)
    v = -y;

  /* Swap u and v if necessary to obtain 0 < v < u. Compute v/u. */

  swap_vu = (u < v);
  if (swap_vu) {
    uu = u;
    u = v;
    v = uu;
  }
  vbyu = v / u;

  if (vbyu > 0.0625) { /* General values of v/u. Use a look-up
                          table and series expansion. */

    index = (int)(256 * vbyu + 0.5);
    r = (256 * v - index * u) / (256 * u + index * v);

    /* Polynomial approximation to atan(vbyu) */

    s = r * r;
    q = atan_jby256[index - 16] + r - r * s * 0.33333333333224095522;
  } else if (vbyu < 1.e-4) { /* v/u is small enough that atan(v/u) = v/u */
    q = vbyu;
  } else /* vbyu <= 0.0625 */
  {
    /* Small values of v/u. Use a series expansion */

    s = vbyu * vbyu;
    q = vbyu -
        vbyu * s * (0.33333333333333170500 -
                    s * (0.19999999999393223405 - s * 0.14285713561807169030));
  }

  /* Tidy-up according to which quadrant the arguments lie in */

  if (swap_vu) {
    q = piby2 - q;
  }
  if (xneg) {
    q = pi - q;
  }
  if (yneg)
    q = -q;
  return (float)q;
}
