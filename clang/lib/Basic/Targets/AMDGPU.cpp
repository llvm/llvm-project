//===--- AMDGPU.cpp - Implement AMDGPU target feature support -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements AMDGPU TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
using namespace clang;
using namespace clang::targets;

namespace clang {
namespace targets {

// If you edit the description strings, make sure you update
// getPointerWidthV().

const LangASMap AMDGPUTargetInfo::AMDGPUAddrSpaceMap = {
    {LangAS::Default, llvm::AMDGPUAS::FLAT_ADDRESS},
    {LangAS::opencl_global, llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {LangAS::opencl_local, llvm::AMDGPUAS::LOCAL_ADDRESS},
    {LangAS::opencl_constant, llvm::AMDGPUAS::CONSTANT_ADDRESS},
    {LangAS::opencl_private, llvm::AMDGPUAS::PRIVATE_ADDRESS},
    {LangAS::opencl_generic, llvm::AMDGPUAS::FLAT_ADDRESS},
    {LangAS::opencl_global_device, llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {LangAS::opencl_global_host, llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {LangAS::cuda_device, llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {LangAS::cuda_constant, llvm::AMDGPUAS::CONSTANT_ADDRESS},
    {LangAS::cuda_shared, llvm::AMDGPUAS::LOCAL_ADDRESS},
    {LangAS::sycl_global, llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {LangAS::sycl_global_device, llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {LangAS::sycl_global_host, llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {LangAS::sycl_local, llvm::AMDGPUAS::LOCAL_ADDRESS},
    {LangAS::sycl_private, llvm::AMDGPUAS::PRIVATE_ADDRESS},
    {LangAS::ptr32_sptr, llvm::AMDGPUAS::FLAT_ADDRESS},
    {LangAS::ptr32_uptr, llvm::AMDGPUAS::FLAT_ADDRESS},
    {LangAS::ptr64, llvm::AMDGPUAS::FLAT_ADDRESS},
    {LangAS::hlsl_groupshared, llvm::AMDGPUAS::FLAT_ADDRESS},
    {LangAS::hlsl_constant, llvm::AMDGPUAS::CONSTANT_ADDRESS},
    // FIXME(pr/122103): hlsl_private -> PRIVATE is wrong, but at least this
    // will break loudly.
    {LangAS::hlsl_private, llvm::AMDGPUAS::PRIVATE_ADDRESS},
    {LangAS::hlsl_device, llvm::AMDGPUAS::GLOBAL_ADDRESS},
    {LangAS::hlsl_input, llvm::AMDGPUAS::PRIVATE_ADDRESS},
    {LangAS::hlsl_output, llvm::AMDGPUAS::PRIVATE_ADDRESS},
    {LangAS::hlsl_push_constant, llvm::AMDGPUAS::GLOBAL_ADDRESS},
};

} // namespace targets
} // namespace clang

static constexpr int NumBuiltins =
    clang::AMDGPU::LastTSBuiltin - Builtin::FirstTSBuiltin;

#define GET_BUILTIN_STR_TABLE
#include "clang/Basic/BuiltinsAMDGPU.inc"
#undef GET_BUILTIN_STR_TABLE

static constexpr Builtin::Info BuiltinInfos[] = {
#define GET_BUILTIN_INFOS
#include "clang/Basic/BuiltinsAMDGPU.inc"
#undef GET_BUILTIN_INFOS
};
static_assert(std::size(BuiltinInfos) == NumBuiltins);

const char *const AMDGPUTargetInfo::GCCRegNames[] = {
  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
  "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
  "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
  "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35",
  "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44",
  "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52", "v53",
  "v54", "v55", "v56", "v57", "v58", "v59", "v60", "v61", "v62",
  "v63", "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71",
  "v72", "v73", "v74", "v75", "v76", "v77", "v78", "v79", "v80",
  "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89",
  "v90", "v91", "v92", "v93", "v94", "v95", "v96", "v97", "v98",
  "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107",
  "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116",
  "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125",
  "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134",
  "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143",
  "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152",
  "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161",
  "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170",
  "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179",
  "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188",
  "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197",
  "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206",
  "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215",
  "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224",
  "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233",
  "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242",
  "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251",
  "v252", "v253", "v254", "v255", "s0", "s1", "s2", "s3", "s4",
  "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13",
  "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22",
  "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
  "s32", "s33", "s34", "s35", "s36", "s37", "s38", "s39", "s40",
  "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49",
  "s50", "s51", "s52", "s53", "s54", "s55", "s56", "s57", "s58",
  "s59", "s60", "s61", "s62", "s63", "s64", "s65", "s66", "s67",
  "s68", "s69", "s70", "s71", "s72", "s73", "s74", "s75", "s76",
  "s77", "s78", "s79", "s80", "s81", "s82", "s83", "s84", "s85",
  "s86", "s87", "s88", "s89", "s90", "s91", "s92", "s93", "s94",
  "s95", "s96", "s97", "s98", "s99", "s100", "s101", "s102", "s103",
  "s104", "s105", "s106", "s107", "s108", "s109", "s110", "s111", "s112",
  "s113", "s114", "s115", "s116", "s117", "s118", "s119", "s120", "s121",
  "s122", "s123", "s124", "s125", "s126", "s127", "exec", "vcc", "scc",
  "m0", "flat_scratch", "exec_lo", "exec_hi", "vcc_lo", "vcc_hi",
  "flat_scratch_lo", "flat_scratch_hi",
  "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8",
  "a9", "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17",
  "a18", "a19", "a20", "a21", "a22", "a23", "a24", "a25", "a26",
  "a27", "a28", "a29", "a30", "a31", "a32", "a33", "a34", "a35",
  "a36", "a37", "a38", "a39", "a40", "a41", "a42", "a43", "a44",
  "a45", "a46", "a47", "a48", "a49", "a50", "a51", "a52", "a53",
  "a54", "a55", "a56", "a57", "a58", "a59", "a60", "a61", "a62",
  "a63", "a64", "a65", "a66", "a67", "a68", "a69", "a70", "a71",
  "a72", "a73", "a74", "a75", "a76", "a77", "a78", "a79", "a80",
  "a81", "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89",
  "a90", "a91", "a92", "a93", "a94", "a95", "a96", "a97", "a98",
  "a99", "a100", "a101", "a102", "a103", "a104", "a105", "a106", "a107",
  "a108", "a109", "a110", "a111", "a112", "a113", "a114", "a115", "a116",
  "a117", "a118", "a119", "a120", "a121", "a122", "a123", "a124", "a125",
  "a126", "a127", "a128", "a129", "a130", "a131", "a132", "a133", "a134",
  "a135", "a136", "a137", "a138", "a139", "a140", "a141", "a142", "a143",
  "a144", "a145", "a146", "a147", "a148", "a149", "a150", "a151", "a152",
  "a153", "a154", "a155", "a156", "a157", "a158", "a159", "a160", "a161",
  "a162", "a163", "a164", "a165", "a166", "a167", "a168", "a169", "a170",
  "a171", "a172", "a173", "a174", "a175", "a176", "a177", "a178", "a179",
  "a180", "a181", "a182", "a183", "a184", "a185", "a186", "a187", "a188",
  "a189", "a190", "a191", "a192", "a193", "a194", "a195", "a196", "a197",
  "a198", "a199", "a200", "a201", "a202", "a203", "a204", "a205", "a206",
  "a207", "a208", "a209", "a210", "a211", "a212", "a213", "a214", "a215",
  "a216", "a217", "a218", "a219", "a220", "a221", "a222", "a223", "a224",
  "a225", "a226", "a227", "a228", "a229", "a230", "a231", "a232", "a233",
  "a234", "a235", "a236", "a237", "a238", "a239", "a240", "a241", "a242",
  "a243", "a244", "a245", "a246", "a247", "a248", "a249", "a250", "a251",
  "a252", "a253", "a254", "a255",
  // High VGPRs v256-v1023, addressable only on gfx1250+ (1024-addressable-vgprs).
  // Kept last so getGCCRegNames() can drop them on targets without that feature.
  "v256", "v257", "v258", "v259", "v260", "v261", "v262", "v263", "v264",
  "v265", "v266", "v267", "v268", "v269", "v270", "v271", "v272", "v273",
  "v274", "v275", "v276", "v277", "v278", "v279", "v280", "v281", "v282",
  "v283", "v284", "v285", "v286", "v287", "v288", "v289", "v290", "v291",
  "v292", "v293", "v294", "v295", "v296", "v297", "v298", "v299", "v300",
  "v301", "v302", "v303", "v304", "v305", "v306", "v307", "v308", "v309",
  "v310", "v311", "v312", "v313", "v314", "v315", "v316", "v317", "v318",
  "v319", "v320", "v321", "v322", "v323", "v324", "v325", "v326", "v327",
  "v328", "v329", "v330", "v331", "v332", "v333", "v334", "v335", "v336",
  "v337", "v338", "v339", "v340", "v341", "v342", "v343", "v344", "v345",
  "v346", "v347", "v348", "v349", "v350", "v351", "v352", "v353", "v354",
  "v355", "v356", "v357", "v358", "v359", "v360", "v361", "v362", "v363",
  "v364", "v365", "v366", "v367", "v368", "v369", "v370", "v371", "v372",
  "v373", "v374", "v375", "v376", "v377", "v378", "v379", "v380", "v381",
  "v382", "v383", "v384", "v385", "v386", "v387", "v388", "v389", "v390",
  "v391", "v392", "v393", "v394", "v395", "v396", "v397", "v398", "v399",
  "v400", "v401", "v402", "v403", "v404", "v405", "v406", "v407", "v408",
  "v409", "v410", "v411", "v412", "v413", "v414", "v415", "v416", "v417",
  "v418", "v419", "v420", "v421", "v422", "v423", "v424", "v425", "v426",
  "v427", "v428", "v429", "v430", "v431", "v432", "v433", "v434", "v435",
  "v436", "v437", "v438", "v439", "v440", "v441", "v442", "v443", "v444",
  "v445", "v446", "v447", "v448", "v449", "v450", "v451", "v452", "v453",
  "v454", "v455", "v456", "v457", "v458", "v459", "v460", "v461", "v462",
  "v463", "v464", "v465", "v466", "v467", "v468", "v469", "v470", "v471",
  "v472", "v473", "v474", "v475", "v476", "v477", "v478", "v479", "v480",
  "v481", "v482", "v483", "v484", "v485", "v486", "v487", "v488", "v489",
  "v490", "v491", "v492", "v493", "v494", "v495", "v496", "v497", "v498",
  "v499", "v500", "v501", "v502", "v503", "v504", "v505", "v506", "v507",
  "v508", "v509", "v510", "v511", "v512", "v513", "v514", "v515", "v516",
  "v517", "v518", "v519", "v520", "v521", "v522", "v523", "v524", "v525",
  "v526", "v527", "v528", "v529", "v530", "v531", "v532", "v533", "v534",
  "v535", "v536", "v537", "v538", "v539", "v540", "v541", "v542", "v543",
  "v544", "v545", "v546", "v547", "v548", "v549", "v550", "v551", "v552",
  "v553", "v554", "v555", "v556", "v557", "v558", "v559", "v560", "v561",
  "v562", "v563", "v564", "v565", "v566", "v567", "v568", "v569", "v570",
  "v571", "v572", "v573", "v574", "v575", "v576", "v577", "v578", "v579",
  "v580", "v581", "v582", "v583", "v584", "v585", "v586", "v587", "v588",
  "v589", "v590", "v591", "v592", "v593", "v594", "v595", "v596", "v597",
  "v598", "v599", "v600", "v601", "v602", "v603", "v604", "v605", "v606",
  "v607", "v608", "v609", "v610", "v611", "v612", "v613", "v614", "v615",
  "v616", "v617", "v618", "v619", "v620", "v621", "v622", "v623", "v624",
  "v625", "v626", "v627", "v628", "v629", "v630", "v631", "v632", "v633",
  "v634", "v635", "v636", "v637", "v638", "v639", "v640", "v641", "v642",
  "v643", "v644", "v645", "v646", "v647", "v648", "v649", "v650", "v651",
  "v652", "v653", "v654", "v655", "v656", "v657", "v658", "v659", "v660",
  "v661", "v662", "v663", "v664", "v665", "v666", "v667", "v668", "v669",
  "v670", "v671", "v672", "v673", "v674", "v675", "v676", "v677", "v678",
  "v679", "v680", "v681", "v682", "v683", "v684", "v685", "v686", "v687",
  "v688", "v689", "v690", "v691", "v692", "v693", "v694", "v695", "v696",
  "v697", "v698", "v699", "v700", "v701", "v702", "v703", "v704", "v705",
  "v706", "v707", "v708", "v709", "v710", "v711", "v712", "v713", "v714",
  "v715", "v716", "v717", "v718", "v719", "v720", "v721", "v722", "v723",
  "v724", "v725", "v726", "v727", "v728", "v729", "v730", "v731", "v732",
  "v733", "v734", "v735", "v736", "v737", "v738", "v739", "v740", "v741",
  "v742", "v743", "v744", "v745", "v746", "v747", "v748", "v749", "v750",
  "v751", "v752", "v753", "v754", "v755", "v756", "v757", "v758", "v759",
  "v760", "v761", "v762", "v763", "v764", "v765", "v766", "v767", "v768",
  "v769", "v770", "v771", "v772", "v773", "v774", "v775", "v776", "v777",
  "v778", "v779", "v780", "v781", "v782", "v783", "v784", "v785", "v786",
  "v787", "v788", "v789", "v790", "v791", "v792", "v793", "v794", "v795",
  "v796", "v797", "v798", "v799", "v800", "v801", "v802", "v803", "v804",
  "v805", "v806", "v807", "v808", "v809", "v810", "v811", "v812", "v813",
  "v814", "v815", "v816", "v817", "v818", "v819", "v820", "v821", "v822",
  "v823", "v824", "v825", "v826", "v827", "v828", "v829", "v830", "v831",
  "v832", "v833", "v834", "v835", "v836", "v837", "v838", "v839", "v840",
  "v841", "v842", "v843", "v844", "v845", "v846", "v847", "v848", "v849",
  "v850", "v851", "v852", "v853", "v854", "v855", "v856", "v857", "v858",
  "v859", "v860", "v861", "v862", "v863", "v864", "v865", "v866", "v867",
  "v868", "v869", "v870", "v871", "v872", "v873", "v874", "v875", "v876",
  "v877", "v878", "v879", "v880", "v881", "v882", "v883", "v884", "v885",
  "v886", "v887", "v888", "v889", "v890", "v891", "v892", "v893", "v894",
  "v895", "v896", "v897", "v898", "v899", "v900", "v901", "v902", "v903",
  "v904", "v905", "v906", "v907", "v908", "v909", "v910", "v911", "v912",
  "v913", "v914", "v915", "v916", "v917", "v918", "v919", "v920", "v921",
  "v922", "v923", "v924", "v925", "v926", "v927", "v928", "v929", "v930",
  "v931", "v932", "v933", "v934", "v935", "v936", "v937", "v938", "v939",
  "v940", "v941", "v942", "v943", "v944", "v945", "v946", "v947", "v948",
  "v949", "v950", "v951", "v952", "v953", "v954", "v955", "v956", "v957",
  "v958", "v959", "v960", "v961", "v962", "v963", "v964", "v965", "v966",
  "v967", "v968", "v969", "v970", "v971", "v972", "v973", "v974", "v975",
  "v976", "v977", "v978", "v979", "v980", "v981", "v982", "v983", "v984",
  "v985", "v986", "v987", "v988", "v989", "v990", "v991", "v992", "v993",
  "v994", "v995", "v996", "v997", "v998", "v999", "v1000", "v1001", "v1002",
  "v1003", "v1004", "v1005", "v1006", "v1007", "v1008", "v1009", "v1010", "v1011",
  "v1012", "v1013", "v1014", "v1015", "v1016", "v1017", "v1018", "v1019", "v1020",
  "v1021", "v1022", "v1023"
};

// Number of trailing high-VGPR names (v256-v1023) in GCCRegNames that are only
// valid on gfx1250+ (targets with the 1024-addressable-vgprs feature).
static constexpr size_t NumHighVGPRRegNames = 1024 - 256;

ArrayRef<const char *> AMDGPUTargetInfo::getGCCRegNames() const {
  ArrayRef<const char *> Names(GCCRegNames);
  // v256-v1023 are addressable only on gfx1250+; hide them elsewhere so they
  // cannot be named in inline asm / clobbers on targets with just 256 VGPRs.
  if (!has1024AddressableVGPRs())
    Names = Names.drop_back(NumHighVGPRRegNames);
  return Names;
}

bool AMDGPUTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeatureVec) const {

  using namespace llvm::AMDGPU;

  if (!TargetInfo::initFeatureMap(Features, Diags, CPU, FeatureVec))
    return false;

  auto HasError = fillAMDGPUFeatureMap(CPU, getTriple(), Features);
  switch (HasError.first) {
  default:
    break;
  case llvm::AMDGPU::INVALID_FEATURE_COMBINATION:
    Diags.Report(diag::err_invalid_feature_combination) << HasError.second;
    return false;
  case llvm::AMDGPU::UNSUPPORTED_TARGET_FEATURE:
    Diags.Report(diag::err_opt_not_valid_on_target) << HasError.second;
    return false;
  }

  return true;
}

void AMDGPUTargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  if (getTriple().isAMDGCN())
    llvm::AMDGPU::fillValidArchListAMDGCN(Values, getTriple().getSubArch());
  else
    llvm::AMDGPU::fillValidArchListR600(Values);
}

AMDGPUTargetInfo::AMDGPUTargetInfo(const llvm::Triple &Triple,
                                   const TargetOptions &Opts)
    : TargetInfo(Triple),
      GPUKind(Triple.isAMDGCN()
                  ? (Opts.CPU.empty() ? llvm::AMDGPU::getGPUKindFromSubArch(
                                            Triple.getSubArch())
                                      : llvm::AMDGPU::parseArchAMDGCN(Opts.CPU))
                  : llvm::AMDGPU::parseArchR600(Opts.CPU)),
      GPUFeatures(Triple.isAMDGCN() ? llvm::AMDGPU::getArchAttrAMDGCN(GPUKind)
                                    : llvm::AMDGPU::getArchAttrR600(GPUKind)) {
  resetDataLayout();

  AddrSpaceMap = &AMDGPUAddrSpaceMap;
  UseAddrSpaceMapMangling = true;

  if (Triple.isAMDGCN()) {
    // __bf16 is always available as a load/store only type on AMDGCN.
    BFloat16Width = BFloat16Align = 16;
    BFloat16Format = &llvm::APFloat::BFloat();
  }

  // TODO: This is not really true for targets without half support, but also
  // should just be assumed true for the dummy target.
  HasFastHalfType = true;
  HasFloat16 = true;
  WavefrontSize = (GPUFeatures & llvm::AMDGPU::FEATURE_WAVE32) ? 32 : 64;

  // Set pointer width and alignment for the generic address space.
  PointerWidth = PointerAlign = getPointerWidthV(LangAS::Default);
  if (getMaxPointerWidth() == 64) {
    LongWidth = LongAlign = 64;
    SizeType = UnsignedLong;
    PtrDiffType = SignedLong;
    IntPtrType = SignedLong;
  }

  MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
  CUMode = !(GPUFeatures & llvm::AMDGPU::FEATURE_WGP);

  for (auto F : {"image-insts", "gws", "vmem-to-lds-load-insts"}) {
    if (GPUKind != llvm::AMDGPU::GK_NONE)
      ReadOnlyFeatures.insert(F);
  }
  HalfArgsAndReturns = true;

  if (Opts.AMDGPUXnackState != TargetOptions::AMDGPUFeatureState::Any) {
    OffloadArchFeatures["xnack"] =
        Opts.AMDGPUXnackState == TargetOptions::AMDGPUFeatureState::Enabled;
  }

  if (Opts.AMDGPUSramEccState != TargetOptions::AMDGPUFeatureState::Any) {
    OffloadArchFeatures["sramecc"] =
        Opts.AMDGPUSramEccState == TargetOptions::AMDGPUFeatureState::Enabled;
  }
}

void AMDGPUTargetInfo::adjust(DiagnosticsEngine &Diags, LangOptions &Opts,
                              const TargetInfo *Aux) {
  TargetInfo::adjust(Diags, Opts, Aux);
  AtomicOpts = AtomicOptions(Opts);
}

llvm::SmallVector<Builtin::InfosShard>
AMDGPUTargetInfo::getTargetBuiltins() const {
  return {{&BuiltinStrings, BuiltinInfos}};
}

void AMDGPUTargetInfo::getTargetDefines(const LangOptions &Opts,
                                        MacroBuilder &Builder) const {
  Builder.defineMacro("__AMD__");
  Builder.defineMacro("__AMDGPU__");

  if (getTriple().isAMDGCN())
    Builder.defineMacro("__AMDGCN__");
  else
    Builder.defineMacro("__R600__");

  // TODO: __HAS_FMAF__, __HAS_LDEXPF__, __HAS_FP64__ are deprecated and will be
  // removed in the near future.
  if (hasFMAF())
    Builder.defineMacro("__HAS_FMAF__");
  if (hasFastFMAF())
    Builder.defineMacro("FP_FAST_FMAF");
  if (hasLDEXPF())
    Builder.defineMacro("__HAS_LDEXPF__");
  if (hasFP64())
    Builder.defineMacro("__HAS_FP64__");
  if (hasFastFMA())
    Builder.defineMacro("FP_FAST_FMA");
  if (HasFastHalfType)
    Builder.defineMacro("FP_FAST_FMA_HALF");

  Builder.defineMacro("__AMDGCN_CUMODE__", Twine(CUMode));

  // Legacy HIP host code relies on these default attributes to be defined.
  bool IsHIPHost = Opts.HIP && !Opts.CUDAIsDevice;
  if (GPUKind == llvm::AMDGPU::GK_NONE && !IsHIPHost)
    return;

  llvm::SmallString<16> CanonName =
      (getTriple().isAMDGCN() ? getArchNameAMDGCN(GPUKind)
                              : getArchNameR600(GPUKind));

  // Sanitize the name of generic targets, the only names containing '-'.
  // e.g. gfx10-1-generic -> gfx10_1_generic
  llvm::replace(CanonName, '-', '_');

  Builder.defineMacro(Twine("__") + Twine(CanonName) + Twine("__"));
  // Emit macros for gfx family e.g. gfx906 -> __GFX9__, gfx1030 -> __GFX10___
  if (getTriple().isAMDGCN() && !IsHIPHost) {
    assert(StringRef(CanonName).starts_with("gfx") &&
           "Invalid amdgcn canonical name");
    StringRef CanonFamilyName = getArchFamilyNameAMDGCN(GPUKind);
    Builder.defineMacro(Twine("__") + Twine(CanonFamilyName.upper()) +
                        Twine("__"));
    Builder.defineMacro("__amdgcn_processor__",
                        Twine("\"") + Twine(CanonName) + Twine("\""));
    Builder.defineMacro(
        "__amdgcn_target_id__",
        Twine("\"") +
            Twine(getCanonicalTargetID(getArchNameAMDGCN(GPUKind),
                                       OffloadArchFeatures)) +
            Twine("\""));
    for (auto F : getAllPossibleTargetIDFeatures(getTriple(), CanonName)) {
      auto Loc = OffloadArchFeatures.find(F);
      if (Loc != OffloadArchFeatures.end()) {
        std::string NewF = F.str();
        llvm::replace(NewF, '-', '_');
        Builder.defineMacro(Twine("__amdgcn_feature_") + Twine(NewF) +
                                Twine("__"),
                            Loc->second ? "1" : "0");
      }
    }
  }

  if (Opts.AtomicIgnoreDenormalMode)
    Builder.defineMacro("__AMDGCN_UNSAFE_FP_ATOMICS__");
}

void AMDGPUTargetInfo::setAuxTarget(const TargetInfo *Aux) {
  assert(HalfFormat == Aux->HalfFormat);
  assert(FloatFormat == Aux->FloatFormat);
  assert(DoubleFormat == Aux->DoubleFormat);

  // On x86_64 long double is 80-bit extended precision format, which is
  // not supported by AMDGPU. 128-bit floating point format is also not
  // supported by AMDGPU. Therefore keep its own format for these two types.
  auto SaveLongDoubleFormat = LongDoubleFormat;
  auto SaveFloat128Format = Float128Format;
  auto SaveLongDoubleWidth = LongDoubleWidth;
  auto SaveLongDoubleAlign = LongDoubleAlign;
  copyAuxTarget(Aux);
  LongDoubleFormat = SaveLongDoubleFormat;
  Float128Format = SaveFloat128Format;
  LongDoubleWidth = SaveLongDoubleWidth;
  LongDoubleAlign = SaveLongDoubleAlign;
  // For certain builtin types support on the host target, claim they are
  // support to pass the compilation of the host code during the device-side
  // compilation.
  // FIXME: As the side effect, we also accept `__float128` uses in the device
  // code. To rejct these builtin types supported in the host target but not in
  // the device target, one approach would support `device_builtin` attribute
  // so that we could tell the device builtin types from the host ones. The
  // also solves the different representations of the same builtin type, such
  // as `size_t` in the MSVC environment.
  if (Aux->hasFloat128Type()) {
    HasFloat128 = true;
    Float128Format = DoubleFormat;
  }
}
