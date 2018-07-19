// RUN: %clang_cc1 -x c++ -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s

const char *l255 = "\xff";
const char *l254 = "\xfe";
const char *l253 = "\xfd";
const char *l252 = "\xfc";
const char *l251 = "\xfb";
const char *l250 = "\xfa";
const char *l249 = "\xf9";
const char *l248 = "\xf8";
const char *l247 = "\xf7";
const char *l246 = "\xf6";
const char *l245 = "\xf5";
const char *l244 = "\xf4";
const char *l243 = "\xf3";
const char *l242 = "\xf2";
const char *l241 = "\xf1";
const char *l240 = "\xf0";
const char *l239 = "\xef";
const char *l238 = "\xee";
const char *l237 = "\xed";
const char *l236 = "\xec";
const char *l235 = "\xeb";
const char *l234 = "\xea";
const char *l233 = "\xe9";
const char *l232 = "\xe8";
const char *l231 = "\xe7";
const char *l230 = "\xe6";
const char *l229 = "\xe5";
const char *l228 = "\xe4";
const char *l227 = "\xe3";
const char *l226 = "\xe2";
const char *l225 = "\xe1";
const char *l224 = "\xe0";
const char *l223 = "\xdf";
const char *l222 = "\xde";
const char *l221 = "\xdd";
const char *l220 = "\xdc";
const char *l219 = "\xdb";
const char *l218 = "\xda";
const char *l217 = "\xd9";
const char *l216 = "\xd8";
const char *l215 = "\xd7";
const char *l214 = "\xd6";
const char *l213 = "\xd5";
const char *l212 = "\xd4";
const char *l211 = "\xd3";
const char *l210 = "\xd2";
const char *l209 = "\xd1";
const char *l208 = "\xd0";
const char *l207 = "\xcf";
const char *l206 = "\xce";
const char *l205 = "\xcd";
const char *l204 = "\xcc";
const char *l203 = "\xcb";
const char *l202 = "\xca";
const char *l201 = "\xc9";
const char *l200 = "\xc8";
const char *l199 = "\xc7";
const char *l198 = "\xc6";
const char *l197 = "\xc5";
const char *l196 = "\xc4";
const char *l195 = "\xc3";
const char *l194 = "\xc2";
const char *l193 = "\xc1";
const char *l192 = "\xc0";
const char *l191 = "\xbf";
const char *l190 = "\xbe";
const char *l189 = "\xbd";
const char *l188 = "\xbc";
const char *l187 = "\xbb";
const char *l186 = "\xba";
const char *l185 = "\xb9";
const char *l184 = "\xb8";
const char *l183 = "\xb7";
const char *l182 = "\xb6";
const char *l181 = "\xb5";
const char *l180 = "\xb4";
const char *l179 = "\xb3";
const char *l178 = "\xb2";
const char *l177 = "\xb1";
const char *l176 = "\xb0";
const char *l175 = "\xaf";
const char *l174 = "\xae";
const char *l173 = "\xad";
const char *l172 = "\xac";
const char *l171 = "\xab";
const char *l170 = "\xaa";
const char *l169 = "\xa9";
const char *l168 = "\xa8";
const char *l167 = "\xa7";
const char *l166 = "\xa6";
const char *l165 = "\xa5";
const char *l164 = "\xa4";
const char *l163 = "\xa3";
const char *l162 = "\xa2";
const char *l161 = "\xa1";
const char *l160 = "\xa0";
const char *l159 = "\x9f";
const char *l158 = "\x9e";
const char *l157 = "\x9d";
const char *l156 = "\x9c";
const char *l155 = "\x9b";
const char *l154 = "\x9a";
const char *l153 = "\x99";
const char *l152 = "\x98";
const char *l151 = "\x97";
const char *l150 = "\x96";
const char *l149 = "\x95";
const char *l148 = "\x94";
const char *l147 = "\x93";
const char *l146 = "\x92";
const char *l145 = "\x91";
const char *l144 = "\x90";
const char *l143 = "\x8f";
const char *l142 = "\x8e";
const char *l141 = "\x8d";
const char *l140 = "\x8c";
const char *l139 = "\x8b";
const char *l138 = "\x8a";
const char *l137 = "\x89";
const char *l136 = "\x88";
const char *l135 = "\x87";
const char *l134 = "\x86";
const char *l133 = "\x85";
const char *l132 = "\x84";
const char *l131 = "\x83";
const char *l130 = "\x82";
const char *l129 = "\x81";
const char *l128 = "\x80";
const char *l127 = "\x7f";
const char *l126 = "\x7e";
const char *l125 = "\x7d";
const char *l124 = "\x7c";
const char *l123 = "\x7b";
const char *l122 = "\x7a";
const char *l121 = "\x79";
const char *l120 = "\x78";
const char *l119 = "\x77";
const char *l118 = "\x76";
const char *l117 = "\x75";
const char *l116 = "\x74";
const char *l115 = "\x73";
const char *l114 = "\x72";
const char *l113 = "\x71";
const char *l112 = "\x70";
const char *l111 = "\x6f";
const char *l110 = "\x6e";
const char *l109 = "\x6d";
const char *l108 = "\x6c";
const char *l107 = "\x6b";
const char *l106 = "\x6a";
const char *l105 = "\x69";
const char *l104 = "\x68";
const char *l103 = "\x67";
const char *l102 = "\x66";
const char *l101 = "\x65";
const char *l100 = "\x64";
const char *l99 = "\x63";
const char *l98 = "\x62";
const char *l97 = "\x61";
const char *l96 = "\x60";
const char *l95 = "\x5f";
const char *l94 = "\x5e";
const char *l93 = "\x5d";
const char *l92 = "\x5c";
const char *l91 = "\x5b";
const char *l90 = "\x5a";
const char *l89 = "\x59";
const char *l88 = "\x58";
const char *l87 = "\x57";
const char *l86 = "\x56";
const char *l85 = "\x55";
const char *l84 = "\x54";
const char *l83 = "\x53";
const char *l82 = "\x52";
const char *l81 = "\x51";
const char *l80 = "\x50";
const char *l79 = "\x4f";
const char *l78 = "\x4e";
const char *l77 = "\x4d";
const char *l76 = "\x4c";
const char *l75 = "\x4b";
const char *l74 = "\x4a";
const char *l73 = "\x49";
const char *l72 = "\x48";
const char *l71 = "\x47";
const char *l70 = "\x46";
const char *l69 = "\x45";
const char *l68 = "\x44";
const char *l67 = "\x43";
const char *l66 = "\x42";
const char *l65 = "\x41";
const char *l64 = "\x40";
const char *l63 = "\x3f";
const char *l62 = "\x3e";
const char *l61 = "\x3d";
const char *l60 = "\x3c";
const char *l59 = "\x3b";
const char *l58 = "\x3a";
const char *l57 = "\x39";
const char *l56 = "\x38";
const char *l55 = "\x37";
const char *l54 = "\x36";
const char *l53 = "\x35";
const char *l52 = "\x34";
const char *l51 = "\x33";
const char *l50 = "\x32";
const char *l49 = "\x31";
const char *l48 = "\x30";
const char *l47 = "\x2f";
const char *l46 = "\x2e";
const char *l45 = "\x2d";
const char *l44 = "\x2c";
const char *l43 = "\x2b";
const char *l42 = "\x2a";
const char *l41 = "\x29";
const char *l40 = "\x28";
const char *l39 = "\x27";
const char *l38 = "\x26";
const char *l37 = "\x25";
const char *l36 = "\x24";
const char *l35 = "\x23";
const char *l34 = "\x22";
const char *l33 = "\x21";
const char *l32 = "\x20";
const char *l31 = "\x1f";
const char *l30 = "\x1e";
const char *l29 = "\x1d";
const char *l28 = "\x1c";
const char *l27 = "\x1b";
const char *l26 = "\x1a";
const char *l25 = "\x19";
const char *l24 = "\x18";
const char *l23 = "\x17";
const char *l22 = "\x16";
const char *l21 = "\x15";
const char *l20 = "\x14";
const char *l19 = "\x13";
const char *l18 = "\x12";
const char *l17 = "\x11";
const char *l16 = "\x10";
const char *l15 = "\xf";
const char *l14 = "\xe";
const char *l13 = "\xd";
const char *l12 = "\xc";
const char *l11 = "\xb";
const char *l10 = "\xa";
const char *l9 = "\x9";
const char *l8 = "\x8";
const char *l7 = "\x7";
const char *l6 = "\x6";
const char *l5 = "\x5";
const char *l4 = "\x4";
const char *l3 = "\x3";
const char *l2 = "\x2";
const char *l1 = "\x1";
const char *l0 = "\x0";

// CHECK: @"\01??_C@_01CNACBAHC@?$PP?$AA@"
// CHECK: @"\01??_C@_01DEBJCBDD@?$PO?$AA@"
// CHECK: @"\01??_C@_01BPDEHCPA@?$PN?$AA@"
// CHECK: @"\01??_C@_01GCPEDLB@?$PM?$AA@"
// CHECK: @"\01??_C@_01EJGONFHG@?$PL?$AA@"
// CHECK: @"\01??_C@_01FAHFOEDH@?z?$AA@"
// CHECK: @"\01??_C@_01HLFILHPE@?y?$AA@"
// CHECK: @"\01??_C@_01GCEDIGLF@?x?$AA@"
// CHECK: @"\01??_C@_01OFNLJKHK@?w?$AA@"
// CHECK: @"\01??_C@_01PMMAKLDL@?v?$AA@"
// CHECK: @"\01??_C@_01NHONPIPI@?u?$AA@"
// CHECK: @"\01??_C@_01MOPGMJLJ@?t?$AA@"
// CHECK: @"\01??_C@_01IBLHFPHO@?s?$AA@"
// CHECK: @"\01??_C@_01JIKMGODP@?r?$AA@"
// CHECK: @"\01??_C@_01LDIBDNPM@?q?$AA@"
// CHECK: @"\01??_C@_01KKJKAMLN@?p?$AA@"
// CHECK: @"\01??_C@_01GHMAACCD@?o?$AA@"
// CHECK: @"\01??_C@_01HONLDDGC@?n?$AA@"
// CHECK: @"\01??_C@_01FFPGGAKB@?m?$AA@"
// CHECK: @"\01??_C@_01EMONFBOA@?l?$AA@"
// CHECK: @"\01??_C@_01DKMMHCH@?k?$AA@"
// CHECK: @"\01??_C@_01BKLHPGGG@?j?$AA@"
// CHECK: @"\01??_C@_01DBJKKFKF@?i?$AA@"
// CHECK: @"\01??_C@_01CIIBJEOE@?h?$AA@"
// CHECK: @"\01??_C@_01KPBJIICL@?g?$AA@"
// CHECK: @"\01??_C@_01LGACLJGK@?f?$AA@"
// CHECK: @"\01??_C@_01JNCPOKKJ@?e?$AA@"
// CHECK: @"\01??_C@_01IEDENLOI@?d?$AA@"
// CHECK: @"\01??_C@_01MLHFENCP@?c?$AA@"
// CHECK: @"\01??_C@_01NCGOHMGO@?b?$AA@"
// CHECK: @"\01??_C@_01PJEDCPKN@?a?$AA@"
// CHECK: @"\01??_C@_01OAFIBOOM@?$OA?$AA@"
// CHECK: @"\01??_C@_01LIIGDENA@?$NP?$AA@"
// CHECK: @"\01??_C@_01KBJNAFJB@?$NO?$AA@"
// CHECK: @"\01??_C@_01IKLAFGFC@?$NN?$AA@"
// CHECK: @"\01??_C@_01JDKLGHBD@?$NM?$AA@"
// CHECK: @"\01??_C@_01NMOKPBNE@?$NL?$AA@"
// CHECK: @"\01??_C@_01MFPBMAJF@?Z?$AA@"
// CHECK: @"\01??_C@_01OONMJDFG@?Y?$AA@"
// CHECK: @"\01??_C@_01PHMHKCBH@?X?$AA@"
// CHECK: @"\01??_C@_01HAFPLONI@?W?$AA@"
// CHECK: @"\01??_C@_01GJEEIPJJ@?V?$AA@"
// CHECK: @"\01??_C@_01ECGJNMFK@?U?$AA@"
// CHECK: @"\01??_C@_01FLHCONBL@?T?$AA@"
// CHECK: @"\01??_C@_01BEDDHLNM@?S?$AA@"
// CHECK: @"\01??_C@_01NCIEKJN@?R?$AA@"
// CHECK: @"\01??_C@_01CGAFBJFO@?Q?$AA@"
// CHECK: @"\01??_C@_01DPBOCIBP@?P?$AA@"
// CHECK: @"\01??_C@_01PCEECGIB@?O?$AA@"
// CHECK: @"\01??_C@_01OLFPBHMA@?N?$AA@"
// CHECK: @"\01??_C@_01MAHCEEAD@?M?$AA@"
// CHECK: @"\01??_C@_01NJGJHFEC@?L?$AA@"
// CHECK: @"\01??_C@_01JGCIODIF@?K?$AA@"
// CHECK: @"\01??_C@_01IPDDNCME@?J?$AA@"
// CHECK: @"\01??_C@_01KEBOIBAH@?I?$AA@"
// CHECK: @"\01??_C@_01LNAFLAEG@?H?$AA@"
// CHECK: @"\01??_C@_01DKJNKMIJ@?G?$AA@"
// CHECK: @"\01??_C@_01CDIGJNMI@?F?$AA@"
// CHECK: @"\01??_C@_01IKLMOAL@?E?$AA@"
// CHECK: @"\01??_C@_01BBLAPPEK@?D?$AA@"
// CHECK: @"\01??_C@_01FOPBGJIN@?C?$AA@"
// CHECK: @"\01??_C@_01EHOKFIMM@?B?$AA@"
// CHECK: @"\01??_C@_01GMMHALAP@?A?$AA@"
// CHECK: @"\01??_C@_01HFNMDKEO@?$MA?$AA@"
// CHECK: @"\01??_C@_01NNHLFPHH@?$LP?$AA@"
// CHECK: @"\01??_C@_01MEGAGODG@?$LO?$AA@"
// CHECK: @"\01??_C@_01OPENDNPF@?$LN?$AA@"
// CHECK: @"\01??_C@_01PGFGAMLE@?$LM?$AA@"
// CHECK: @"\01??_C@_01LJBHJKHD@?$LL?$AA@"
// CHECK: @"\01??_C@_01KAAMKLDC@?$LK?$AA@"
// CHECK: @"\01??_C@_01ILCBPIPB@?$LJ?$AA@"
// CHECK: @"\01??_C@_01JCDKMJLA@?$LI?$AA@"
// CHECK: @"\01??_C@_01BFKCNFHP@?$LH?$AA@"
// CHECK: @"\01??_C@_01MLJOEDO@?$LG?$AA@"
// CHECK: @"\01??_C@_01CHJELHPN@?$LF?$AA@"
// CHECK: @"\01??_C@_01DOIPIGLM@?$LE?$AA@"
// CHECK: @"\01??_C@_01HBMOBAHL@?$LD?$AA@"
// CHECK: @"\01??_C@_01GINFCBDK@?$LC?$AA@"
// CHECK: @"\01??_C@_01EDPIHCPJ@?$LB?$AA@"
// CHECK: @"\01??_C@_01FKODEDLI@?$LA?$AA@"
// CHECK: @"\01??_C@_01JHLJENCG@?$KP?$AA@"
// CHECK: @"\01??_C@_01IOKCHMGH@?$KO?$AA@"
// CHECK: @"\01??_C@_01KFIPCPKE@?$KN?$AA@"
// CHECK: @"\01??_C@_01LMJEBOOF@?$KM?$AA@"
// CHECK: @"\01??_C@_01PDNFIICC@?$KL?$AA@"
// CHECK: @"\01??_C@_01OKMOLJGD@?$KK?$AA@"
// CHECK: @"\01??_C@_01MBODOKKA@?$KJ?$AA@"
// CHECK: @"\01??_C@_01NIPINLOB@?$KI?$AA@"
// CHECK: @"\01??_C@_01FPGAMHCO@?$KH?$AA@"
// CHECK: @"\01??_C@_01EGHLPGGP@?$KG?$AA@"
// CHECK: @"\01??_C@_01GNFGKFKM@?$KF?$AA@"
// CHECK: @"\01??_C@_01HEENJEON@?$KE?$AA@"
// CHECK: @"\01??_C@_01DLAMACCK@?$KD?$AA@"
// CHECK: @"\01??_C@_01CCBHDDGL@?$KC?$AA@"
// CHECK: @"\01??_C@_01JDKGAKI@?$KB?$AA@"
// CHECK: @"\01??_C@_01BACBFBOJ@?$KA?$AA@"
// CHECK: @"\01??_C@_01EIPPHLNF@?$JP?$AA@"
// CHECK: @"\01??_C@_01FBOEEKJE@?$JO?$AA@"
// CHECK: @"\01??_C@_01HKMJBJFH@?$JN?$AA@"
// CHECK: @"\01??_C@_01GDNCCIBG@?$JM?$AA@"
// CHECK: @"\01??_C@_01CMJDLONB@?$JL?$AA@"
// CHECK: @"\01??_C@_01DFIIIPJA@?$JK?$AA@"
// CHECK: @"\01??_C@_01BOKFNMFD@?$JJ?$AA@"
// CHECK: @"\01??_C@_01HLOONBC@?$JI?$AA@"
// CHECK: @"\01??_C@_01IACGPBNN@?$JH?$AA@"
// CHECK: @"\01??_C@_01JJDNMAJM@?$JG?$AA@"
// CHECK: @"\01??_C@_01LCBAJDFP@?$JF?$AA@"
// CHECK: @"\01??_C@_01KLALKCBO@?$JE?$AA@"
// CHECK: @"\01??_C@_01OEEKDENJ@?$JD?$AA@"
// CHECK: @"\01??_C@_01PNFBAFJI@?$JC?$AA@"
// CHECK: @"\01??_C@_01NGHMFGFL@?$JB?$AA@"
// CHECK: @"\01??_C@_01MPGHGHBK@?$JA?$AA@"
// CHECK: @"\01??_C@_01CDNGJIE@?$IP?$AA@"
// CHECK: @"\01??_C@_01BLCGFIMF@?$IO?$AA@"
// CHECK: @"\01??_C@_01DAALALAG@?$IN?$AA@"
// CHECK: @"\01??_C@_01CJBADKEH@?$IM?$AA@"
// CHECK: @"\01??_C@_01GGFBKMIA@?$IL?$AA@"
// CHECK: @"\01??_C@_01HPEKJNMB@?$IK?$AA@"
// CHECK: @"\01??_C@_01FEGHMOAC@?$IJ?$AA@"
// CHECK: @"\01??_C@_01ENHMPPED@?$II?$AA@"
// CHECK: @"\01??_C@_01MKOEODIM@?$IH?$AA@"
// CHECK: @"\01??_C@_01NDPPNCMN@?$IG?$AA@"
// CHECK: @"\01??_C@_01PINCIBAO@?$IF?$AA@"
// CHECK: @"\01??_C@_01OBMJLAEP@?$IE?$AA@"
// CHECK: @"\01??_C@_01KOIICGII@?$ID?$AA@"
// CHECK: @"\01??_C@_01LHJDBHMJ@?$IC?$AA@"
// CHECK: @"\01??_C@_01JMLOEEAK@?$IB?$AA@"
// CHECK: @"\01??_C@_01IFKFHFEL@?$IA?$AA@"
// CHECK: @"\01??_C@_01BGIBIIDJ@?$HP?$AA@"
// CHECK: @"\01??_C@_01PJKLJHI@?$HO?$AA@"
// CHECK: @"\01??_C@_01CELHOKLL@?$HN?$AA@"
// CHECK: @"\01??_C@_01DNKMNLPK@?$HM?$AA@"
// CHECK: @"\01??_C@_01HCONENDN@?$HL?$AA@"
// CHECK: @"\01??_C@_01GLPGHMHM@z?$AA@"
// CHECK: @"\01??_C@_01EANLCPLP@y?$AA@"
// CHECK: @"\01??_C@_01FJMABOPO@x?$AA@"
// CHECK: @"\01??_C@_01NOFIACDB@w?$AA@"
// CHECK: @"\01??_C@_01MHEDDDHA@v?$AA@"
// CHECK: @"\01??_C@_01OMGOGALD@u?$AA@"
// CHECK: @"\01??_C@_01PFHFFBPC@t?$AA@"
// CHECK: @"\01??_C@_01LKDEMHDF@s?$AA@"
// CHECK: @"\01??_C@_01KDCPPGHE@r?$AA@"
// CHECK: @"\01??_C@_01IIACKFLH@q?$AA@"
// CHECK: @"\01??_C@_01JBBJJEPG@p?$AA@"
// CHECK: @"\01??_C@_01FMEDJKGI@o?$AA@"
// CHECK: @"\01??_C@_01EFFIKLCJ@n?$AA@"
// CHECK: @"\01??_C@_01GOHFPIOK@m?$AA@"
// CHECK: @"\01??_C@_01HHGOMJKL@l?$AA@"
// CHECK: @"\01??_C@_01DICPFPGM@k?$AA@"
// CHECK: @"\01??_C@_01CBDEGOCN@j?$AA@"
// CHECK: @"\01??_C@_01KBJDNOO@i?$AA@"
// CHECK: @"\01??_C@_01BDACAMKP@h?$AA@"
// CHECK: @"\01??_C@_01JEJKBAGA@g?$AA@"
// CHECK: @"\01??_C@_01INIBCBCB@f?$AA@"
// CHECK: @"\01??_C@_01KGKMHCOC@e?$AA@"
// CHECK: @"\01??_C@_01LPLHEDKD@d?$AA@"
// CHECK: @"\01??_C@_01PAPGNFGE@c?$AA@"
// CHECK: @"\01??_C@_01OJONOECF@b?$AA@"
// CHECK: @"\01??_C@_01MCMALHOG@a?$AA@"
// CHECK: @"\01??_C@_01NLNLIGKH@?$GA?$AA@"
// CHECK: @"\01??_C@_01IDAFKMJL@_?$AA@"
// CHECK: @"\01??_C@_01JKBOJNNK@?$FO?$AA@"
// CHECK: @"\01??_C@_01LBDDMOBJ@?$FN?$AA@"
// CHECK: @"\01??_C@_01KICIPPFI@?2?$AA@"
// CHECK: @"\01??_C@_01OHGJGJJP@?$FL?$AA@"
// CHECK: @"\01??_C@_01POHCFINO@Z?$AA@"
// CHECK: @"\01??_C@_01NFFPALBN@Y?$AA@"
// CHECK: @"\01??_C@_01MMEEDKFM@X?$AA@"
// CHECK: @"\01??_C@_01ELNMCGJD@W?$AA@"
// CHECK: @"\01??_C@_01FCMHBHNC@V?$AA@"
// CHECK: @"\01??_C@_01HJOKEEBB@U?$AA@"
// CHECK: @"\01??_C@_01GAPBHFFA@T?$AA@"
// CHECK: @"\01??_C@_01CPLAODJH@S?$AA@"
// CHECK: @"\01??_C@_01DGKLNCNG@R?$AA@"
// CHECK: @"\01??_C@_01BNIGIBBF@Q?$AA@"
// CHECK: @"\01??_C@_01EJNLAFE@P?$AA@"
// CHECK: @"\01??_C@_01MJMHLOMK@O?$AA@"
// CHECK: @"\01??_C@_01NANMIPIL@N?$AA@"
// CHECK: @"\01??_C@_01PLPBNMEI@M?$AA@"
// CHECK: @"\01??_C@_01OCOKONAJ@L?$AA@"
// CHECK: @"\01??_C@_01KNKLHLMO@K?$AA@"
// CHECK: @"\01??_C@_01LELAEKIP@J?$AA@"
// CHECK: @"\01??_C@_01JPJNBJEM@I?$AA@"
// CHECK: @"\01??_C@_01IGIGCIAN@H?$AA@"
// CHECK: @"\01??_C@_01BBODEMC@G?$AA@"
// CHECK: @"\01??_C@_01BIAFAFID@F?$AA@"
// CHECK: @"\01??_C@_01DDCIFGEA@E?$AA@"
// CHECK: @"\01??_C@_01CKDDGHAB@D?$AA@"
// CHECK: @"\01??_C@_01GFHCPBMG@C?$AA@"
// CHECK: @"\01??_C@_01HMGJMAIH@B?$AA@"
// CHECK: @"\01??_C@_01FHEEJDEE@A?$AA@"
// CHECK: @"\01??_C@_01EOFPKCAF@?$EA?$AA@"
// CHECK: @"\01??_C@_01OGPIMHDM@?$DP?$AA@"
// CHECK: @"\01??_C@_01PPODPGHN@?$DO?$AA@"
// CHECK: @"\01??_C@_01NEMOKFLO@?$DN?$AA@"
// CHECK: @"\01??_C@_01MNNFJEPP@?$DM?$AA@"
// CHECK: @"\01??_C@_01ICJEACDI@?$DL?$AA@"
// CHECK: @"\01??_C@_01JLIPDDHJ@?3?$AA@"
// CHECK: @"\01??_C@_01LAKCGALK@9?$AA@"
// CHECK: @"\01??_C@_01KJLJFBPL@8?$AA@"
// CHECK: @"\01??_C@_01COCBENDE@7?$AA@"
// CHECK: @"\01??_C@_01DHDKHMHF@6?$AA@"
// CHECK: @"\01??_C@_01BMBHCPLG@5?$AA@"
// CHECK: @"\01??_C@_01FAMBOPH@4?$AA@"
// CHECK: @"\01??_C@_01EKENIIDA@3?$AA@"
// CHECK: @"\01??_C@_01FDFGLJHB@2?$AA@"
// CHECK: @"\01??_C@_01HIHLOKLC@1?$AA@"
// CHECK: @"\01??_C@_01GBGANLPD@0?$AA@"
// CHECK: @"\01??_C@_01KMDKNFGN@?1?$AA@"
// CHECK: @"\01??_C@_01LFCBOECM@?4?$AA@"
// CHECK: @"\01??_C@_01JOAMLHOP@?9?$AA@"
// CHECK: @"\01??_C@_01IHBHIGKO@?0?$AA@"
// CHECK: @"\01??_C@_01MIFGBAGJ@?$CL?$AA@"
// CHECK: @"\01??_C@_01NBENCBCI@?$CK?$AA@"
// CHECK: @"\01??_C@_01PKGAHCOL@?$CJ?$AA@"
// CHECK: @"\01??_C@_01ODHLEDKK@?$CI?$AA@"
// CHECK: @"\01??_C@_01GEODFPGF@?8?$AA@"
// CHECK: @"\01??_C@_01HNPIGOCE@?$CG?$AA@"
// CHECK: @"\01??_C@_01FGNFDNOH@?$CF?$AA@"
// CHECK: @"\01??_C@_01EPMOAMKG@$?$AA@"
// CHECK: @"\01??_C@_01IPJKGB@?$CD?$AA@"
// CHECK: @"\01??_C@_01BJJEKLCA@?$CC?$AA@"
// CHECK: @"\01??_C@_01DCLJPIOD@?$CB?$AA@"
// CHECK: @"\01??_C@_01CLKCMJKC@?5?$AA@"
// CHECK: @"\01??_C@_01HDHMODJO@?$BP?$AA@"
// CHECK: @"\01??_C@_01GKGHNCNP@?$BO?$AA@"
// CHECK: @"\01??_C@_01EBEKIBBM@?$BN?$AA@"
// CHECK: @"\01??_C@_01FIFBLAFN@?$BM?$AA@"
// CHECK: @"\01??_C@_01BHBACGJK@?$BL?$AA@"
// CHECK: @"\01??_C@_01OALBHNL@?$BK?$AA@"
// CHECK: @"\01??_C@_01CFCGEEBI@?$BJ?$AA@"
// CHECK: @"\01??_C@_01DMDNHFFJ@?$BI?$AA@"
// CHECK: @"\01??_C@_01LLKFGJJG@?$BH?$AA@"
// CHECK: @"\01??_C@_01KCLOFINH@?$BG?$AA@"
// CHECK: @"\01??_C@_01IJJDALBE@?$BF?$AA@"
// CHECK: @"\01??_C@_01JAIIDKFF@?$BE?$AA@"
// CHECK: @"\01??_C@_01NPMJKMJC@?$BD?$AA@"
// CHECK: @"\01??_C@_01MGNCJNND@?$BC?$AA@"
// CHECK: @"\01??_C@_01ONPPMOBA@?$BB?$AA@"
// CHECK: @"\01??_C@_01PEOEPPFB@?$BA?$AA@"
// CHECK: @"\01??_C@_01DJLOPBMP@?$AP?$AA@"
// CHECK: @"\01??_C@_01CAKFMAIO@?$AO?$AA@"
// CHECK: @"\01??_C@_01LIIJDEN@?$AN?$AA@"
// CHECK: @"\01??_C@_01BCJDKCAM@?$AM?$AA@"
// CHECK: @"\01??_C@_01FNNCDEML@?$AL?$AA@"
// CHECK: @"\01??_C@_01EEMJAFIK@?6?$AA@"
// CHECK: @"\01??_C@_01GPOEFGEJ@?7?$AA@"
// CHECK: @"\01??_C@_01HGPPGHAI@?$AI?$AA@"
// CHECK: @"\01??_C@_01PBGHHLMH@?$AH?$AA@"
// CHECK: @"\01??_C@_01OIHMEKIG@?$AG?$AA@"
// CHECK: @"\01??_C@_01MDFBBJEF@?$AF?$AA@"
// CHECK: @"\01??_C@_01NKEKCIAE@?$AE?$AA@"
// CHECK: @"\01??_C@_01JFALLOMD@?$AD?$AA@"
// CHECK: @"\01??_C@_01IMBAIPIC@?$AC?$AA@"
// CHECK: @"\01??_C@_01KHDNNMEB@?$AB?$AA@"
// CHECK: @"\01??_C@_01LOCGONAA@?$AA?$AA@"

const wchar_t *wl9 = L"\t";
const wchar_t *wl10 = L"\n";
const wchar_t *wl11 = L"\v";
const wchar_t *wl32 = L" ";
const wchar_t *wl33 = L"!";
const wchar_t *wl34 = L"\"";
const wchar_t *wl35 = L"#";
const wchar_t *wl36 = L"$";
const wchar_t *wl37 = L"%";
const wchar_t *wl38 = L"&";
const wchar_t *wl39 = L"'";
const wchar_t *wl40 = L"(";
const wchar_t *wl41 = L")";
const wchar_t *wl42 = L"*";
const wchar_t *wl43 = L"+";
const wchar_t *wl44 = L",";
const wchar_t *wl45 = L"-";
const wchar_t *wl46 = L".";
const wchar_t *wl47 = L"/";
const wchar_t *wl48 = L"0";
const wchar_t *wl49 = L"1";
const wchar_t *wl50 = L"2";
const wchar_t *wl51 = L"3";
const wchar_t *wl52 = L"4";
const wchar_t *wl53 = L"5";
const wchar_t *wl54 = L"6";
const wchar_t *wl55 = L"7";
const wchar_t *wl56 = L"8";
const wchar_t *wl57 = L"9";
const wchar_t *wl58 = L":";
const wchar_t *wl59 = L";";
const wchar_t *wl60 = L"<";
const wchar_t *wl61 = L"=";
const wchar_t *wl62 = L">";
const wchar_t *wl63 = L"?";
const wchar_t *wl64 = L"@";
const wchar_t *wl65 = L"A";
const wchar_t *wl66 = L"B";
const wchar_t *wl67 = L"C";
const wchar_t *wl68 = L"D";
const wchar_t *wl69 = L"E";
const wchar_t *wl70 = L"F";
const wchar_t *wl71 = L"G";
const wchar_t *wl72 = L"H";
const wchar_t *wl73 = L"I";
const wchar_t *wl74 = L"J";
const wchar_t *wl75 = L"K";
const wchar_t *wl76 = L"L";
const wchar_t *wl77 = L"M";
const wchar_t *wl78 = L"N";
const wchar_t *wl79 = L"O";
const wchar_t *wl80 = L"P";
const wchar_t *wl81 = L"Q";
const wchar_t *wl82 = L"R";
const wchar_t *wl83 = L"S";
const wchar_t *wl84 = L"T";
const wchar_t *wl85 = L"U";
const wchar_t *wl86 = L"V";
const wchar_t *wl87 = L"W";
const wchar_t *wl88 = L"X";
const wchar_t *wl89 = L"Y";
const wchar_t *wl90 = L"Z";
const wchar_t *wl91 = L"[";
const wchar_t *wl92 = L"\\";
const wchar_t *wl93 = L"]";
const wchar_t *wl94 = L"^";
const wchar_t *wl95 = L"_";
const wchar_t *wl96 = L"`";
const wchar_t *wl97 = L"a";
const wchar_t *wl98 = L"b";
const wchar_t *wl99 = L"c";
const wchar_t *wl100 = L"d";
const wchar_t *wl101 = L"e";
const wchar_t *wl102 = L"f";
const wchar_t *wl103 = L"g";
const wchar_t *wl104 = L"h";
const wchar_t *wl105 = L"i";
const wchar_t *wl106 = L"j";
const wchar_t *wl107 = L"k";
const wchar_t *wl108 = L"l";
const wchar_t *wl109 = L"m";
const wchar_t *wl110 = L"n";
const wchar_t *wl111 = L"o";
const wchar_t *wl112 = L"p";
const wchar_t *wl113 = L"q";
const wchar_t *wl114 = L"r";
const wchar_t *wl115 = L"s";
const wchar_t *wl116 = L"t";
const wchar_t *wl117 = L"u";
const wchar_t *wl118 = L"v";
const wchar_t *wl119 = L"w";
const wchar_t *wl120 = L"x";
const wchar_t *wl121 = L"y";
const wchar_t *wl122 = L"z";
const wchar_t *wl123 = L"{";
const wchar_t *wl124 = L"|";
const wchar_t *wl125 = L"}";
const wchar_t *wl126 = L"~";

// CHECK: @"\01??_C@_13KDLDGPGJ@?$AA?7?$AA?$AA@"
// CHECK: @"\01??_C@_13LBAGMAIH@?$AA?6?$AA?$AA@"
// CHECK: @"\01??_C@_13JLKKHOC@?$AA?$AL?$AA?$AA@"
// CHECK: @"\01??_C@_13HOIJIPNN@?$AA?5?$AA?$AA@"
// CHECK: @"\01??_C@_13MGDFOILI@?$AA?$CB?$AA?$AA@"
// CHECK: @"\01??_C@_13NEIAEHFG@?$AA?$CC?$AA?$AA@"
// CHECK: @"\01??_C@_13GMDMCADD@?$AA?$CD?$AA?$AA@"
// CHECK: @"\01??_C@_13PBOLBIIK@?$AA$?$AA?$AA@"
// CHECK: @"\01??_C@_13EJFHHPOP@?$AA?$CF?$AA?$AA@"
// CHECK: @"\01??_C@_13FLOCNAAB@?$AA?$CG?$AA?$AA@"
// CHECK: @"\01??_C@_13ODFOLHGE@?$AA?8?$AA?$AA@"
// CHECK: @"\01??_C@_13LLDNKHDC@?$AA?$CI?$AA?$AA@"
// CHECK: @"\01??_C@_13DIBMAFH@?$AA?$CJ?$AA?$AA@"
// CHECK: @"\01??_C@_13BBDEGPLJ@?$AA?$CK?$AA?$AA@"
// CHECK: @"\01??_C@_13KJIIAINM@?$AA?$CL?$AA?$AA@"
// CHECK: @"\01??_C@_13DEFPDAGF@?$AA?0?$AA?$AA@"
// CHECK: @"\01??_C@_13IMODFHAA@?$AA?9?$AA?$AA@"
// CHECK: @"\01??_C@_13JOFGPIOO@?$AA?4?$AA?$AA@"
// CHECK: @"\01??_C@_13CGOKJPIL@?$AA?1?$AA?$AA@"
// CHECK: @"\01??_C@_13COJANIEC@?$AA0?$AA?$AA@"
// CHECK: @"\01??_C@_13JGCMLPCH@?$AA1?$AA?$AA@"
// CHECK: @"\01??_C@_13IEJJBAMJ@?$AA2?$AA?$AA@"
// CHECK: @"\01??_C@_13DMCFHHKM@?$AA3?$AA?$AA@"
// CHECK: @"\01??_C@_13KBPCEPBF@?$AA4?$AA?$AA@"
// CHECK: @"\01??_C@_13BJEOCIHA@?$AA5?$AA?$AA@"
// CHECK: @"\01??_C@_13LPLIHJO@?$AA6?$AA?$AA@"
// CHECK: @"\01??_C@_13LDEHOAPL@?$AA7?$AA?$AA@"
// CHECK: @"\01??_C@_13OLCEPAKN@?$AA8?$AA?$AA@"
// CHECK: @"\01??_C@_13FDJIJHMI@?$AA9?$AA?$AA@"
// CHECK: @"\01??_C@_13EBCNDICG@?$AA?3?$AA?$AA@"
// CHECK: @"\01??_C@_13PJJBFPED@?$AA?$DL?$AA?$AA@"
// CHECK: @"\01??_C@_13GEEGGHPK@?$AA?$DM?$AA?$AA@"
// CHECK: @"\01??_C@_13NMPKAAJP@?$AA?$DN?$AA?$AA@"
// CHECK: @"\01??_C@_13MOEPKPHB@?$AA?$DO?$AA?$AA@"
// CHECK: @"\01??_C@_13HGPDMIBE@?$AA?$DP?$AA?$AA@"
// CHECK: @"\01??_C@_13EFKPHINO@?$AA?$EA?$AA?$AA@"
// CHECK: @"\01??_C@_13PNBDBPLL@?$AAA?$AA?$AA@"
// CHECK: @"\01??_C@_13OPKGLAFF@?$AAB?$AA?$AA@"
// CHECK: @"\01??_C@_13FHBKNHDA@?$AAC?$AA?$AA@"
// CHECK: @"\01??_C@_13MKMNOPIJ@?$AAD?$AA?$AA@"
// CHECK: @"\01??_C@_13HCHBIIOM@?$AAE?$AA?$AA@"
// CHECK: @"\01??_C@_13GAMECHAC@?$AAF?$AA?$AA@"
// CHECK: @"\01??_C@_13NIHIEAGH@?$AAG?$AA?$AA@"
// CHECK: @"\01??_C@_13IABLFADB@?$AAH?$AA?$AA@"
// CHECK: @"\01??_C@_13DIKHDHFE@?$AAI?$AA?$AA@"
// CHECK: @"\01??_C@_13CKBCJILK@?$AAJ?$AA?$AA@"
// CHECK: @"\01??_C@_13JCKOPPNP@?$AAK?$AA?$AA@"
// CHECK: @"\01??_C@_13PHJMHGG@?$AAL?$AA?$AA@"
// CHECK: @"\01??_C@_13LHMFKAAD@?$AAM?$AA?$AA@"
// CHECK: @"\01??_C@_13KFHAAPON@?$AAN?$AA?$AA@"
// CHECK: @"\01??_C@_13BNMMGIII@?$AAO?$AA?$AA@"
// CHECK: @"\01??_C@_13BFLGCPEB@?$AAP?$AA?$AA@"
// CHECK: @"\01??_C@_13KNAKEICE@?$AAQ?$AA?$AA@"
// CHECK: @"\01??_C@_13LPLPOHMK@?$AAR?$AA?$AA@"
// CHECK: @"\01??_C@_13HADIAKP@?$AAS?$AA?$AA@"
// CHECK: @"\01??_C@_13JKNELIBG@?$AAT?$AA?$AA@"
// CHECK: @"\01??_C@_13CCGINPHD@?$AAU?$AA?$AA@"
// CHECK: @"\01??_C@_13DANNHAJN@?$AAV?$AA?$AA@"
// CHECK: @"\01??_C@_13IIGBBHPI@?$AAW?$AA?$AA@"
// CHECK: @"\01??_C@_13NAACAHKO@?$AAX?$AA?$AA@"
// CHECK: @"\01??_C@_13GILOGAML@?$AAY?$AA?$AA@"
// CHECK: @"\01??_C@_13HKALMPCF@?$AAZ?$AA?$AA@"
// CHECK: @"\01??_C@_13MCLHKIEA@?$AA?$FL?$AA?$AA@"
// CHECK: @"\01??_C@_13FPGAJAPJ@?$AA?2?$AA?$AA@"
// CHECK: @"\01??_C@_13OHNMPHJM@?$AA?$FN?$AA?$AA@"
// CHECK: @"\01??_C@_13PFGJFIHC@?$AA?$FO?$AA?$AA@"
// CHECK: @"\01??_C@_13ENNFDPBH@?$AA_?$AA?$AA@"
// CHECK: @"\01??_C@_13OFJNNHOA@?$AA?$GA?$AA?$AA@"
// CHECK: @"\01??_C@_13FNCBLAIF@?$AAa?$AA?$AA@"
// CHECK: @"\01??_C@_13EPJEBPGL@?$AAb?$AA?$AA@"
// CHECK: @"\01??_C@_13PHCIHIAO@?$AAc?$AA?$AA@"
// CHECK: @"\01??_C@_13GKPPEALH@?$AAd?$AA?$AA@"
// CHECK: @"\01??_C@_13NCEDCHNC@?$AAe?$AA?$AA@"
// CHECK: @"\01??_C@_13MAPGIIDM@?$AAf?$AA?$AA@"
// CHECK: @"\01??_C@_13HIEKOPFJ@?$AAg?$AA?$AA@"
// CHECK: @"\01??_C@_13CACJPPAP@?$AAh?$AA?$AA@"
// CHECK: @"\01??_C@_13JIJFJIGK@?$AAi?$AA?$AA@"
// CHECK: @"\01??_C@_13IKCADHIE@?$AAj?$AA?$AA@"
// CHECK: @"\01??_C@_13DCJMFAOB@?$AAk?$AA?$AA@"
// CHECK: @"\01??_C@_13KPELGIFI@?$AAl?$AA?$AA@"
// CHECK: @"\01??_C@_13BHPHAPDN@?$AAm?$AA?$AA@"
// CHECK: @"\01??_C@_13FECKAND@?$AAn?$AA?$AA@"
// CHECK: @"\01??_C@_13LNPOMHLG@?$AAo?$AA?$AA@"
// CHECK: @"\01??_C@_13LFIEIAHP@?$AAp?$AA?$AA@"
// CHECK: @"\01??_C@_13NDIOHBK@?$AAq?$AA?$AA@"
// CHECK: @"\01??_C@_13BPINEIPE@?$AAr?$AA?$AA@"
// CHECK: @"\01??_C@_13KHDBCPJB@?$AAs?$AA?$AA@"
// CHECK: @"\01??_C@_13DKOGBHCI@?$AAt?$AA?$AA@"
// CHECK: @"\01??_C@_13ICFKHAEN@?$AAu?$AA?$AA@"
// CHECK: @"\01??_C@_13JAOPNPKD@?$AAv?$AA?$AA@"
// CHECK: @"\01??_C@_13CIFDLIMG@?$AAw?$AA?$AA@"
// CHECK: @"\01??_C@_13HADAKIJA@?$AAx?$AA?$AA@"
// CHECK: @"\01??_C@_13MIIMMPPF@?$AAy?$AA?$AA@"
// CHECK: @"\01??_C@_13NKDJGABL@?$AAz?$AA?$AA@"
// CHECK: @"\01??_C@_13GCIFAHHO@?$AA?$HL?$AA?$AA@"
// CHECK: @"\01??_C@_13PPFCDPMH@?$AA?$HM?$AA?$AA@"
// CHECK: @"\01??_C@_13EHOOFIKC@?$AA?$HN?$AA?$AA@"
// CHECK: @"\01??_C@_13FFFLPHEM@?$AA?$HO?$AA?$AA@"

const char *LongASCIIString = "012345678901234567890123456789ABCDEF";
// CHECK: @"\01??_C@_0CF@LABBIIMO@012345678901234567890123456789AB@"
const wchar_t *LongWideString = L"012345678901234567890123456789ABCDEF";
// CHECK: @"\01??_C@_1EK@KFPEBLPK@?$AA0?$AA1?$AA2?$AA3?$AA4?$AA5?$AA6?$AA7?$AA8?$AA9?$AA0?$AA1?$AA2?$AA3?$AA4?$AA5?$AA6?$AA7?$AA8?$AA9?$AA0?$AA1?$AA2?$AA3?$AA4?$AA5?$AA6?$AA7?$AA8?$AA9?$AAA?$AAB@"
const wchar_t *UnicodeLiteral = L"\ud7ff";
// CHECK: @"\01??_C@_13IIHIAFKH@?W?$PP?$AA?$AA@"
const char *U8Literal = u8"hi";
// CHECK: @"\01??_C@_02PCEFGMJL@hi?$AA@"
const char16_t *U16Literal = u"hi";
// CHECK: @"\01??_C@_05OMLEGLOC@h?$AAi?$AA?$AA?$AA@"
const char32_t *U32Literal = U"hi";
// CHECK: @"\01??_C@_0M@GFNAJIPG@h?$AA?$AA?$AAi?$AA?$AA?$AA?$AA?$AA?$AA?$AA@"
