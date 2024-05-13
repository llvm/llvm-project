// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef DOUBLE_GENERAL_PRECISION_TO_CHARS_TEST_CASES_HPP
#define DOUBLE_GENERAL_PRECISION_TO_CHARS_TEST_CASES_HPP

#include <charconv>

#include "test.hpp"
using namespace std;

// C11 7.21.6.1 "The fprintf function"/8:

// "Then, if a conversion with style E would have an exponent of X:
// - if P > X >= -4, the conversion is with style f (or F) and precision P - (X + 1).
// - otherwise, the conversion is with style e (or E) and precision P - 1."

// "Finally, [...] any trailing zeros are removed from the fractional portion of the result
// and the decimal-point character is removed if there is no fractional portion remaining."

inline constexpr DoublePrecisionToCharsTestCase double_general_precision_to_chars_test_cases[] = {
    // Test special cases (zero, inf, nan) and an ordinary case. Also test negative signs.
    {0.0, chars_format::general, 4, "0"},
    {-0.0, chars_format::general, 4, "-0"},
    {double_inf, chars_format::general, 4, "inf"},
    {-double_inf, chars_format::general, 4, "-inf"},
    {double_nan, chars_format::general, 4, "nan"},
    {-double_nan, chars_format::general, 4, "-nan(ind)"},
    {double_nan_payload, chars_format::general, 4, "nan"},
    {-double_nan_payload, chars_format::general, 4, "-nan"},
    {1.729, chars_format::general, 4, "1.729"},
    {-1.729, chars_format::general, 4, "-1.729"},

    // Test corner cases.
    {0x0.0000000000001p-1022, chars_format::general, 1000,
        "4."
        "94065645841246544176568792868221372365059802614324764425585682500675507270208751865299836361635992379796564695"
        "44571773092665671035593979639877479601078187812630071319031140452784581716784898210368871863605699873072305000"
        "63874091535649843873124733972731696151400317153853980741262385655911710266585566867681870395603106249319452715"
        "91492455329305456544401127480129709999541931989409080416563324524757147869014726780159355238611550134803526493"
        "47201937902681071074917033322268447533357208324319360923828934583680601060115061698097530783422773183292479049"
        "82524730776375927247874656084778203734469699533647017972677717585125660551199131504891101451037862738167250955"
        "837389733598993664809941164205702637090279242767544565229087538682506419718265533447265625"
        "e-324"}, // min subnormal
    {0x0.fffffffffffffp-1022, chars_format::general, 1000,
        "2."
        "22507385850720088902458687608585988765042311224095946549352480256244000922823569517877588880375915526423097809"
        "50434312085877387158357291821993020294379224223559819827501242041788969571311791082261043971979604000454897391"
        "93807919893608152561311337614984204327175103362739154978273159414382813627511383860409424946494228631669542910"
        "50802018159266421349966065178030950759130587198464239060686371020051087232827846788436319445158661350412234790"
        "14792369585208321597621066375401613736583044193603714778355306682834535634005074073040135602968046375918583163"
        "12422452159926254649430083685186171942241764645513713542013221703137049658321015465406803539741790602258950302"
        "3501937519773030945763173210852507299305089761582519159720757232455434770912461317493580281734466552734375"
        "e-308"}, // max subnormal
    {0x1p-1022, chars_format::general, 1000,
        "2."
        "22507385850720138309023271733240406421921598046233183055332741688720443481391819585428315901251102056406733973"
        "10358110051524341615534601088560123853777188211307779935320023304796101474425836360719215650469425037342083752"
        "50806650616658158948720491179968591639648500635908770118304874799780887753749949451580451605050915399856582470"
        "81864511353793580499211598108576605199243335211435239014879569960959128889160299264151106346631339366347758651"
        "30293717620473256317814856643508721228286376420448468114076139114770628016898532441100241614474216185671661505"
        "40154285084716752901903161322778896729707373123334086988983175067838846926092773977972858659654941091369095406"
        "136467568702398678315290680984617210924625396728515625e-308"}, // min normal
    {0x1.fffffffffffffp+1023, chars_format::general, 1000,
        "17976931348623157081452742373170435679807056752584499659891747680315726078002853876058955863276687817154045895"
        "35143824642343213268894641827684675467035375169860499105765512820762454900903893289440758685084551339423045832"
        "36903222948165808559332123348274797826204144723168738177180919299881250404026184124858368"}, // max normal

    {0x0.0000000000001p-1022, chars_format::general, 6, "4.94066e-324"}, // min subnormal
    {0x0.fffffffffffffp-1022, chars_format::general, 6, "2.22507e-308"}, // max subnormal
    {0x1p-1022, chars_format::general, 6, "2.22507e-308"}, // min normal
    {0x1.fffffffffffffp+1023, chars_format::general, 6, "1.79769e+308"}, // max normal

    // Test maximum-length output (excluding minus signs).
    {0x1.fffffffffffffp-1022, chars_format::general, 1000,
        "4."
        "45014771701440227211481959341826395186963909270329129604685221944964444404215389103305904781627017582829831782"
        "60792422137401728773891892910553144148156412434867599762821265346585071045737627442980259622449029037796981144"
        "44614570510266311510031828794952795966823603998647925096578034214163701381261333311989876551545144031526125381"
        "32666529513060001849177663286607555958373922409899478075565940981010216121988146052587425791790000716759993441"
        "45086087205681577915435923018910334964869420614052182892431445797605163650903606514140377217442262561590244668"
        "52576737244643007551333245007965068671949137768847800530996396770975896584413789443379662199396731693628045708"
        "4866613206797017728916080020698679408551343728867675409720757232455434770912461317493580281734466552734375e-"
        "308"}, // scientific, happens to be the same length as max subnormal
    {0x1.fffffffffffffp-14, chars_format::general, 1000,
        "0.000122070312499999986447472843931194574906839989125728607177734375"}, // fixed

    // Test varying precision. Negative precision requests P == 6. Zero precision requests P == 1.
    // Here, the scientific exponent X is 0.
    // Therefore, fixed notation is always chosen with precision P - (X + 1) == P - 1.
    {0x1.b04p0, chars_format::general, -2, "1.68848"},
    {0x1.b04p0, chars_format::general, -1, "1.68848"},
    {0x1.b04p0, chars_format::general, 0, "2"},
    {0x1.b04p0, chars_format::general, 1, "2"}, // fixed notation trims decimal point
    {0x1.b04p0, chars_format::general, 2, "1.7"},
    {0x1.b04p0, chars_format::general, 3, "1.69"},
    {0x1.b04p0, chars_format::general, 4, "1.688"},
    {0x1.b04p0, chars_format::general, 5, "1.6885"},
    {0x1.b04p0, chars_format::general, 6, "1.68848"},
    {0x1.b04p0, chars_format::general, 7, "1.688477"},
    {0x1.b04p0, chars_format::general, 8, "1.6884766"},
    {0x1.b04p0, chars_format::general, 9, "1.68847656"},
    {0x1.b04p0, chars_format::general, 10, "1.688476562"}, // round to even
    {0x1.b04p0, chars_format::general, 11, "1.6884765625"}, // exact
    {0x1.b04p0, chars_format::general, 12, "1.6884765625"}, // trim trailing zeros
    {0x1.b04p0, chars_format::general, 13, "1.6884765625"},

    // Here, the scientific exponent X is -5.
    // Therefore, scientific notation is always chosen with precision P - 1.
    {0x1.8p-15, chars_format::general, -2, "4.57764e-05"},
    {0x1.8p-15, chars_format::general, -1, "4.57764e-05"},
    {0x1.8p-15, chars_format::general, 0, "5e-05"},
    {0x1.8p-15, chars_format::general, 1, "5e-05"}, // scientific notation trims decimal point
    {0x1.8p-15, chars_format::general, 2, "4.6e-05"},
    {0x1.8p-15, chars_format::general, 3, "4.58e-05"},
    {0x1.8p-15, chars_format::general, 4, "4.578e-05"},
    {0x1.8p-15, chars_format::general, 5, "4.5776e-05"},
    {0x1.8p-15, chars_format::general, 6, "4.57764e-05"},
    {0x1.8p-15, chars_format::general, 7, "4.577637e-05"},
    {0x1.8p-15, chars_format::general, 8, "4.5776367e-05"},
    {0x1.8p-15, chars_format::general, 9, "4.57763672e-05"},
    {0x1.8p-15, chars_format::general, 10, "4.577636719e-05"},
    {0x1.8p-15, chars_format::general, 11, "4.5776367188e-05"}, // round to even
    {0x1.8p-15, chars_format::general, 12, "4.57763671875e-05"}, // exact
    {0x1.8p-15, chars_format::general, 13, "4.57763671875e-05"}, // trim trailing zeros
    {0x1.8p-15, chars_format::general, 14, "4.57763671875e-05"},

    // Trim trailing zeros.
    {0x1.80015p0, chars_format::general, 1, "2"}, // fixed notation trims decimal point
    {0x1.80015p0, chars_format::general, 2, "1.5"},
    {0x1.80015p0, chars_format::general, 3, "1.5"}, // general trims trailing zeros
    {0x1.80015p0, chars_format::general, 4, "1.5"},
    {0x1.80015p0, chars_format::general, 5, "1.5"},
    {0x1.80015p0, chars_format::general, 6, "1.50002"},
    {0x1.80015p0, chars_format::general, 7, "1.50002"},
    {0x1.80015p0, chars_format::general, 8, "1.50002"},
    {0x1.80015p0, chars_format::general, 9, "1.50002003"},
    {0x1.80015p0, chars_format::general, 10, "1.500020027"},
    {0x1.80015p0, chars_format::general, 11, "1.5000200272"},
    {0x1.80015p0, chars_format::general, 12, "1.50002002716"},
    {0x1.80015p0, chars_format::general, 13, "1.500020027161"},
    {0x1.80015p0, chars_format::general, 14, "1.5000200271606"},
    {0x1.80015p0, chars_format::general, 15, "1.50002002716064"},
    {0x1.80015p0, chars_format::general, 16, "1.500020027160645"},
    {0x1.80015p0, chars_format::general, 17, "1.5000200271606445"},
    {0x1.80015p0, chars_format::general, 18, "1.50002002716064453"},
    {0x1.80015p0, chars_format::general, 19, "1.500020027160644531"},
    {0x1.80015p0, chars_format::general, 20, "1.5000200271606445312"}, // round to even
    {0x1.80015p0, chars_format::general, 21, "1.50002002716064453125"}, // exact

    // Trim trailing zeros and decimal point.
    {0x1.00015p0, chars_format::general, 1, "1"}, // fixed notation trims decimal point
    {0x1.00015p0, chars_format::general, 2, "1"}, // general trims decimal point and trailing zeros
    {0x1.00015p0, chars_format::general, 3, "1"},
    {0x1.00015p0, chars_format::general, 4, "1"},
    {0x1.00015p0, chars_format::general, 5, "1"},
    {0x1.00015p0, chars_format::general, 6, "1.00002"},
    {0x1.00015p0, chars_format::general, 7, "1.00002"},
    {0x1.00015p0, chars_format::general, 8, "1.00002"},
    {0x1.00015p0, chars_format::general, 9, "1.00002003"},
    {0x1.00015p0, chars_format::general, 10, "1.000020027"},
    {0x1.00015p0, chars_format::general, 11, "1.0000200272"},
    {0x1.00015p0, chars_format::general, 12, "1.00002002716"},
    {0x1.00015p0, chars_format::general, 13, "1.000020027161"},
    {0x1.00015p0, chars_format::general, 14, "1.0000200271606"},
    {0x1.00015p0, chars_format::general, 15, "1.00002002716064"},
    {0x1.00015p0, chars_format::general, 16, "1.000020027160645"},
    {0x1.00015p0, chars_format::general, 17, "1.0000200271606445"},
    {0x1.00015p0, chars_format::general, 18, "1.00002002716064453"},
    {0x1.00015p0, chars_format::general, 19, "1.000020027160644531"},
    {0x1.00015p0, chars_format::general, 20, "1.0000200271606445312"}, // round to even
    {0x1.00015p0, chars_format::general, 21, "1.00002002716064453125"}, // exact

    // Trim trailing zeros, scientific notation.
    {0x1.5cf751db94e6bp-20, chars_format::general, 1, "1e-06"}, // scientific notation trims decimal point
    {0x1.5cf751db94e6bp-20, chars_format::general, 2, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 3, "1.3e-06"}, // general trims trailing zeros
    {0x1.5cf751db94e6bp-20, chars_format::general, 4, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 5, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 6, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 7, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 8, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 9, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 10, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 11, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 12, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 13, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 14, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 15, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 16, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 17, "1.3e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 18, "1.30000000000000005e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 19, "1.300000000000000047e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 20, "1.3000000000000000471e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 21, "1.30000000000000004705e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 22, "1.300000000000000047052e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 23, "1.3000000000000000470517e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 24, "1.30000000000000004705166e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 25, "1.300000000000000047051664e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 26, "1.3000000000000000470516638e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 27, "1.30000000000000004705166378e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 28, "1.30000000000000004705166378e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 29, "1.3000000000000000470516637804e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 30, "1.30000000000000004705166378044e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 31, "1.30000000000000004705166378044e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 32, "1.3000000000000000470516637804397e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 33, "1.30000000000000004705166378043968e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 34, "1.300000000000000047051663780439679e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 35, "1.3000000000000000470516637804396787e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 36, "1.30000000000000004705166378043967867e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 37, "1.300000000000000047051663780439678675e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 38, "1.3000000000000000470516637804396786748e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 39, "1.30000000000000004705166378043967867484e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 40, "1.300000000000000047051663780439678674838e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 41, "1.3000000000000000470516637804396786748384e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 42, "1.30000000000000004705166378043967867483843e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 43, "1.300000000000000047051663780439678674838433e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 44, "1.3000000000000000470516637804396786748384329e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 45, "1.30000000000000004705166378043967867483843293e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 46, "1.300000000000000047051663780439678674838432926e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 47, "1.3000000000000000470516637804396786748384329258e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 48, "1.30000000000000004705166378043967867483843292575e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 49, "1.300000000000000047051663780439678674838432925753e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 50, "1.3000000000000000470516637804396786748384329257533e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 51, "1.3000000000000000470516637804396786748384329257533e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 52, "1.300000000000000047051663780439678674838432925753295e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 53, "1.3000000000000000470516637804396786748384329257532954e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 54, "1.30000000000000004705166378043967867483843292575329542e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 55, "1.300000000000000047051663780439678674838432925753295422e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 56, "1.3000000000000000470516637804396786748384329257532954216e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 57, "1.3000000000000000470516637804396786748384329257532954216e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 58, "1.3000000000000000470516637804396786748384329257532954216e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 59,
        "1.3000000000000000470516637804396786748384329257532954216003e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 60,
        "1.30000000000000004705166378043967867483843292575329542160034e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 61,
        "1.300000000000000047051663780439678674838432925753295421600342e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 62,
        "1.3000000000000000470516637804396786748384329257532954216003418e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 63,
        "1.3000000000000000470516637804396786748384329257532954216003418e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 64,
        "1.300000000000000047051663780439678674838432925753295421600341797e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 65,
        "1.3000000000000000470516637804396786748384329257532954216003417969e-06"},
    {0x1.5cf751db94e6bp-20, chars_format::general, 66,
        "1.30000000000000004705166378043967867483843292575329542160034179688e-06"}, // round to even
    {0x1.5cf751db94e6bp-20, chars_format::general, 67,
        "1.300000000000000047051663780439678674838432925753295421600341796875e-06"}, // exact

    // Trim trailing zeros and decimal point, scientific notation.
    {0x1.92a737110e454p-19, chars_format::general, 1, "3e-06"}, // scientific notation trims decimal point
    {0x1.92a737110e454p-19, chars_format::general, 2, "3e-06"}, // general trims decimal point and trailing zeros
    {0x1.92a737110e454p-19, chars_format::general, 3, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 4, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 5, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 6, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 7, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 8, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 9, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 10, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 11, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 12, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 13, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 14, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 15, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 16, "3e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 17, "3.0000000000000001e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 18, "3.00000000000000008e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 19, "3.000000000000000076e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 20, "3.000000000000000076e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 21, "3.000000000000000076e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 22, "3.000000000000000076003e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 23, "3.0000000000000000760026e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 24, "3.00000000000000007600257e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 25, "3.000000000000000076002572e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 26, "3.0000000000000000760025723e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 27, "3.00000000000000007600257229e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 28, "3.000000000000000076002572291e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 29, "3.0000000000000000760025722912e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 30, "3.00000000000000007600257229123e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 31, "3.000000000000000076002572291234e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 32, "3.0000000000000000760025722912339e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 33, "3.00000000000000007600257229123386e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 34, "3.000000000000000076002572291233861e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 35, "3.0000000000000000760025722912338608e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 36, "3.00000000000000007600257229123386082e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 37, "3.000000000000000076002572291233860824e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 38, "3.0000000000000000760025722912338608239e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 39, "3.00000000000000007600257229123386082392e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 40, "3.000000000000000076002572291233860823922e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 41, "3.0000000000000000760025722912338608239224e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 42, "3.00000000000000007600257229123386082392244e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 43, "3.000000000000000076002572291233860823922441e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 44, "3.0000000000000000760025722912338608239224413e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 45, "3.00000000000000007600257229123386082392244134e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 46, "3.000000000000000076002572291233860823922441341e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 47, "3.000000000000000076002572291233860823922441341e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 48, "3.00000000000000007600257229123386082392244134098e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 49, "3.000000000000000076002572291233860823922441340983e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 50, "3.0000000000000000760025722912338608239224413409829e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 51, "3.00000000000000007600257229123386082392244134098291e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 52, "3.000000000000000076002572291233860823922441340982914e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 53, "3.000000000000000076002572291233860823922441340982914e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 54, "3.00000000000000007600257229123386082392244134098291397e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 55, "3.000000000000000076002572291233860823922441340982913971e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 56, "3.0000000000000000760025722912338608239224413409829139709e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 57,
        "3.00000000000000007600257229123386082392244134098291397095e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 58,
        "3.000000000000000076002572291233860823922441340982913970947e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 59,
        "3.0000000000000000760025722912338608239224413409829139709473e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 60,
        "3.00000000000000007600257229123386082392244134098291397094727e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 61,
        "3.000000000000000076002572291233860823922441340982913970947266e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 62,
        "3.0000000000000000760025722912338608239224413409829139709472656e-06"},
    {0x1.92a737110e454p-19, chars_format::general, 63,
        "3.00000000000000007600257229123386082392244134098291397094726562e-06"}, // round to even
    {0x1.92a737110e454p-19, chars_format::general, 64,
        "3.000000000000000076002572291233860823922441340982913970947265625e-06"}, // exact

    // Test a large precision with fixed notation and scientific notation,
    // verifying that we remain within the bounds of any lookup tables.
    {0x1.ba9fbe76c8b44p+0, chars_format::general, 5000, "1.72900000000000009237055564881302416324615478515625"},
    {0x1.d01ff9abb93d1p-20, chars_format::general, 5000,
        "1.729000000000000090107283613749533657255597063340246677398681640625e-06"},

    // Test the transitions between fixed notation and scientific notation.
    {5555555.0, chars_format::general, 1, "6e+06"},
    {555555.0, chars_format::general, 1, "6e+05"},
    {55555.0, chars_format::general, 1, "6e+04"},
    {5555.0, chars_format::general, 1, "6e+03"},
    {555.0, chars_format::general, 1, "6e+02"},
    {55.0, chars_format::general, 1, "6e+01"}, // round to even
    {5.0, chars_format::general, 1, "5"},
    {0x1p-3, chars_format::general, 1, "0.1"}, // 0.125
    {0x1p-6, chars_format::general, 1, "0.02"}, // 0.015625
    {0x1p-9, chars_format::general, 1, "0.002"}, // 0.001953125
    {0x1p-13, chars_format::general, 1, "0.0001"}, // 0.0001220703125
    {0x1p-16, chars_format::general, 1, "2e-05"}, // 1.52587890625e-05
    {0x1p-19, chars_format::general, 1, "2e-06"}, // 1.9073486328125e-06

    {5555555.0, chars_format::general, 2, "5.6e+06"},
    {555555.0, chars_format::general, 2, "5.6e+05"},
    {55555.0, chars_format::general, 2, "5.6e+04"},
    {5555.0, chars_format::general, 2, "5.6e+03"},
    {555.0, chars_format::general, 2, "5.6e+02"}, // round to even
    {55.0, chars_format::general, 2, "55"},
    {5.0, chars_format::general, 2, "5"},
    {0x1p-3, chars_format::general, 2, "0.12"}, // round to even
    {0x1p-6, chars_format::general, 2, "0.016"},
    {0x1p-9, chars_format::general, 2, "0.002"},
    {0x1p-13, chars_format::general, 2, "0.00012"},
    {0x1p-16, chars_format::general, 2, "1.5e-05"},
    {0x1p-19, chars_format::general, 2, "1.9e-06"},

    {5555555.0, chars_format::general, 3, "5.56e+06"},
    {555555.0, chars_format::general, 3, "5.56e+05"},
    {55555.0, chars_format::general, 3, "5.56e+04"},
    {5555.0, chars_format::general, 3, "5.56e+03"}, // round to even
    {555.0, chars_format::general, 3, "555"},
    {55.0, chars_format::general, 3, "55"},
    {5.0, chars_format::general, 3, "5"},
    {0x1p-3, chars_format::general, 3, "0.125"},
    {0x1p-6, chars_format::general, 3, "0.0156"},
    {0x1p-9, chars_format::general, 3, "0.00195"},
    {0x1p-13, chars_format::general, 3, "0.000122"},
    {0x1p-16, chars_format::general, 3, "1.53e-05"},
    {0x1p-19, chars_format::general, 3, "1.91e-06"},

    {5555555.0, chars_format::general, 4, "5.556e+06"},
    {555555.0, chars_format::general, 4, "5.556e+05"},
    {55555.0, chars_format::general, 4, "5.556e+04"}, // round to even
    {5555.0, chars_format::general, 4, "5555"},
    {555.0, chars_format::general, 4, "555"},
    {55.0, chars_format::general, 4, "55"},
    {5.0, chars_format::general, 4, "5"},
    {0x1p-3, chars_format::general, 4, "0.125"},
    {0x1p-6, chars_format::general, 4, "0.01562"}, // round to even
    {0x1p-9, chars_format::general, 4, "0.001953"},
    {0x1p-13, chars_format::general, 4, "0.0001221"},
    {0x1p-16, chars_format::general, 4, "1.526e-05"},
    {0x1p-19, chars_format::general, 4, "1.907e-06"},

    {5555555.0, chars_format::general, 5, "5.5556e+06"},
    {555555.0, chars_format::general, 5, "5.5556e+05"}, // round to even
    {55555.0, chars_format::general, 5, "55555"},
    {5555.0, chars_format::general, 5, "5555"},
    {555.0, chars_format::general, 5, "555"},
    {55.0, chars_format::general, 5, "55"},
    {5.0, chars_format::general, 5, "5"},
    {0x1p-3, chars_format::general, 5, "0.125"},
    {0x1p-6, chars_format::general, 5, "0.015625"},
    {0x1p-9, chars_format::general, 5, "0.0019531"},
    {0x1p-13, chars_format::general, 5, "0.00012207"},
    {0x1p-16, chars_format::general, 5, "1.5259e-05"},
    {0x1p-19, chars_format::general, 5, "1.9073e-06"},

    // Tricky corner cases.
    // In these scenarios, rounding can adjust the scientific exponent X,
    // which affects the transition between fixed notation and scientific notation.
    {999.999, chars_format::general, 1, "1e+03"}, // "%.0e" is "1e+03"; X == 3
    {999.999, chars_format::general, 2, "1e+03"}, // "%.1e" is "1.0e+03"; X == 3
    {999.999, chars_format::general, 3, "1e+03"}, // "%.2e" is "1.00e+03"; X == 3
    {999.999, chars_format::general, 4, "1000"}, // "%.3e" is "1.000e+03"; X == 3
    {999.999, chars_format::general, 5, "1000"}, // "%.4e" is "1.0000e+03"; X == 3
    {999.999, chars_format::general, 6, "999.999"}, // "%.5e" is "9.99999e+02"; X == 2

    {999.99, chars_format::general, 1, "1e+03"},
    {999.99, chars_format::general, 2, "1e+03"},
    {999.99, chars_format::general, 3, "1e+03"},
    {999.99, chars_format::general, 4, "1000"},
    {999.99, chars_format::general, 5, "999.99"},
    {999.99, chars_format::general, 6, "999.99"},

    // C11's Standardese is slightly vague about how to perform the trial formatting in scientific notation,
    // but the intention is to use precision P - 1, which is what's used when scientific notation is actually chosen.
    // This example verifies this behavior. Here, P == 3 performs trial formatting with "%.2e", triggering rounding.
    // That increases X to 3, forcing scientific notation to be chosen.
    // If P == 3 performed trial formatting with "%.3e", rounding wouldn't happen,
    // X would be 2, and fixed notation would be chosen.
    {999.9, chars_format::general, 1, "1e+03"}, // "%.0e" is "1e+03"; X == 3
    {999.9, chars_format::general, 2, "1e+03"}, // "%.1e" is "1.0e+03"; X == 3
    {999.9, chars_format::general, 3, "1e+03"}, // "%.2e" is "1.00e+03"; X == 3; SPECIAL CORNER CASE
    {999.9, chars_format::general, 4, "999.9"}, // "%.3e" is "9.999e+02"; X == 2
    {999.9, chars_format::general, 5, "999.9"}, // "%.4e" is "9.9990e+02"; X == 2
    {999.9, chars_format::general, 6, "999.9"}, // "%.5e" is "9.99900e+02"; X == 2

    {999.0, chars_format::general, 1, "1e+03"},
    {999.0, chars_format::general, 2, "1e+03"},
    {999.0, chars_format::general, 3, "999"},
    {999.0, chars_format::general, 4, "999"},
    {999.0, chars_format::general, 5, "999"},
    {999.0, chars_format::general, 6, "999"},

    {99.9999, chars_format::general, 1, "1e+02"},
    {99.9999, chars_format::general, 2, "1e+02"},
    {99.9999, chars_format::general, 3, "100"},
    {99.9999, chars_format::general, 4, "100"},
    {99.9999, chars_format::general, 5, "100"},
    {99.9999, chars_format::general, 6, "99.9999"},

    {99.999, chars_format::general, 1, "1e+02"},
    {99.999, chars_format::general, 2, "1e+02"},
    {99.999, chars_format::general, 3, "100"},
    {99.999, chars_format::general, 4, "100"},
    {99.999, chars_format::general, 5, "99.999"},
    {99.999, chars_format::general, 6, "99.999"},

    {99.99, chars_format::general, 1, "1e+02"},
    {99.99, chars_format::general, 2, "1e+02"},
    {99.99, chars_format::general, 3, "100"},
    {99.99, chars_format::general, 4, "99.99"},
    {99.99, chars_format::general, 5, "99.99"},
    {99.99, chars_format::general, 6, "99.99"},

    {99.9, chars_format::general, 1, "1e+02"},
    {99.9, chars_format::general, 2, "1e+02"},
    {99.9, chars_format::general, 3, "99.9"},
    {99.9, chars_format::general, 4, "99.9"},
    {99.9, chars_format::general, 5, "99.9"},
    {99.9, chars_format::general, 6, "99.9"},

    {99.0, chars_format::general, 1, "1e+02"},
    {99.0, chars_format::general, 2, "99"},
    {99.0, chars_format::general, 3, "99"},
    {99.0, chars_format::general, 4, "99"},
    {99.0, chars_format::general, 5, "99"},
    {99.0, chars_format::general, 6, "99"},

    {9.99999, chars_format::general, 1, "1e+01"},
    {9.99999, chars_format::general, 2, "10"},
    {9.99999, chars_format::general, 3, "10"},
    {9.99999, chars_format::general, 4, "10"},
    {9.99999, chars_format::general, 5, "10"},
    {9.99999, chars_format::general, 6, "9.99999"},

    {9.9999, chars_format::general, 1, "1e+01"},
    {9.9999, chars_format::general, 2, "10"},
    {9.9999, chars_format::general, 3, "10"},
    {9.9999, chars_format::general, 4, "10"},
    {9.9999, chars_format::general, 5, "9.9999"},
    {9.9999, chars_format::general, 6, "9.9999"},

    {9.999, chars_format::general, 1, "1e+01"},
    {9.999, chars_format::general, 2, "10"},
    {9.999, chars_format::general, 3, "10"},
    {9.999, chars_format::general, 4, "9.999"},
    {9.999, chars_format::general, 5, "9.999"},
    {9.999, chars_format::general, 6, "9.999"},

    {9.99, chars_format::general, 1, "1e+01"},
    {9.99, chars_format::general, 2, "10"},
    {9.99, chars_format::general, 3, "9.99"},
    {9.99, chars_format::general, 4, "9.99"},
    {9.99, chars_format::general, 5, "9.99"},
    {9.99, chars_format::general, 6, "9.99"},

    {9.9, chars_format::general, 1, "1e+01"},
    {9.9, chars_format::general, 2, "9.9"},
    {9.9, chars_format::general, 3, "9.9"},
    {9.9, chars_format::general, 4, "9.9"},
    {9.9, chars_format::general, 5, "9.9"},
    {9.9, chars_format::general, 6, "9.9"},

    {9.0, chars_format::general, 1, "9"},
    {9.0, chars_format::general, 2, "9"},
    {9.0, chars_format::general, 3, "9"},
    {9.0, chars_format::general, 4, "9"},
    {9.0, chars_format::general, 5, "9"},
    {9.0, chars_format::general, 6, "9"},

    {0.999999, chars_format::general, 1, "1"},
    {0.999999, chars_format::general, 2, "1"},
    {0.999999, chars_format::general, 3, "1"},
    {0.999999, chars_format::general, 4, "1"},
    {0.999999, chars_format::general, 5, "1"},
    {0.999999, chars_format::general, 6, "0.999999"},

    {0.99999, chars_format::general, 1, "1"},
    {0.99999, chars_format::general, 2, "1"},
    {0.99999, chars_format::general, 3, "1"},
    {0.99999, chars_format::general, 4, "1"},
    {0.99999, chars_format::general, 5, "0.99999"},
    {0.99999, chars_format::general, 6, "0.99999"},

    {0.9999, chars_format::general, 1, "1"},
    {0.9999, chars_format::general, 2, "1"},
    {0.9999, chars_format::general, 3, "1"},
    {0.9999, chars_format::general, 4, "0.9999"},
    {0.9999, chars_format::general, 5, "0.9999"},
    {0.9999, chars_format::general, 6, "0.9999"},

    {0.999, chars_format::general, 1, "1"},
    {0.999, chars_format::general, 2, "1"},
    {0.999, chars_format::general, 3, "0.999"},
    {0.999, chars_format::general, 4, "0.999"},
    {0.999, chars_format::general, 5, "0.999"},
    {0.999, chars_format::general, 6, "0.999"},

    {0.99, chars_format::general, 1, "1"},
    {0.99, chars_format::general, 2, "0.99"},
    {0.99, chars_format::general, 3, "0.99"},
    {0.99, chars_format::general, 4, "0.99"},
    {0.99, chars_format::general, 5, "0.99"},
    {0.99, chars_format::general, 6, "0.99"},

    {0.9, chars_format::general, 1, "0.9"},
    {0.9, chars_format::general, 2, "0.9"},
    {0.9, chars_format::general, 3, "0.9"},
    {0.9, chars_format::general, 4, "0.9"},
    {0.9, chars_format::general, 5, "0.9"},
    {0.9, chars_format::general, 6, "0.9"},

    {0.0999999, chars_format::general, 1, "0.1"},
    {0.0999999, chars_format::general, 2, "0.1"},
    {0.0999999, chars_format::general, 3, "0.1"},
    {0.0999999, chars_format::general, 4, "0.1"},
    {0.0999999, chars_format::general, 5, "0.1"},
    {0.0999999, chars_format::general, 6, "0.0999999"},

    {0.099999, chars_format::general, 1, "0.1"},
    {0.099999, chars_format::general, 2, "0.1"},
    {0.099999, chars_format::general, 3, "0.1"},
    {0.099999, chars_format::general, 4, "0.1"},
    {0.099999, chars_format::general, 5, "0.099999"},
    {0.099999, chars_format::general, 6, "0.099999"},

    {0.09999, chars_format::general, 1, "0.1"},
    {0.09999, chars_format::general, 2, "0.1"},
    {0.09999, chars_format::general, 3, "0.1"},
    {0.09999, chars_format::general, 4, "0.09999"},
    {0.09999, chars_format::general, 5, "0.09999"},
    {0.09999, chars_format::general, 6, "0.09999"},

    {0.0999, chars_format::general, 1, "0.1"},
    {0.0999, chars_format::general, 2, "0.1"},
    {0.0999, chars_format::general, 3, "0.0999"},
    {0.0999, chars_format::general, 4, "0.0999"},
    {0.0999, chars_format::general, 5, "0.0999"},
    {0.0999, chars_format::general, 6, "0.0999"},

    {0.099, chars_format::general, 1, "0.1"},
    {0.099, chars_format::general, 2, "0.099"},
    {0.099, chars_format::general, 3, "0.099"},
    {0.099, chars_format::general, 4, "0.099"},
    {0.099, chars_format::general, 5, "0.099"},
    {0.099, chars_format::general, 6, "0.099"},

    {0.09, chars_format::general, 1, "0.09"},
    {0.09, chars_format::general, 2, "0.09"},
    {0.09, chars_format::general, 3, "0.09"},
    {0.09, chars_format::general, 4, "0.09"},
    {0.09, chars_format::general, 5, "0.09"},
    {0.09, chars_format::general, 6, "0.09"},

    {0.00999999, chars_format::general, 1, "0.01"},
    {0.00999999, chars_format::general, 2, "0.01"},
    {0.00999999, chars_format::general, 3, "0.01"},
    {0.00999999, chars_format::general, 4, "0.01"},
    {0.00999999, chars_format::general, 5, "0.01"},
    {0.00999999, chars_format::general, 6, "0.00999999"},

    {0.0099999, chars_format::general, 1, "0.01"},
    {0.0099999, chars_format::general, 2, "0.01"},
    {0.0099999, chars_format::general, 3, "0.01"},
    {0.0099999, chars_format::general, 4, "0.01"},
    {0.0099999, chars_format::general, 5, "0.0099999"},
    {0.0099999, chars_format::general, 6, "0.0099999"},

    {0.009999, chars_format::general, 1, "0.01"},
    {0.009999, chars_format::general, 2, "0.01"},
    {0.009999, chars_format::general, 3, "0.01"},
    {0.009999, chars_format::general, 4, "0.009999"},
    {0.009999, chars_format::general, 5, "0.009999"},
    {0.009999, chars_format::general, 6, "0.009999"},

    {0.00999, chars_format::general, 1, "0.01"},
    {0.00999, chars_format::general, 2, "0.01"},
    {0.00999, chars_format::general, 3, "0.00999"},
    {0.00999, chars_format::general, 4, "0.00999"},
    {0.00999, chars_format::general, 5, "0.00999"},
    {0.00999, chars_format::general, 6, "0.00999"},

    {0.0099, chars_format::general, 1, "0.01"},
    {0.0099, chars_format::general, 2, "0.0099"},
    {0.0099, chars_format::general, 3, "0.0099"},
    {0.0099, chars_format::general, 4, "0.0099"},
    {0.0099, chars_format::general, 5, "0.0099"},
    {0.0099, chars_format::general, 6, "0.0099"},

    {0.009, chars_format::general, 1, "0.009"},
    {0.009, chars_format::general, 2, "0.009"},
    {0.009, chars_format::general, 3, "0.009"},
    {0.009, chars_format::general, 4, "0.009"},
    {0.009, chars_format::general, 5, "0.009"},
    {0.009, chars_format::general, 6, "0.009"},

    {0.000999999, chars_format::general, 1, "0.001"},
    {0.000999999, chars_format::general, 2, "0.001"},
    {0.000999999, chars_format::general, 3, "0.001"},
    {0.000999999, chars_format::general, 4, "0.001"},
    {0.000999999, chars_format::general, 5, "0.001"},
    {0.000999999, chars_format::general, 6, "0.000999999"},

    {0.00099999, chars_format::general, 1, "0.001"},
    {0.00099999, chars_format::general, 2, "0.001"},
    {0.00099999, chars_format::general, 3, "0.001"},
    {0.00099999, chars_format::general, 4, "0.001"},
    {0.00099999, chars_format::general, 5, "0.00099999"},
    {0.00099999, chars_format::general, 6, "0.00099999"},

    {0.0009999, chars_format::general, 1, "0.001"},
    {0.0009999, chars_format::general, 2, "0.001"},
    {0.0009999, chars_format::general, 3, "0.001"},
    {0.0009999, chars_format::general, 4, "0.0009999"},
    {0.0009999, chars_format::general, 5, "0.0009999"},
    {0.0009999, chars_format::general, 6, "0.0009999"},

    {0.000999, chars_format::general, 1, "0.001"},
    {0.000999, chars_format::general, 2, "0.001"},
    {0.000999, chars_format::general, 3, "0.000999"},
    {0.000999, chars_format::general, 4, "0.000999"},
    {0.000999, chars_format::general, 5, "0.000999"},
    {0.000999, chars_format::general, 6, "0.000999"},

    {0.00099, chars_format::general, 1, "0.001"},
    {0.00099, chars_format::general, 2, "0.00099"},
    {0.00099, chars_format::general, 3, "0.00099"},
    {0.00099, chars_format::general, 4, "0.00099"},
    {0.00099, chars_format::general, 5, "0.00099"},
    {0.00099, chars_format::general, 6, "0.00099"},

    {0.0009, chars_format::general, 1, "0.0009"},
    {0.0009, chars_format::general, 2, "0.0009"},
    {0.0009, chars_format::general, 3, "0.0009"},
    {0.0009, chars_format::general, 4, "0.0009"},
    {0.0009, chars_format::general, 5, "0.0009"},
    {0.0009, chars_format::general, 6, "0.0009"},

    // Having a scientific exponent X == -5 triggers scientific notation.
    // If rounding adjusts this to X == -4, then fixed notation will be selected.
    {0.0000999999, chars_format::general, 1, "0.0001"},
    {0.0000999999, chars_format::general, 2, "0.0001"},
    {0.0000999999, chars_format::general, 3, "0.0001"},
    {0.0000999999, chars_format::general, 4, "0.0001"},
    {0.0000999999, chars_format::general, 5, "0.0001"},
    {0.0000999999, chars_format::general, 6, "9.99999e-05"},

    {0.000099999, chars_format::general, 1, "0.0001"},
    {0.000099999, chars_format::general, 2, "0.0001"},
    {0.000099999, chars_format::general, 3, "0.0001"},
    {0.000099999, chars_format::general, 4, "0.0001"},
    {0.000099999, chars_format::general, 5, "9.9999e-05"},
    {0.000099999, chars_format::general, 6, "9.9999e-05"},

    {0.00009999, chars_format::general, 1, "0.0001"},
    {0.00009999, chars_format::general, 2, "0.0001"},
    {0.00009999, chars_format::general, 3, "0.0001"},
    {0.00009999, chars_format::general, 4, "9.999e-05"},
    {0.00009999, chars_format::general, 5, "9.999e-05"},
    {0.00009999, chars_format::general, 6, "9.999e-05"},

    {0.0000999, chars_format::general, 1, "0.0001"},
    {0.0000999, chars_format::general, 2, "0.0001"},
    {0.0000999, chars_format::general, 3, "9.99e-05"},
    {0.0000999, chars_format::general, 4, "9.99e-05"},
    {0.0000999, chars_format::general, 5, "9.99e-05"},
    {0.0000999, chars_format::general, 6, "9.99e-05"},

    {0.000099, chars_format::general, 1, "0.0001"},
    {0.000099, chars_format::general, 2, "9.9e-05"},
    {0.000099, chars_format::general, 3, "9.9e-05"},
    {0.000099, chars_format::general, 4, "9.9e-05"},
    {0.000099, chars_format::general, 5, "9.9e-05"},
    {0.000099, chars_format::general, 6, "9.9e-05"},

    {0.00009, chars_format::general, 1, "9e-05"},
    {0.00009, chars_format::general, 2, "9e-05"},
    {0.00009, chars_format::general, 3, "9e-05"},
    {0.00009, chars_format::general, 4, "9e-05"},
    {0.00009, chars_format::general, 5, "9e-05"},
    {0.00009, chars_format::general, 6, "9e-05"},

    // Rounding test cases without exponent-adjusting behavior.
    {2999.999, chars_format::general, 1, "3e+03"},
    {2999.999, chars_format::general, 2, "3e+03"},
    {2999.999, chars_format::general, 3, "3e+03"},
    {2999.999, chars_format::general, 4, "3000"},
    {2999.999, chars_format::general, 5, "3000"},
    {2999.999, chars_format::general, 6, "3000"},

    {2999.99, chars_format::general, 1, "3e+03"},
    {2999.99, chars_format::general, 2, "3e+03"},
    {2999.99, chars_format::general, 3, "3e+03"},
    {2999.99, chars_format::general, 4, "3000"},
    {2999.99, chars_format::general, 5, "3000"},
    {2999.99, chars_format::general, 6, "2999.99"},

    {2999.9, chars_format::general, 1, "3e+03"},
    {2999.9, chars_format::general, 2, "3e+03"},
    {2999.9, chars_format::general, 3, "3e+03"},
    {2999.9, chars_format::general, 4, "3000"},
    {2999.9, chars_format::general, 5, "2999.9"},
    {2999.9, chars_format::general, 6, "2999.9"},

    {2999.0, chars_format::general, 1, "3e+03"},
    {2999.0, chars_format::general, 2, "3e+03"},
    {2999.0, chars_format::general, 3, "3e+03"},
    {2999.0, chars_format::general, 4, "2999"},
    {2999.0, chars_format::general, 5, "2999"},
    {2999.0, chars_format::general, 6, "2999"},

    {299.999, chars_format::general, 1, "3e+02"},
    {299.999, chars_format::general, 2, "3e+02"},
    {299.999, chars_format::general, 3, "300"},
    {299.999, chars_format::general, 4, "300"},
    {299.999, chars_format::general, 5, "300"},
    {299.999, chars_format::general, 6, "299.999"},

    {299.99, chars_format::general, 1, "3e+02"},
    {299.99, chars_format::general, 2, "3e+02"},
    {299.99, chars_format::general, 3, "300"},
    {299.99, chars_format::general, 4, "300"},
    {299.99, chars_format::general, 5, "299.99"},
    {299.99, chars_format::general, 6, "299.99"},

    {299.9, chars_format::general, 1, "3e+02"},
    {299.9, chars_format::general, 2, "3e+02"},
    {299.9, chars_format::general, 3, "300"},
    {299.9, chars_format::general, 4, "299.9"},
    {299.9, chars_format::general, 5, "299.9"},
    {299.9, chars_format::general, 6, "299.9"},

    {299.0, chars_format::general, 1, "3e+02"},
    {299.0, chars_format::general, 2, "3e+02"},
    {299.0, chars_format::general, 3, "299"},
    {299.0, chars_format::general, 4, "299"},
    {299.0, chars_format::general, 5, "299"},
    {299.0, chars_format::general, 6, "299"},

    {29.999, chars_format::general, 1, "3e+01"},
    {29.999, chars_format::general, 2, "30"},
    {29.999, chars_format::general, 3, "30"},
    {29.999, chars_format::general, 4, "30"},
    {29.999, chars_format::general, 5, "29.999"},
    {29.999, chars_format::general, 6, "29.999"},

    {29.99, chars_format::general, 1, "3e+01"},
    {29.99, chars_format::general, 2, "30"},
    {29.99, chars_format::general, 3, "30"},
    {29.99, chars_format::general, 4, "29.99"},
    {29.99, chars_format::general, 5, "29.99"},
    {29.99, chars_format::general, 6, "29.99"},

    {29.9, chars_format::general, 1, "3e+01"},
    {29.9, chars_format::general, 2, "30"},
    {29.9, chars_format::general, 3, "29.9"},
    {29.9, chars_format::general, 4, "29.9"},
    {29.9, chars_format::general, 5, "29.9"},
    {29.9, chars_format::general, 6, "29.9"},

    {29.0, chars_format::general, 1, "3e+01"},
    {29.0, chars_format::general, 2, "29"},
    {29.0, chars_format::general, 3, "29"},
    {29.0, chars_format::general, 4, "29"},
    {29.0, chars_format::general, 5, "29"},
    {29.0, chars_format::general, 6, "29"},

    {2.999, chars_format::general, 1, "3"},
    {2.999, chars_format::general, 2, "3"},
    {2.999, chars_format::general, 3, "3"},
    {2.999, chars_format::general, 4, "2.999"},
    {2.999, chars_format::general, 5, "2.999"},
    {2.999, chars_format::general, 6, "2.999"},

    {2.99, chars_format::general, 1, "3"},
    {2.99, chars_format::general, 2, "3"},
    {2.99, chars_format::general, 3, "2.99"},
    {2.99, chars_format::general, 4, "2.99"},
    {2.99, chars_format::general, 5, "2.99"},
    {2.99, chars_format::general, 6, "2.99"},

    {2.9, chars_format::general, 1, "3"},
    {2.9, chars_format::general, 2, "2.9"},
    {2.9, chars_format::general, 3, "2.9"},
    {2.9, chars_format::general, 4, "2.9"},
    {2.9, chars_format::general, 5, "2.9"},
    {2.9, chars_format::general, 6, "2.9"},

    {2.0, chars_format::general, 1, "2"},
    {2.0, chars_format::general, 2, "2"},
    {2.0, chars_format::general, 3, "2"},
    {2.0, chars_format::general, 4, "2"},
    {2.0, chars_format::general, 5, "2"},
    {2.0, chars_format::general, 6, "2"},

    {0.2999, chars_format::general, 1, "0.3"},
    {0.2999, chars_format::general, 2, "0.3"},
    {0.2999, chars_format::general, 3, "0.3"},
    {0.2999, chars_format::general, 4, "0.2999"},
    {0.2999, chars_format::general, 5, "0.2999"},
    {0.2999, chars_format::general, 6, "0.2999"},

    {0.299, chars_format::general, 1, "0.3"},
    {0.299, chars_format::general, 2, "0.3"},
    {0.299, chars_format::general, 3, "0.299"},
    {0.299, chars_format::general, 4, "0.299"},
    {0.299, chars_format::general, 5, "0.299"},
    {0.299, chars_format::general, 6, "0.299"},

    {0.29, chars_format::general, 1, "0.3"},
    {0.29, chars_format::general, 2, "0.29"},
    {0.29, chars_format::general, 3, "0.29"},
    {0.29, chars_format::general, 4, "0.29"},
    {0.29, chars_format::general, 5, "0.29"},
    {0.29, chars_format::general, 6, "0.29"},

    {0.2, chars_format::general, 1, "0.2"},
    {0.2, chars_format::general, 2, "0.2"},
    {0.2, chars_format::general, 3, "0.2"},
    {0.2, chars_format::general, 4, "0.2"},
    {0.2, chars_format::general, 5, "0.2"},
    {0.2, chars_format::general, 6, "0.2"},

    {0.02999, chars_format::general, 1, "0.03"},
    {0.02999, chars_format::general, 2, "0.03"},
    {0.02999, chars_format::general, 3, "0.03"},
    {0.02999, chars_format::general, 4, "0.02999"},
    {0.02999, chars_format::general, 5, "0.02999"},
    {0.02999, chars_format::general, 6, "0.02999"},

    {0.0299, chars_format::general, 1, "0.03"},
    {0.0299, chars_format::general, 2, "0.03"},
    {0.0299, chars_format::general, 3, "0.0299"},
    {0.0299, chars_format::general, 4, "0.0299"},
    {0.0299, chars_format::general, 5, "0.0299"},
    {0.0299, chars_format::general, 6, "0.0299"},

    {0.029, chars_format::general, 1, "0.03"},
    {0.029, chars_format::general, 2, "0.029"},
    {0.029, chars_format::general, 3, "0.029"},
    {0.029, chars_format::general, 4, "0.029"},
    {0.029, chars_format::general, 5, "0.029"},
    {0.029, chars_format::general, 6, "0.029"},

    {0.02, chars_format::general, 1, "0.02"},
    {0.02, chars_format::general, 2, "0.02"},
    {0.02, chars_format::general, 3, "0.02"},
    {0.02, chars_format::general, 4, "0.02"},
    {0.02, chars_format::general, 5, "0.02"},
    {0.02, chars_format::general, 6, "0.02"},

    {0.002999, chars_format::general, 1, "0.003"},
    {0.002999, chars_format::general, 2, "0.003"},
    {0.002999, chars_format::general, 3, "0.003"},
    {0.002999, chars_format::general, 4, "0.002999"},
    {0.002999, chars_format::general, 5, "0.002999"},
    {0.002999, chars_format::general, 6, "0.002999"},

    {0.00299, chars_format::general, 1, "0.003"},
    {0.00299, chars_format::general, 2, "0.003"},
    {0.00299, chars_format::general, 3, "0.00299"},
    {0.00299, chars_format::general, 4, "0.00299"},
    {0.00299, chars_format::general, 5, "0.00299"},
    {0.00299, chars_format::general, 6, "0.00299"},

    {0.0029, chars_format::general, 1, "0.003"},
    {0.0029, chars_format::general, 2, "0.0029"},
    {0.0029, chars_format::general, 3, "0.0029"},
    {0.0029, chars_format::general, 4, "0.0029"},
    {0.0029, chars_format::general, 5, "0.0029"},
    {0.0029, chars_format::general, 6, "0.0029"},

    {0.002, chars_format::general, 1, "0.002"},
    {0.002, chars_format::general, 2, "0.002"},
    {0.002, chars_format::general, 3, "0.002"},
    {0.002, chars_format::general, 4, "0.002"},
    {0.002, chars_format::general, 5, "0.002"},
    {0.002, chars_format::general, 6, "0.002"},

    {0.0002999, chars_format::general, 1, "0.0003"},
    {0.0002999, chars_format::general, 2, "0.0003"},
    {0.0002999, chars_format::general, 3, "0.0003"},
    {0.0002999, chars_format::general, 4, "0.0002999"},
    {0.0002999, chars_format::general, 5, "0.0002999"},
    {0.0002999, chars_format::general, 6, "0.0002999"},

    {0.000299, chars_format::general, 1, "0.0003"},
    {0.000299, chars_format::general, 2, "0.0003"},
    {0.000299, chars_format::general, 3, "0.000299"},
    {0.000299, chars_format::general, 4, "0.000299"},
    {0.000299, chars_format::general, 5, "0.000299"},
    {0.000299, chars_format::general, 6, "0.000299"},

    {0.00029, chars_format::general, 1, "0.0003"},
    {0.00029, chars_format::general, 2, "0.00029"},
    {0.00029, chars_format::general, 3, "0.00029"},
    {0.00029, chars_format::general, 4, "0.00029"},
    {0.00029, chars_format::general, 5, "0.00029"},
    {0.00029, chars_format::general, 6, "0.00029"},

    {0.0002, chars_format::general, 1, "0.0002"},
    {0.0002, chars_format::general, 2, "0.0002"},
    {0.0002, chars_format::general, 3, "0.0002"},
    {0.0002, chars_format::general, 4, "0.0002"},
    {0.0002, chars_format::general, 5, "0.0002"},
    {0.0002, chars_format::general, 6, "0.0002"},

    {0.00002999, chars_format::general, 1, "3e-05"},
    {0.00002999, chars_format::general, 2, "3e-05"},
    {0.00002999, chars_format::general, 3, "3e-05"},
    {0.00002999, chars_format::general, 4, "2.999e-05"},
    {0.00002999, chars_format::general, 5, "2.999e-05"},
    {0.00002999, chars_format::general, 6, "2.999e-05"},

    {0.0000299, chars_format::general, 1, "3e-05"},
    {0.0000299, chars_format::general, 2, "3e-05"},
    {0.0000299, chars_format::general, 3, "2.99e-05"},
    {0.0000299, chars_format::general, 4, "2.99e-05"},
    {0.0000299, chars_format::general, 5, "2.99e-05"},
    {0.0000299, chars_format::general, 6, "2.99e-05"},

    {0.000029, chars_format::general, 1, "3e-05"},
    {0.000029, chars_format::general, 2, "2.9e-05"},
    {0.000029, chars_format::general, 3, "2.9e-05"},
    {0.000029, chars_format::general, 4, "2.9e-05"},
    {0.000029, chars_format::general, 5, "2.9e-05"},
    {0.000029, chars_format::general, 6, "2.9e-05"},

    {0.00002, chars_format::general, 1, "2e-05"},
    {0.00002, chars_format::general, 2, "2e-05"},
    {0.00002, chars_format::general, 3, "2e-05"},
    {0.00002, chars_format::general, 4, "2e-05"},
    {0.00002, chars_format::general, 5, "2e-05"},
    {0.00002, chars_format::general, 6, "2e-05"},

    // Test the transitions between values of the scientific exponent X.
    // For brevity, we avoid testing all possible combinations of P and X. Instead, we test:
    // * All values of P where some X can be affected by rounding. (For double, this is [1, 15].)
    // * P == 25, which is arbitrary.
    // * P == numeric_limits::max_exponent10 + 1. This selects fixed notation for numeric_limits::max(),
    //   so it's the largest interesting value of P.
    // * Finally, we test the transitions around X == P - 1, ensuring that we can recognize every value of X.
    {0x1.8e757928e0c9dp-14, chars_format::general, 1, "9e-05"},
    {0x1.8e757928e0c9ep-14, chars_format::general, 1, "0.0001"},
    {0x1.f212d77318fc5p-11, chars_format::general, 1, "0.0009"},
    {0x1.f212d77318fc6p-11, chars_format::general, 1, "0.001"},
    {0x1.374bc6a7ef9dbp-7, chars_format::general, 1, "0.009"},
    {0x1.374bc6a7ef9dcp-7, chars_format::general, 1, "0.01"},
    {0x1.851eb851eb851p-4, chars_format::general, 1, "0.09"},
    {0x1.851eb851eb852p-4, chars_format::general, 1, "0.1"},
    {0x1.e666666666666p-1, chars_format::general, 1, "0.9"},
    {0x1.e666666666667p-1, chars_format::general, 1, "1"},
    {0x1.2ffffffffffffp+3, chars_format::general, 1, "9"},
    {0x1.3000000000000p+3, chars_format::general, 1, "1e+01"},
    {0x1.a1554fbdad751p-14, chars_format::general, 2, "9.9e-05"},
    {0x1.a1554fbdad752p-14, chars_format::general, 2, "0.0001"},
    {0x1.04d551d68c692p-10, chars_format::general, 2, "0.00099"},
    {0x1.04d551d68c693p-10, chars_format::general, 2, "0.001"},
    {0x1.460aa64c2f837p-7, chars_format::general, 2, "0.0099"},
    {0x1.460aa64c2f838p-7, chars_format::general, 2, "0.01"},
    {0x1.978d4fdf3b645p-4, chars_format::general, 2, "0.099"},
    {0x1.978d4fdf3b646p-4, chars_format::general, 2, "0.1"},
    {0x1.fd70a3d70a3d7p-1, chars_format::general, 2, "0.99"},
    {0x1.fd70a3d70a3d8p-1, chars_format::general, 2, "1"},
    {0x1.3e66666666666p+3, chars_format::general, 2, "9.9"},
    {0x1.3e66666666667p+3, chars_format::general, 2, "10"},
    {0x1.8dfffffffffffp+6, chars_format::general, 2, "99"},
    {0x1.8e00000000000p+6, chars_format::general, 2, "1e+02"},
    {0x1.a3387ecc8eb96p-14, chars_format::general, 3, "9.99e-05"},
    {0x1.a3387ecc8eb97p-14, chars_format::general, 3, "0.0001"},
    {0x1.06034f3fd933ep-10, chars_format::general, 3, "0.000999"},
    {0x1.06034f3fd933fp-10, chars_format::general, 3, "0.001"},
    {0x1.4784230fcf80dp-7, chars_format::general, 3, "0.00999"},
    {0x1.4784230fcf80ep-7, chars_format::general, 3, "0.01"},
    {0x1.99652bd3c3611p-4, chars_format::general, 3, "0.0999"},
    {0x1.99652bd3c3612p-4, chars_format::general, 3, "0.1"},
    {0x1.ffbe76c8b4395p-1, chars_format::general, 3, "0.999"},
    {0x1.ffbe76c8b4396p-1, chars_format::general, 3, "1"},
    {0x1.3fd70a3d70a3dp+3, chars_format::general, 3, "9.99"},
    {0x1.3fd70a3d70a3ep+3, chars_format::general, 3, "10"},
    {0x1.8fcccccccccccp+6, chars_format::general, 3, "99.9"},
    {0x1.8fccccccccccdp+6, chars_format::general, 3, "100"},
    {0x1.f3bffffffffffp+9, chars_format::general, 3, "999"},
    {0x1.f3c0000000000p+9, chars_format::general, 3, "1e+03"},
    {0x1.a368d04e0ba6ap-14, chars_format::general, 4, "9.999e-05"},
    {0x1.a368d04e0ba6bp-14, chars_format::general, 4, "0.0001"},
    {0x1.06218230c7482p-10, chars_format::general, 4, "0.0009999"},
    {0x1.06218230c7483p-10, chars_format::general, 4, "0.001"},
    {0x1.47a9e2bcf91a3p-7, chars_format::general, 4, "0.009999"},
    {0x1.47a9e2bcf91a4p-7, chars_format::general, 4, "0.01"},
    {0x1.99945b6c3760bp-4, chars_format::general, 4, "0.09999"},
    {0x1.99945b6c3760cp-4, chars_format::general, 4, "0.1"},
    {0x1.fff972474538ep-1, chars_format::general, 4, "0.9999"},
    {0x1.fff972474538fp-1, chars_format::general, 4, "1"},
    {0x1.3ffbe76c8b439p+3, chars_format::general, 4, "9.999"},
    {0x1.3ffbe76c8b43ap+3, chars_format::general, 4, "10"},
    {0x1.8ffae147ae147p+6, chars_format::general, 4, "99.99"},
    {0x1.8ffae147ae148p+6, chars_format::general, 4, "100"},
    {0x1.f3f9999999999p+9, chars_format::general, 4, "999.9"},
    {0x1.f3f999999999ap+9, chars_format::general, 4, "1000"},
    {0x1.387bfffffffffp+13, chars_format::general, 4, "9999"},
    {0x1.387c000000000p+13, chars_format::general, 4, "1e+04"},
    {0x1.a36da54164f19p-14, chars_format::general, 5, "9.9999e-05"},
    {0x1.a36da54164f1ap-14, chars_format::general, 5, "0.0001"},
    {0x1.06248748df16fp-10, chars_format::general, 5, "0.00099999"},
    {0x1.06248748df170p-10, chars_format::general, 5, "0.001"},
    {0x1.47ada91b16dcbp-7, chars_format::general, 5, "0.0099999"},
    {0x1.47ada91b16dccp-7, chars_format::general, 5, "0.01"},
    {0x1.99991361dc93ep-4, chars_format::general, 5, "0.099999"},
    {0x1.99991361dc93fp-4, chars_format::general, 5, "0.1"},
    {0x1.ffff583a53b8ep-1, chars_format::general, 5, "0.99999"},
    {0x1.ffff583a53b8fp-1, chars_format::general, 5, "1"},
    {0x1.3fff972474538p+3, chars_format::general, 5, "9.9999"},
    {0x1.3fff972474539p+3, chars_format::general, 5, "10"},
    {0x1.8fff7ced91687p+6, chars_format::general, 5, "99.999"},
    {0x1.8fff7ced91688p+6, chars_format::general, 5, "100"},
    {0x1.f3ff5c28f5c28p+9, chars_format::general, 5, "999.99"},
    {0x1.f3ff5c28f5c29p+9, chars_format::general, 5, "1000"},
    {0x1.387f999999999p+13, chars_format::general, 5, "9999.9"},
    {0x1.387f99999999ap+13, chars_format::general, 5, "10000"},
    {0x1.869f7ffffffffp+16, chars_format::general, 5, "99999"},
    {0x1.869f800000000p+16, chars_format::general, 5, "1e+05"},
    {0x1.a36e20f35445dp-14, chars_format::general, 6, "9.99999e-05"},
    {0x1.a36e20f35445ep-14, chars_format::general, 6, "0.0001"},
    {0x1.0624d49814abap-10, chars_format::general, 6, "0.000999999"},
    {0x1.0624d49814abbp-10, chars_format::general, 6, "0.001"},
    {0x1.47ae09be19d69p-7, chars_format::general, 6, "0.00999999"},
    {0x1.47ae09be19d6ap-7, chars_format::general, 6, "0.01"},
    {0x1.99998c2da04c3p-4, chars_format::general, 6, "0.0999999"},
    {0x1.99998c2da04c4p-4, chars_format::general, 6, "0.1"},
    {0x1.ffffef39085f4p-1, chars_format::general, 6, "0.999999"},
    {0x1.ffffef39085f5p-1, chars_format::general, 6, "1"},
    {0x1.3ffff583a53b8p+3, chars_format::general, 6, "9.99999"},
    {0x1.3ffff583a53b9p+3, chars_format::general, 6, "10"},
    {0x1.8ffff2e48e8a7p+6, chars_format::general, 6, "99.9999"},
    {0x1.8ffff2e48e8a8p+6, chars_format::general, 6, "100"},
    {0x1.f3ffef9db22d0p+9, chars_format::general, 6, "999.999"},
    {0x1.f3ffef9db22d1p+9, chars_format::general, 6, "1000"},
    {0x1.387ff5c28f5c2p+13, chars_format::general, 6, "9999.99"},
    {0x1.387ff5c28f5c3p+13, chars_format::general, 6, "10000"},
    {0x1.869ff33333333p+16, chars_format::general, 6, "99999.9"},
    {0x1.869ff33333334p+16, chars_format::general, 6, "100000"},
    {0x1.e847effffffffp+19, chars_format::general, 6, "999999"},
    {0x1.e847f00000000p+19, chars_format::general, 6, "1e+06"},
    {0x1.a36e2d51ec34bp-14, chars_format::general, 7, "9.999999e-05"},
    {0x1.a36e2d51ec34cp-14, chars_format::general, 7, "0.0001"},
    {0x1.0624dc5333a0ep-10, chars_format::general, 7, "0.0009999999"},
    {0x1.0624dc5333a0fp-10, chars_format::general, 7, "0.001"},
    {0x1.47ae136800892p-7, chars_format::general, 7, "0.009999999"},
    {0x1.47ae136800893p-7, chars_format::general, 7, "0.01"},
    {0x1.9999984200ab7p-4, chars_format::general, 7, "0.09999999"},
    {0x1.9999984200ab8p-4, chars_format::general, 7, "0.1"},
    {0x1.fffffe5280d65p-1, chars_format::general, 7, "0.9999999"},
    {0x1.fffffe5280d66p-1, chars_format::general, 7, "1"},
    {0x1.3ffffef39085fp+3, chars_format::general, 7, "9.999999"},
    {0x1.3ffffef390860p+3, chars_format::general, 7, "10"},
    {0x1.8ffffeb074a77p+6, chars_format::general, 7, "99.99999"},
    {0x1.8ffffeb074a78p+6, chars_format::general, 7, "100"},
    {0x1.f3fffe5c91d14p+9, chars_format::general, 7, "999.9999"},
    {0x1.f3fffe5c91d15p+9, chars_format::general, 7, "1000"},
    {0x1.387ffef9db22dp+13, chars_format::general, 7, "9999.999"},
    {0x1.387ffef9db22ep+13, chars_format::general, 7, "10000"},
    {0x1.869ffeb851eb8p+16, chars_format::general, 7, "99999.99"},
    {0x1.869ffeb851eb9p+16, chars_format::general, 7, "100000"},
    {0x1.e847fe6666666p+19, chars_format::general, 7, "999999.9"},
    {0x1.e847fe6666667p+19, chars_format::general, 7, "1000000"},
    {0x1.312cfefffffffp+23, chars_format::general, 7, "9999999"},
    {0x1.312cff0000000p+23, chars_format::general, 7, "1e+07"},
    {0x1.a36e2e8e94ffcp-14, chars_format::general, 8, "9.9999999e-05"},
    {0x1.a36e2e8e94ffdp-14, chars_format::general, 8, "0.0001"},
    {0x1.0624dd191d1fdp-10, chars_format::general, 8, "0.00099999999"},
    {0x1.0624dd191d1fep-10, chars_format::general, 8, "0.001"},
    {0x1.47ae145f6467dp-7, chars_format::general, 8, "0.0099999999"},
    {0x1.47ae145f6467ep-7, chars_format::general, 8, "0.01"},
    {0x1.999999773d81cp-4, chars_format::general, 8, "0.099999999"},
    {0x1.999999773d81dp-4, chars_format::general, 8, "0.1"},
    {0x1.ffffffd50ce23p-1, chars_format::general, 8, "0.99999999"},
    {0x1.ffffffd50ce24p-1, chars_format::general, 8, "1"},
    {0x1.3fffffe5280d6p+3, chars_format::general, 8, "9.9999999"},
    {0x1.3fffffe5280d7p+3, chars_format::general, 8, "10"},
    {0x1.8fffffde7210bp+6, chars_format::general, 8, "99.999999"},
    {0x1.8fffffde7210cp+6, chars_format::general, 8, "100"},
    {0x1.f3ffffd60e94ep+9, chars_format::general, 8, "999.99999"},
    {0x1.f3ffffd60e94fp+9, chars_format::general, 8, "1000"},
    {0x1.387fffe5c91d1p+13, chars_format::general, 8, "9999.9999"},
    {0x1.387fffe5c91d2p+13, chars_format::general, 8, "10000"},
    {0x1.869fffdf3b645p+16, chars_format::general, 8, "99999.999"},
    {0x1.869fffdf3b646p+16, chars_format::general, 8, "100000"},
    {0x1.e847ffd70a3d7p+19, chars_format::general, 8, "999999.99"},
    {0x1.e847ffd70a3d8p+19, chars_format::general, 8, "1000000"},
    {0x1.312cffe666666p+23, chars_format::general, 8, "9999999.9"},
    {0x1.312cffe666667p+23, chars_format::general, 8, "10000000"},
    {0x1.7d783fdffffffp+26, chars_format::general, 8, "99999999"},
    {0x1.7d783fe000000p+26, chars_format::general, 8, "1e+08"},
    {0x1.a36e2eae3f7a7p-14, chars_format::general, 9, "9.99999999e-05"},
    {0x1.a36e2eae3f7a8p-14, chars_format::general, 9, "0.0001"},
    {0x1.0624dd2ce7ac8p-10, chars_format::general, 9, "0.000999999999"},
    {0x1.0624dd2ce7ac9p-10, chars_format::general, 9, "0.001"},
    {0x1.47ae14782197bp-7, chars_format::general, 9, "0.00999999999"},
    {0x1.47ae14782197cp-7, chars_format::general, 9, "0.01"},
    {0x1.9999999629fd9p-4, chars_format::general, 9, "0.0999999999"},
    {0x1.9999999629fdap-4, chars_format::general, 9, "0.1"},
    {0x1.fffffffbb47d0p-1, chars_format::general, 9, "0.999999999"},
    {0x1.fffffffbb47d1p-1, chars_format::general, 9, "1"},
    {0x1.3ffffffd50ce2p+3, chars_format::general, 9, "9.99999999"},
    {0x1.3ffffffd50ce3p+3, chars_format::general, 9, "10"},
    {0x1.8ffffffca501ap+6, chars_format::general, 9, "99.9999999"},
    {0x1.8ffffffca501bp+6, chars_format::general, 9, "100"},
    {0x1.f3fffffbce421p+9, chars_format::general, 9, "999.999999"},
    {0x1.f3fffffbce422p+9, chars_format::general, 9, "1000"},
    {0x1.387ffffd60e94p+13, chars_format::general, 9, "9999.99999"},
    {0x1.387ffffd60e95p+13, chars_format::general, 9, "10000"},
    {0x1.869ffffcb923ap+16, chars_format::general, 9, "99999.9999"},
    {0x1.869ffffcb923bp+16, chars_format::general, 9, "100000"},
    {0x1.e847fffbe76c8p+19, chars_format::general, 9, "999999.999"},
    {0x1.e847fffbe76c9p+19, chars_format::general, 9, "1000000"},
    {0x1.312cfffd70a3dp+23, chars_format::general, 9, "9999999.99"},
    {0x1.312cfffd70a3ep+23, chars_format::general, 9, "10000000"},
    {0x1.7d783ffccccccp+26, chars_format::general, 9, "99999999.9"},
    {0x1.7d783ffcccccdp+26, chars_format::general, 9, "100000000"},
    {0x1.dcd64ffbfffffp+29, chars_format::general, 9, "999999999"},
    {0x1.dcd64ffc00000p+29, chars_format::general, 9, "1e+09"},
    {0x1.a36e2eb16a205p-14, chars_format::general, 10, "9.999999999e-05"},
    {0x1.a36e2eb16a206p-14, chars_format::general, 10, "0.0001"},
    {0x1.0624dd2ee2543p-10, chars_format::general, 10, "0.0009999999999"},
    {0x1.0624dd2ee2544p-10, chars_format::general, 10, "0.001"},
    {0x1.47ae147a9ae94p-7, chars_format::general, 10, "0.009999999999"},
    {0x1.47ae147a9ae95p-7, chars_format::general, 10, "0.01"},
    {0x1.9999999941a39p-4, chars_format::general, 10, "0.09999999999"},
    {0x1.9999999941a3ap-4, chars_format::general, 10, "0.1"},
    {0x1.ffffffff920c8p-1, chars_format::general, 10, "0.9999999999"},
    {0x1.ffffffff920c9p-1, chars_format::general, 10, "1"},
    {0x1.3fffffffbb47dp+3, chars_format::general, 10, "9.999999999"},
    {0x1.3fffffffbb47ep+3, chars_format::general, 10, "10"},
    {0x1.8fffffffaa19cp+6, chars_format::general, 10, "99.99999999"},
    {0x1.8fffffffaa19dp+6, chars_format::general, 10, "100"},
    {0x1.f3ffffff94a03p+9, chars_format::general, 10, "999.9999999"},
    {0x1.f3ffffff94a04p+9, chars_format::general, 10, "1000"},
    {0x1.387fffffbce42p+13, chars_format::general, 10, "9999.999999"},
    {0x1.387fffffbce43p+13, chars_format::general, 10, "10000"},
    {0x1.869fffffac1d2p+16, chars_format::general, 10, "99999.99999"},
    {0x1.869fffffac1d3p+16, chars_format::general, 10, "100000"},
    {0x1.e847ffff97247p+19, chars_format::general, 10, "999999.9999"},
    {0x1.e847ffff97248p+19, chars_format::general, 10, "1000000"},
    {0x1.312cffffbe76cp+23, chars_format::general, 10, "9999999.999"},
    {0x1.312cffffbe76dp+23, chars_format::general, 10, "10000000"},
    {0x1.7d783fffae147p+26, chars_format::general, 10, "99999999.99"},
    {0x1.7d783fffae148p+26, chars_format::general, 10, "100000000"},
    {0x1.dcd64fff99999p+29, chars_format::general, 10, "999999999.9"},
    {0x1.dcd64fff9999ap+29, chars_format::general, 10, "1000000000"},
    {0x1.2a05f1ffbffffp+33, chars_format::general, 10, "9999999999"},
    {0x1.2a05f1ffc0000p+33, chars_format::general, 10, "1e+10"},
    {0x1.a36e2eb1bb30fp-14, chars_format::general, 11, "9.9999999999e-05"},
    {0x1.a36e2eb1bb310p-14, chars_format::general, 11, "0.0001"},
    {0x1.0624dd2f14fe9p-10, chars_format::general, 11, "0.00099999999999"},
    {0x1.0624dd2f14feap-10, chars_format::general, 11, "0.001"},
    {0x1.47ae147ada3e3p-7, chars_format::general, 11, "0.0099999999999"},
    {0x1.47ae147ada3e4p-7, chars_format::general, 11, "0.01"},
    {0x1.9999999990cdcp-4, chars_format::general, 11, "0.099999999999"},
    {0x1.9999999990cddp-4, chars_format::general, 11, "0.1"},
    {0x1.fffffffff5014p-1, chars_format::general, 11, "0.99999999999"},
    {0x1.fffffffff5015p-1, chars_format::general, 11, "1"},
    {0x1.3ffffffff920cp+3, chars_format::general, 11, "9.9999999999"},
    {0x1.3ffffffff920dp+3, chars_format::general, 11, "10"},
    {0x1.8ffffffff768fp+6, chars_format::general, 11, "99.999999999"},
    {0x1.8ffffffff7690p+6, chars_format::general, 11, "100"},
    {0x1.f3fffffff5433p+9, chars_format::general, 11, "999.99999999"},
    {0x1.f3fffffff5434p+9, chars_format::general, 11, "1000"},
    {0x1.387ffffff94a0p+13, chars_format::general, 11, "9999.9999999"},
    {0x1.387ffffff94a1p+13, chars_format::general, 11, "10000"},
    {0x1.869ffffff79c8p+16, chars_format::general, 11, "99999.999999"},
    {0x1.869ffffff79c9p+16, chars_format::general, 11, "100000"},
    {0x1.e847fffff583ap+19, chars_format::general, 11, "999999.99999"},
    {0x1.e847fffff583bp+19, chars_format::general, 11, "1000000"},
    {0x1.312cfffff9724p+23, chars_format::general, 11, "9999999.9999"},
    {0x1.312cfffff9725p+23, chars_format::general, 11, "10000000"},
    {0x1.7d783ffff7cedp+26, chars_format::general, 11, "99999999.999"},
    {0x1.7d783ffff7ceep+26, chars_format::general, 11, "100000000"},
    {0x1.dcd64ffff5c28p+29, chars_format::general, 11, "999999999.99"},
    {0x1.dcd64ffff5c29p+29, chars_format::general, 11, "1000000000"},
    {0x1.2a05f1fff9999p+33, chars_format::general, 11, "9999999999.9"},
    {0x1.2a05f1fff999ap+33, chars_format::general, 11, "10000000000"},
    {0x1.74876e7ff7fffp+36, chars_format::general, 11, "99999999999"},
    {0x1.74876e7ff8000p+36, chars_format::general, 11, "1e+11"},
    {0x1.a36e2eb1c34c3p-14, chars_format::general, 12, "9.99999999999e-05"},
    {0x1.a36e2eb1c34c4p-14, chars_format::general, 12, "0.0001"},
    {0x1.0624dd2f1a0fap-10, chars_format::general, 12, "0.000999999999999"},
    {0x1.0624dd2f1a0fbp-10, chars_format::general, 12, "0.001"},
    {0x1.47ae147ae0938p-7, chars_format::general, 12, "0.00999999999999"},
    {0x1.47ae147ae0939p-7, chars_format::general, 12, "0.01"},
    {0x1.9999999998b86p-4, chars_format::general, 12, "0.0999999999999"},
    {0x1.9999999998b87p-4, chars_format::general, 12, "0.1"},
    {0x1.fffffffffee68p-1, chars_format::general, 12, "0.999999999999"},
    {0x1.fffffffffee69p-1, chars_format::general, 12, "1"},
    {0x1.3fffffffff501p+3, chars_format::general, 12, "9.99999999999"},
    {0x1.3fffffffff502p+3, chars_format::general, 12, "10"},
    {0x1.8fffffffff241p+6, chars_format::general, 12, "99.9999999999"},
    {0x1.8fffffffff242p+6, chars_format::general, 12, "100"},
    {0x1.f3fffffffeed1p+9, chars_format::general, 12, "999.999999999"},
    {0x1.f3fffffffeed2p+9, chars_format::general, 12, "1000"},
    {0x1.387fffffff543p+13, chars_format::general, 12, "9999.99999999"},
    {0x1.387fffffff544p+13, chars_format::general, 12, "10000"},
    {0x1.869fffffff294p+16, chars_format::general, 12, "99999.9999999"},
    {0x1.869fffffff295p+16, chars_format::general, 12, "100000"},
    {0x1.e847fffffef39p+19, chars_format::general, 12, "999999.999999"},
    {0x1.e847fffffef3ap+19, chars_format::general, 12, "1000000"},
    {0x1.312cffffff583p+23, chars_format::general, 12, "9999999.99999"},
    {0x1.312cffffff584p+23, chars_format::general, 12, "10000000"},
    {0x1.7d783fffff2e4p+26, chars_format::general, 12, "99999999.9999"},
    {0x1.7d783fffff2e5p+26, chars_format::general, 12, "100000000"},
    {0x1.dcd64ffffef9dp+29, chars_format::general, 12, "999999999.999"},
    {0x1.dcd64ffffef9ep+29, chars_format::general, 12, "1000000000"},
    {0x1.2a05f1ffff5c2p+33, chars_format::general, 12, "9999999999.99"},
    {0x1.2a05f1ffff5c3p+33, chars_format::general, 12, "10000000000"},
    {0x1.74876e7fff333p+36, chars_format::general, 12, "99999999999.9"},
    {0x1.74876e7fff334p+36, chars_format::general, 12, "100000000000"},
    {0x1.d1a94a1ffefffp+39, chars_format::general, 12, "999999999999"},
    {0x1.d1a94a1fff000p+39, chars_format::general, 12, "1e+12"},
    {0x1.a36e2eb1c41bbp-14, chars_format::general, 13, "9.999999999999e-05"},
    {0x1.a36e2eb1c41bcp-14, chars_format::general, 13, "0.0001"},
    {0x1.0624dd2f1a915p-10, chars_format::general, 13, "0.0009999999999999"},
    {0x1.0624dd2f1a916p-10, chars_format::general, 13, "0.001"},
    {0x1.47ae147ae135ap-7, chars_format::general, 13, "0.009999999999999"},
    {0x1.47ae147ae135bp-7, chars_format::general, 13, "0.01"},
    {0x1.9999999999831p-4, chars_format::general, 13, "0.09999999999999"},
    {0x1.9999999999832p-4, chars_format::general, 13, "0.1"},
    {0x1.ffffffffffe3dp-1, chars_format::general, 13, "0.9999999999999"},
    {0x1.ffffffffffe3ep-1, chars_format::general, 13, "1"},
    {0x1.3fffffffffee6p+3, chars_format::general, 13, "9.999999999999"},
    {0x1.3fffffffffee7p+3, chars_format::general, 13, "10"},
    {0x1.8fffffffffea0p+6, chars_format::general, 13, "99.99999999999"},
    {0x1.8fffffffffea1p+6, chars_format::general, 13, "100"},
    {0x1.f3ffffffffe48p+9, chars_format::general, 13, "999.9999999999"},
    {0x1.f3ffffffffe49p+9, chars_format::general, 13, "1000"},
    {0x1.387fffffffeedp+13, chars_format::general, 13, "9999.999999999"},
    {0x1.387fffffffeeep+13, chars_format::general, 13, "10000"},
    {0x1.869fffffffea8p+16, chars_format::general, 13, "99999.99999999"},
    {0x1.869fffffffea9p+16, chars_format::general, 13, "100000"},
    {0x1.e847ffffffe52p+19, chars_format::general, 13, "999999.9999999"},
    {0x1.e847ffffffe53p+19, chars_format::general, 13, "1000000"},
    {0x1.312cffffffef3p+23, chars_format::general, 13, "9999999.999999"},
    {0x1.312cffffffef4p+23, chars_format::general, 13, "10000000"},
    {0x1.7d783fffffeb0p+26, chars_format::general, 13, "99999999.99999"},
    {0x1.7d783fffffeb1p+26, chars_format::general, 13, "100000000"},
    {0x1.dcd64fffffe5cp+29, chars_format::general, 13, "999999999.9999"},
    {0x1.dcd64fffffe5dp+29, chars_format::general, 13, "1000000000"},
    {0x1.2a05f1ffffef9p+33, chars_format::general, 13, "9999999999.999"},
    {0x1.2a05f1ffffefap+33, chars_format::general, 13, "10000000000"},
    {0x1.74876e7fffeb8p+36, chars_format::general, 13, "99999999999.99"},
    {0x1.74876e7fffeb9p+36, chars_format::general, 13, "100000000000"},
    {0x1.d1a94a1fffe66p+39, chars_format::general, 13, "999999999999.9"},
    {0x1.d1a94a1fffe67p+39, chars_format::general, 13, "1000000000000"},
    {0x1.2309ce53ffeffp+43, chars_format::general, 13, "9999999999999"},
    {0x1.2309ce53fff00p+43, chars_format::general, 13, "1e+13"},
    {0x1.a36e2eb1c4307p-14, chars_format::general, 14, "9.9999999999999e-05"},
    {0x1.a36e2eb1c4308p-14, chars_format::general, 14, "0.0001"},
    {0x1.0624dd2f1a9e4p-10, chars_format::general, 14, "0.00099999999999999"},
    {0x1.0624dd2f1a9e5p-10, chars_format::general, 14, "0.001"},
    {0x1.47ae147ae145ep-7, chars_format::general, 14, "0.0099999999999999"},
    {0x1.47ae147ae145fp-7, chars_format::general, 14, "0.01"},
    {0x1.9999999999975p-4, chars_format::general, 14, "0.099999999999999"},
    {0x1.9999999999976p-4, chars_format::general, 14, "0.1"},
    {0x1.fffffffffffd2p-1, chars_format::general, 14, "0.99999999999999"},
    {0x1.fffffffffffd3p-1, chars_format::general, 14, "1"},
    {0x1.3ffffffffffe3p+3, chars_format::general, 14, "9.9999999999999"},
    {0x1.3ffffffffffe4p+3, chars_format::general, 14, "10"},
    {0x1.8ffffffffffdcp+6, chars_format::general, 14, "99.999999999999"},
    {0x1.8ffffffffffddp+6, chars_format::general, 14, "100"},
    {0x1.f3fffffffffd4p+9, chars_format::general, 14, "999.99999999999"},
    {0x1.f3fffffffffd5p+9, chars_format::general, 14, "1000"},
    {0x1.387ffffffffe4p+13, chars_format::general, 14, "9999.9999999999"},
    {0x1.387ffffffffe5p+13, chars_format::general, 14, "10000"},
    {0x1.869ffffffffddp+16, chars_format::general, 14, "99999.999999999"},
    {0x1.869ffffffffdep+16, chars_format::general, 14, "100000"},
    {0x1.e847fffffffd5p+19, chars_format::general, 14, "999999.99999999"},
    {0x1.e847fffffffd6p+19, chars_format::general, 14, "1000000"},
    {0x1.312cfffffffe5p+23, chars_format::general, 14, "9999999.9999999"},
    {0x1.312cfffffffe6p+23, chars_format::general, 14, "10000000"},
    {0x1.7d783ffffffdep+26, chars_format::general, 14, "99999999.999999"},
    {0x1.7d783ffffffdfp+26, chars_format::general, 14, "100000000"},
    {0x1.dcd64ffffffd6p+29, chars_format::general, 14, "999999999.99999"},
    {0x1.dcd64ffffffd7p+29, chars_format::general, 14, "1000000000"},
    {0x1.2a05f1fffffe5p+33, chars_format::general, 14, "9999999999.9999"},
    {0x1.2a05f1fffffe6p+33, chars_format::general, 14, "10000000000"},
    {0x1.74876e7ffffdfp+36, chars_format::general, 14, "99999999999.999"},
    {0x1.74876e7ffffe0p+36, chars_format::general, 14, "100000000000"},
    {0x1.d1a94a1ffffd7p+39, chars_format::general, 14, "999999999999.99"},
    {0x1.d1a94a1ffffd8p+39, chars_format::general, 14, "1000000000000"},
    {0x1.2309ce53fffe6p+43, chars_format::general, 14, "9999999999999.9"},
    {0x1.2309ce53fffe7p+43, chars_format::general, 14, "10000000000000"},
    {0x1.6bcc41e8fffdfp+46, chars_format::general, 14, "99999999999999"},
    {0x1.6bcc41e8fffe0p+46, chars_format::general, 14, "1e+14"},
    {0x1.a36e2eb1c4328p-14, chars_format::general, 15, "9.99999999999999e-05"},
    {0x1.a36e2eb1c4329p-14, chars_format::general, 15, "0.0001"},
    {0x1.0624dd2f1a9f9p-10, chars_format::general, 15, "0.000999999999999999"},
    {0x1.0624dd2f1a9fap-10, chars_format::general, 15, "0.001"},
    {0x1.47ae147ae1477p-7, chars_format::general, 15, "0.00999999999999999"},
    {0x1.47ae147ae1478p-7, chars_format::general, 15, "0.01"},
    {0x1.9999999999995p-4, chars_format::general, 15, "0.0999999999999999"},
    {0x1.9999999999996p-4, chars_format::general, 15, "0.1"},
    {0x1.ffffffffffffbp-1, chars_format::general, 15, "0.999999999999999"},
    {0x1.ffffffffffffcp-1, chars_format::general, 15, "1"},
    {0x1.3fffffffffffdp+3, chars_format::general, 15, "9.99999999999999"},
    {0x1.3fffffffffffep+3, chars_format::general, 15, "10"},
    {0x1.8fffffffffffcp+6, chars_format::general, 15, "99.9999999999999"},
    {0x1.8fffffffffffdp+6, chars_format::general, 15, "100"},
    {0x1.f3ffffffffffbp+9, chars_format::general, 15, "999.999999999999"},
    {0x1.f3ffffffffffcp+9, chars_format::general, 15, "1000"},
    {0x1.387fffffffffdp+13, chars_format::general, 15, "9999.99999999999"},
    {0x1.387fffffffffep+13, chars_format::general, 15, "10000"},
    {0x1.869fffffffffcp+16, chars_format::general, 15, "99999.9999999999"},
    {0x1.869fffffffffdp+16, chars_format::general, 15, "100000"},
    {0x1.e847ffffffffbp+19, chars_format::general, 15, "999999.999999999"},
    {0x1.e847ffffffffcp+19, chars_format::general, 15, "1000000"},
    {0x1.312cffffffffdp+23, chars_format::general, 15, "9999999.99999999"},
    {0x1.312cffffffffep+23, chars_format::general, 15, "10000000"},
    {0x1.7d783fffffffcp+26, chars_format::general, 15, "99999999.9999999"},
    {0x1.7d783fffffffdp+26, chars_format::general, 15, "100000000"},
    {0x1.dcd64fffffffbp+29, chars_format::general, 15, "999999999.999999"},
    {0x1.dcd64fffffffcp+29, chars_format::general, 15, "1000000000"},
    {0x1.2a05f1ffffffdp+33, chars_format::general, 15, "9999999999.99999"},
    {0x1.2a05f1ffffffep+33, chars_format::general, 15, "10000000000"},
    {0x1.74876e7fffffcp+36, chars_format::general, 15, "99999999999.9999"},
    {0x1.74876e7fffffdp+36, chars_format::general, 15, "100000000000"},
    {0x1.d1a94a1fffffbp+39, chars_format::general, 15, "999999999999.999"},
    {0x1.d1a94a1fffffcp+39, chars_format::general, 15, "1000000000000"},
    {0x1.2309ce53ffffdp+43, chars_format::general, 15, "9999999999999.99"},
    {0x1.2309ce53ffffep+43, chars_format::general, 15, "10000000000000"},
    {0x1.6bcc41e8ffffcp+46, chars_format::general, 15, "99999999999999.9"},
    {0x1.6bcc41e8ffffdp+46, chars_format::general, 15, "100000000000000"},
    {0x1.c6bf52633fffbp+49, chars_format::general, 15, "999999999999999"},
    {0x1.c6bf52633fffcp+49, chars_format::general, 15, "1e+15"},
    {0x1.1c37937e07fffp+53, chars_format::general, 16, "9999999999999998"},
    {0x1.1c37937e08000p+53, chars_format::general, 16, "1e+16"},
    {0x1.6345785d89fffp+56, chars_format::general, 17, "99999999999999984"},
    {0x1.6345785d8a000p+56, chars_format::general, 17, "1e+17"},
    {0x1.bc16d674ec7ffp+59, chars_format::general, 18, "999999999999999872"},
    {0x1.bc16d674ec800p+59, chars_format::general, 18, "1e+18"},
    {0x1.158e460913cffp+63, chars_format::general, 19, "9999999999999997952"},
    {0x1.158e460913d00p+63, chars_format::general, 19, "1e+19"},
    {0x1.5af1d78b58c3fp+66, chars_format::general, 20, "99999999999999983616"},
    {0x1.5af1d78b58c40p+66, chars_format::general, 20, "1e+20"},
    {0x1.b1ae4d6e2ef4fp+69, chars_format::general, 21, "999999999999999868928"},
    {0x1.b1ae4d6e2ef50p+69, chars_format::general, 21, "1e+21"},
    {0x1.0f0cf064dd591p+73, chars_format::general, 22, "9999999999999997902848"},
    {0x1.0f0cf064dd592p+73, chars_format::general, 22, "1e+22"},
    {0x1.52d02c7e14af6p+76, chars_format::general, 23, "99999999999999991611392"},
    {0x1.52d02c7e14af7p+76, chars_format::general, 23, "1.0000000000000000838861e+23"},
    {0x1.a784379d99db4p+79, chars_format::general, 24, "999999999999999983222784"},
    {0x1.a784379d99db5p+79, chars_format::general, 24, "1.00000000000000011744051e+24"},
    {0x1.a36e2eb1c432cp-14, chars_format::general, 25, "9.999999999999999123964645e-05"},
    {0x1.a36e2eb1c432dp-14, chars_format::general, 25, "0.0001000000000000000047921736"},
    {0x1.0624dd2f1a9fbp-10, chars_format::general, 25, "0.0009999999999999998039762472"},
    {0x1.0624dd2f1a9fcp-10, chars_format::general, 25, "0.001000000000000000020816682"},
    {0x1.47ae147ae147ap-7, chars_format::general, 25, "0.009999999999999998473443341"},
    {0x1.47ae147ae147bp-7, chars_format::general, 25, "0.01000000000000000020816682"},
    {0x1.9999999999999p-4, chars_format::general, 25, "0.09999999999999999167332732"},
    {0x1.999999999999ap-4, chars_format::general, 25, "0.1000000000000000055511151"},
    {0x1.fffffffffffffp-1, chars_format::general, 25, "0.9999999999999998889776975"},
    {0x1.0000000000000p+0, chars_format::general, 25, "1"},
    {0x1.3ffffffffffffp+3, chars_format::general, 25, "9.999999999999998223643161"},
    {0x1.4000000000000p+3, chars_format::general, 25, "10"},
    {0x1.8ffffffffffffp+6, chars_format::general, 25, "99.99999999999998578914528"},
    {0x1.9000000000000p+6, chars_format::general, 25, "100"},
    {0x1.f3fffffffffffp+9, chars_format::general, 25, "999.9999999999998863131623"},
    {0x1.f400000000000p+9, chars_format::general, 25, "1000"},
    {0x1.387ffffffffffp+13, chars_format::general, 25, "9999.999999999998181010596"},
    {0x1.3880000000000p+13, chars_format::general, 25, "10000"},
    {0x1.869ffffffffffp+16, chars_format::general, 25, "99999.99999999998544808477"},
    {0x1.86a0000000000p+16, chars_format::general, 25, "100000"},
    {0x1.e847fffffffffp+19, chars_format::general, 25, "999999.9999999998835846782"},
    {0x1.e848000000000p+19, chars_format::general, 25, "1000000"},
    {0x1.312cfffffffffp+23, chars_format::general, 25, "9999999.999999998137354851"},
    {0x1.312d000000000p+23, chars_format::general, 25, "10000000"},
    {0x1.7d783ffffffffp+26, chars_format::general, 25, "99999999.99999998509883881"},
    {0x1.7d78400000000p+26, chars_format::general, 25, "100000000"},
    {0x1.dcd64ffffffffp+29, chars_format::general, 25, "999999999.9999998807907104"},
    {0x1.dcd6500000000p+29, chars_format::general, 25, "1000000000"},
    {0x1.2a05f1fffffffp+33, chars_format::general, 25, "9999999999.999998092651367"},
    {0x1.2a05f20000000p+33, chars_format::general, 25, "10000000000"},
    {0x1.74876e7ffffffp+36, chars_format::general, 25, "99999999999.99998474121094"},
    {0x1.74876e8000000p+36, chars_format::general, 25, "100000000000"},
    {0x1.d1a94a1ffffffp+39, chars_format::general, 25, "999999999999.9998779296875"},
    {0x1.d1a94a2000000p+39, chars_format::general, 25, "1000000000000"},
    {0x1.2309ce53fffffp+43, chars_format::general, 25, "9999999999999.998046875"},
    {0x1.2309ce5400000p+43, chars_format::general, 25, "10000000000000"},
    {0x1.6bcc41e8fffffp+46, chars_format::general, 25, "99999999999999.984375"},
    {0x1.6bcc41e900000p+46, chars_format::general, 25, "100000000000000"},
    {0x1.c6bf52633ffffp+49, chars_format::general, 25, "999999999999999.875"},
    {0x1.c6bf526340000p+49, chars_format::general, 25, "1000000000000000"},
    {0x1.1c37937e07fffp+53, chars_format::general, 25, "9999999999999998"},
    {0x1.1c37937e08000p+53, chars_format::general, 25, "10000000000000000"},
    {0x1.6345785d89fffp+56, chars_format::general, 25, "99999999999999984"},
    {0x1.6345785d8a000p+56, chars_format::general, 25, "100000000000000000"},
    {0x1.bc16d674ec7ffp+59, chars_format::general, 25, "999999999999999872"},
    {0x1.bc16d674ec800p+59, chars_format::general, 25, "1000000000000000000"},
    {0x1.158e460913cffp+63, chars_format::general, 25, "9999999999999997952"},
    {0x1.158e460913d00p+63, chars_format::general, 25, "10000000000000000000"},
    {0x1.5af1d78b58c3fp+66, chars_format::general, 25, "99999999999999983616"},
    {0x1.5af1d78b58c40p+66, chars_format::general, 25, "100000000000000000000"},
    {0x1.b1ae4d6e2ef4fp+69, chars_format::general, 25, "999999999999999868928"},
    {0x1.b1ae4d6e2ef50p+69, chars_format::general, 25, "1000000000000000000000"},
    {0x1.0f0cf064dd591p+73, chars_format::general, 25, "9999999999999997902848"},
    {0x1.0f0cf064dd592p+73, chars_format::general, 25, "10000000000000000000000"},
    {0x1.52d02c7e14af6p+76, chars_format::general, 25, "99999999999999991611392"},
    {0x1.52d02c7e14af7p+76, chars_format::general, 25, "100000000000000008388608"},
    {0x1.a784379d99db4p+79, chars_format::general, 25, "999999999999999983222784"},
    {0x1.a784379d99db5p+79, chars_format::general, 25, "1000000000000000117440512"},
    {0x1.08b2a2c280290p+83, chars_format::general, 25, "9999999999999998758486016"},
    {0x1.08b2a2c280291p+83, chars_format::general, 25, "1.000000000000000090596966e+25"},
    {0x1.4adf4b7320334p+86, chars_format::general, 26, "99999999999999987584860160"},
    {0x1.4adf4b7320335p+86, chars_format::general, 26, "1.0000000000000000476472934e+26"},
    {0x1.9d971e4fe8401p+89, chars_format::general, 27, "999999999999999875848601600"},
    {0x1.9d971e4fe8402p+89, chars_format::general, 27, "1.00000000000000001328755507e+27"},
    {0x1.027e72f1f1281p+93, chars_format::general, 28, "9999999999999999583119736832"},
    {0x1.027e72f1f1282p+93, chars_format::general, 28, "1.000000000000000178214299238e+28"},
    {0x1.431e0fae6d721p+96, chars_format::general, 29, "99999999999999991433150857216"},
    {0x1.431e0fae6d722p+96, chars_format::general, 29, "1.0000000000000000902533690163e+29"},
    {0x1.93e5939a08ce9p+99, chars_format::general, 30, "999999999999999879147136483328"},
    {0x1.93e5939a08ceap+99, chars_format::general, 30, "1.00000000000000001988462483866e+30"},
    {0x1.f8def8808b024p+102, chars_format::general, 31, "9999999999999999635896294965248"},
    {0x1.f8def8808b025p+102, chars_format::general, 31, "1.000000000000000076179620180787e+31"},
    {0x1.3b8b5b5056e16p+106, chars_format::general, 32, "99999999999999987351763694911488"},
    {0x1.3b8b5b5056e17p+106, chars_format::general, 32, "1.0000000000000000536616220439347e+32"},
    {0x1.8a6e32246c99cp+109, chars_format::general, 33, "999999999999999945575230987042816"},
    {0x1.8a6e32246c99dp+109, chars_format::general, 33, "1.00000000000000008969041906289869e+33"},
    {0x1.ed09bead87c03p+112, chars_format::general, 34, "9999999999999999455752309870428160"},
    {0x1.ed09bead87c04p+112, chars_format::general, 34, "1.000000000000000060867381447727514e+34"},
    {0x1.3426172c74d82p+116, chars_format::general, 35, "99999999999999996863366107917975552"},
    {0x1.3426172c74d83p+116, chars_format::general, 35, "1.0000000000000001531011018162752717e+35"},
    {0x1.812f9cf7920e2p+119, chars_format::general, 36, "999999999999999894846684784341549056"},
    {0x1.812f9cf7920e3p+119, chars_format::general, 36, "1.00000000000000004242063737401796198e+36"},
    {0x1.e17b84357691bp+122, chars_format::general, 37, "9999999999999999538762658202121142272"},
    {0x1.e17b84357691cp+122, chars_format::general, 37, "1.00000000000000007193542789195324457e+37"},
    {0x1.2ced32a16a1b1p+126, chars_format::general, 38, "99999999999999997748809823456034029568"},
    {0x1.2ced32a16a1b2p+126, chars_format::general, 38, "1.0000000000000001663827575493461488435e+38"},
    {0x1.78287f49c4a1dp+129, chars_format::general, 39, "999999999999999939709166371603178586112"},
    {0x1.78287f49c4a1ep+129, chars_format::general, 39, "1.00000000000000009082489382343182542438e+39"},
    {0x1.d6329f1c35ca4p+132, chars_format::general, 40, "9999999999999999094860208812374492184576"},
    {0x1.d6329f1c35ca5p+132, chars_format::general, 40, "1.000000000000000030378602842700366689075e+40"},
    {0x1.25dfa371a19e6p+136, chars_format::general, 41, "99999999999999981277195531206711524196352"},
    {0x1.25dfa371a19e7p+136, chars_format::general, 41, "1.0000000000000000062000864504077831949517e+41"},
    {0x1.6f578c4e0a060p+139, chars_format::general, 42, "999999999999999890143207767403382423158784"},
    {0x1.6f578c4e0a061p+139, chars_format::general, 42, "1.00000000000000004488571267807591678554931e+42"},
    {0x1.cb2d6f618c878p+142, chars_format::general, 43, "9999999999999998901432077674033824231587840"},
    {0x1.cb2d6f618c879p+142, chars_format::general, 43, "1.000000000000000013937211695941409913071206e+43"},
    {0x1.1efc659cf7d4bp+146, chars_format::general, 44, "99999999999999989014320776740338242315878400"},
    {0x1.1efc659cf7d4cp+146, chars_format::general, 44, "1.0000000000000000882136140530642264070186598e+44"},
    {0x1.66bb7f0435c9ep+149, chars_format::general, 45, "999999999999999929757289024535551219930759168"},
    {0x1.66bb7f0435c9fp+149, chars_format::general, 45, "1.00000000000000008821361405306422640701865984e+45"},
    {0x1.c06a5ec5433c6p+152, chars_format::general, 46, "9999999999999999931398190359470212947659194368"},
    {0x1.c06a5ec5433c7p+152, chars_format::general, 46, "1.000000000000000119904879058769961444436239974e+46"},
    {0x1.18427b3b4a05bp+156, chars_format::general, 47, "99999999999999984102174700855949311516153479168"},
    {0x1.18427b3b4a05cp+156, chars_format::general, 47, "1.0000000000000000438458430450761973546340476518e+47"},
    {0x1.5e531a0a1c872p+159, chars_format::general, 48, "999999999999999881586566215862833963056037363712"},
    {0x1.5e531a0a1c873p+159, chars_format::general, 48, "1.00000000000000004384584304507619735463404765184e+48"},
    {0x1.b5e7e08ca3a8fp+162, chars_format::general, 49, "9999999999999999464902769475481793196872414789632"},
    {0x1.b5e7e08ca3a90p+162, chars_format::general, 49, "1.000000000000000076297698410918870032949649709466e+49"},
    {0x1.11b0ec57e6499p+166, chars_format::general, 50, "99999999999999986860582406952576489172979654066176"},
    {0x1.11b0ec57e649ap+166, chars_format::general, 50, "1.0000000000000000762976984109188700329496497094656e+50"},
    {0x1.561d276ddfdc0p+169, chars_format::general, 51, "999999999999999993220948674361627976461708441944064"},
    {0x1.561d276ddfdc1p+169, chars_format::general, 51, "1.00000000000000015937444814747611208943759097698714e+51"},
    {0x1.aba4714957d30p+172, chars_format::general, 52, "9999999999999999932209486743616279764617084419440640"},
    {0x1.aba4714957d31p+172, chars_format::general, 52, "1.000000000000000126143748252853215266842414469978522e+52"},
    {0x1.0b46c6cdd6e3ep+176, chars_format::general, 53, "99999999999999999322094867436162797646170844194406400"},
    {0x1.0b46c6cdd6e3fp+176, chars_format::general, 53, "1.0000000000000002058974279999481676410708380867991962e+53"},
    {0x1.4e1878814c9cdp+179, chars_format::general, 54, "999999999999999908150356944127012110618056584002011136"},
    {0x1.4e1878814c9cep+179, chars_format::general, 54, "1.00000000000000007829154040459624384230536029988611686e+54"},
    {0x1.a19e96a19fc40p+182, chars_format::general, 55, "9999999999999998741221202520331657642805958408251899904"},
    {0x1.a19e96a19fc41p+182, chars_format::general, 55, "1.000000000000000010235067020408551149630438813532474573e+55"},
    {0x1.05031e2503da8p+186, chars_format::general, 56, "99999999999999987412212025203316576428059584082518999040"},
    {0x1.05031e2503da9p+186, chars_format::general, 56,
        "1.0000000000000000919028350814337823808403445971568453222e+56"},
    {0x1.4643e5ae44d12p+189, chars_format::general, 57, "999999999999999874122120252033165764280595840825189990400"},
    {0x1.4643e5ae44d13p+189, chars_format::general, 57,
        "1.00000000000000004834669211555365905752839484589051425587e+57"},
    {0x1.97d4df19d6057p+192, chars_format::general, 58, "9999999999999999438119489974413630815797154428513196965888"},
    {0x1.97d4df19d6058p+192, chars_format::general, 58,
        "1.000000000000000083191606488257757716177954646903579108966e+58"},
    {0x1.fdca16e04b86dp+195, chars_format::general, 59, "99999999999999997168788049560464200849936328366177157906432"},
    {0x1.fdca16e04b86ep+195, chars_format::general, 59,
        "1.0000000000000000831916064882577577161779546469035791089664e+59"},
    {0x1.3e9e4e4c2f344p+199, chars_format::general, 60, "999999999999999949387135297074018866963645011013410073083904"},
    {0x1.3e9e4e4c2f345p+199, chars_format::general, 60,
        "1.00000000000000012779309688531900399924939119220030212092723e+60"},
    {0x1.8e45e1df3b015p+202, chars_format::general, 61,
        "9999999999999999493871352970740188669636450110134100730839040"},
    {0x1.8e45e1df3b016p+202, chars_format::general, 61,
        "1.000000000000000092111904567670006972792241955962923711358566e+61"},
    {0x1.f1d75a5709c1ap+205, chars_format::general, 62,
        "99999999999999992084218144295482124579792562202350734542897152"},
    {0x1.f1d75a5709c1bp+205, chars_format::general, 62,
        "1.0000000000000000350219968594316117304608031779831182560487014e+62"},
    {0x1.3726987666190p+209, chars_format::general, 63,
        "999999999999999875170255276364105051932774599639662981181079552"},
    {0x1.3726987666191p+209, chars_format::general, 63,
        "1.00000000000000005785795994272696982739337868917504043817264742e+63"},
    {0x1.84f03e93ff9f4p+212, chars_format::general, 64,
        "9999999999999998751702552763641050519327745996396629811810795520"},
    {0x1.84f03e93ff9f5p+212, chars_format::general, 64,
        "1.00000000000000002132041900945439687230125787126796494677433385e+64"},
    {0x1.e62c4e38ff872p+215, chars_format::general, 65,
        "99999999999999999209038626283633850822756121694230455365568299008"},
    {0x1.e62c4e38ff873p+215, chars_format::general, 65,
        "1.0000000000000001090105172493085719645223478342449461261302864282e+65"},
    {0x1.2fdbb0e39fb47p+219, chars_format::general, 66,
        "999999999999999945322333868247445125709646570021247924665841614848"},
    {0x1.2fdbb0e39fb48p+219, chars_format::general, 66,
        "1.00000000000000013239454344660301865578130515770547444062520711578e+66"},
    {0x1.7bd29d1c87a19p+222, chars_format::general, 67,
        "9999999999999999827367757839185598317239782875580932278577147150336"},
    {0x1.7bd29d1c87a1ap+222, chars_format::general, 67,
        "1.000000000000000132394543446603018655781305157705474440625207115776e+67"},
    {0x1.dac74463a989fp+225, chars_format::general, 68,
        "99999999999999995280522225138166806691251291352861698530421623488512"},
    {0x1.dac74463a98a0p+225, chars_format::general, 68,
        "1.000000000000000072531436381529235126158374409646521955518210155479e+68"},
    {0x1.28bc8abe49f63p+229, chars_format::general, 69,
        "999999999999999880969493773293127831364996015857874003175819882528768"},
    {0x1.28bc8abe49f64p+229, chars_format::general, 69,
        "1.00000000000000007253143638152923512615837440964652195551821015547904e+69"},
    {0x1.72ebad6ddc73cp+232, chars_format::general, 70,
        "9999999999999999192818822949403492903236716946156035936442979371188224"},
    {0x1.72ebad6ddc73dp+232, chars_format::general, 70,
        "1.00000000000000007253143638152923512615837440964652195551821015547904e+70"},
    {0x1.cfa698c95390bp+235, chars_format::general, 71,
        "99999999999999991928188229494034929032367169461560359364429793711882240"},
    {0x1.cfa698c95390cp+235, chars_format::general, 71,
        "1.0000000000000000418815255642114579589914338666403382831434277118069965e+71"},
    {0x1.21c81f7dd43a7p+239, chars_format::general, 72,
        "999999999999999943801810948794571024057224129020550531544123892056457216"},
    {0x1.21c81f7dd43a8p+239, chars_format::general, 72,
        "1.00000000000000013996124017962834489392564360426012603474273153155753574e+72"},
    {0x1.6a3a275d49491p+242, chars_format::general, 73,
        "9999999999999999830336967949613257980309080240684656321838454199566729216"},
    {0x1.6a3a275d49492p+242, chars_format::general, 73,
        "1.000000000000000139961240179628344893925643604260126034742731531557535744e+73"},
    {0x1.c4c8b1349b9b5p+245, chars_format::general, 74,
        "99999999999999995164818811802792197885196090803013355167206819763650035712"},
    {0x1.c4c8b1349b9b6p+245, chars_format::general, 74,
        "1.000000000000000077190222825761537255567749372183461873719177086917190615e+74"},
    {0x1.1afd6ec0e1411p+249, chars_format::general, 75,
        "999999999999999926539781176481198923508803215199467887262646419780362305536"},
    {0x1.1afd6ec0e1412p+249, chars_format::general, 75,
        "1.00000000000000012740703670885498336625406475784479320253802064262946671821e+75"},
    {0x1.61bcca7119915p+252, chars_format::general, 76,
        "9999999999999998863663300700064420349597509066704028242075715752105414230016"},
    {0x1.61bcca7119916p+252, chars_format::general, 76,
        "1.000000000000000047060134495905469589155960140786663076427870953489824953139e+76"},
    {0x1.ba2bfd0d5ff5bp+255, chars_format::general, 77,
        "99999999999999998278261272554585856747747644714015897553975120217811154108416"},
    {0x1.ba2bfd0d5ff5cp+255, chars_format::general, 77,
        "1.0000000000000001113376562662650806108344438344331671773159907048015383651942e+77"},
    {0x1.145b7e285bf98p+259, chars_format::general, 78,
        "999999999999999802805551768538947706777722104929947493053015898505313987330048"},
    {0x1.145b7e285bf99p+259, chars_format::general, 78,
        "1.00000000000000000849362143368970297614886992459876061589499910270279690590618e+78"},
    {0x1.59725db272f7fp+262, chars_format::general, 79,
        "9999999999999999673560075006595519222746403606649979913266024618633003221909504"},
    {0x1.59725db272f80p+262, chars_format::general, 79,
        "1.000000000000000131906463232780156137771558616400048489600189025221286657051853e+79"},
    {0x1.afcef51f0fb5ep+265, chars_format::general, 80,
        "99999999999999986862573406138718939297648940722396769236245052384850852127440896"},
    {0x1.afcef51f0fb5fp+265, chars_format::general, 80,
        "1.0000000000000000002660986470836727653740240118120080909813197745348975891631309e+80"},
    {0x1.0de1593369d1bp+269, chars_format::general, 81,
        "999999999999999921281879895665782741935503249059183851809998224123064148429897728"},
    {0x1.0de1593369d1cp+269, chars_format::general, 81,
        "1.0000000000000001319064632327801561377715586164000484896001890252212866570518528e+81"},
    {0x1.5159af8044462p+272, chars_format::general, 82,
        "9999999999999999634067965630886574211027143225273567793680363843427086501542887424"},
    {0x1.5159af8044463p+272, chars_format::general, 82,
        "1.0000000000000001319064632327801561377715586164000484896001890252212866570518528e+82"},
    {0x1.a5b01b605557ap+275, chars_format::general, 83,
        "99999999999999989600692989521205793443517660497828009527517532799127744739526311936"},
    {0x1.a5b01b605557bp+275, chars_format::general, 83,
        "1.0000000000000000308066632309652569077702520400764334634608974406941398529133143654e+83"},
    {0x1.078e111c3556cp+279, chars_format::general, 84,
        "999999999999999842087036560910778345101146430939018748000886482910132485188042620928"},
    {0x1.078e111c3556dp+279, chars_format::general, 84,
        "1.00000000000000005776660989811589670243726712709606413709804186323471233401692461466e+84"},
    {0x1.4971956342ac7p+282, chars_format::general, 85,
        "9999999999999998420870365609107783451011464309390187480008864829101324851880426209280"},
    {0x1.4971956342ac8p+282, chars_format::general, 85,
        "1.00000000000000001463069523067487303097004298786465505927861078716979636425114821591e+85"},
    {0x1.9bcdfabc13579p+285, chars_format::general, 86,
        "99999999999999987659576829486359728227492574232414601025643134376206526100066373992448"},
    {0x1.9bcdfabc1357ap+285, chars_format::general, 86,
        "1.0000000000000000146306952306748730309700429878646550592786107871697963642511482159104e+86"},
    {0x1.0160bcb58c16cp+289, chars_format::general, 87,
        "999999999999999959416724456350362731491996089648451439669739009806703922950954425516032"},
    {0x1.0160bcb58c16dp+289, chars_format::general, 87,
        "1.0000000000000001802726075536484039294041836825132659181052261192590736881517295870935e+87"},
    {0x1.41b8ebe2ef1c7p+292, chars_format::general, 88,
        "9999999999999999594167244563503627314919960896484514396697390098067039229509544255160320"},
    {0x1.41b8ebe2ef1c8p+292, chars_format::general, 88,
        "1.00000000000000013610143093418879568982174616394030302241812869736859973511157455477801e+88"},
    {0x1.922726dbaae39p+295, chars_format::general, 89,
        "99999999999999999475366575191804932315794610450682175621941694731908308538307845136842752"},
    {0x1.922726dbaae3ap+295, chars_format::general, 89,
        "1.0000000000000001361014309341887956898217461639403030224181286973685997351115745547780096e+89"},
    {0x1.f6b0f092959c7p+298, chars_format::general, 90,
        "999999999999999966484112715463900049825186092620125502979674597309179755437379230686511104"},
    {0x1.f6b0f092959c8p+298, chars_format::general, 90,
        "1.00000000000000007956232486128049714315622614016691051593864399734879307522017611341417677e+90"},
    {0x1.3a2e965b9d81cp+302, chars_format::general, 91,
        "9999999999999998986371854279739417938265620640920544952042929572854117635677011010499117056"},
    {0x1.3a2e965b9d81dp+302, chars_format::general, 91,
        "1.000000000000000079562324861280497143156226140166910515938643997348793075220176113414176768e+91"},
    {0x1.88ba3bf284e23p+305, chars_format::general, 92,
        "99999999999999989863718542797394179382656206409205449520429295728541176356770110104991170560"},
    {0x1.88ba3bf284e24p+305, chars_format::general, 92,
        "1.0000000000000000433772969746191860732902933249519393117917737893361168128896811109413237555e+92"},
    {0x1.eae8caef261acp+308, chars_format::general, 93,
        "999999999999999927585207737302990649719308316264031458521789123695552773432097103028194115584"},
    {0x1.eae8caef261adp+308, chars_format::general, 93,
        "1.00000000000000004337729697461918607329029332495193931179177378933611681288968111094132375552e+93"},
    {0x1.32d17ed577d0bp+312, chars_format::general, 94,
        "9999999999999998349515363474500343108625203093137051759058013911831015418660298966976904036352"},
    {0x1.32d17ed577d0cp+312, chars_format::general, 94,
        "1.000000000000000020218879127155946988576096323214357741137776856208004004998164309358697827533e+94"},
    {0x1.7f85de8ad5c4ep+315, chars_format::general, 95,
        "99999999999999987200500490339121684640523551209383568895219648418808203449245677922989188841472"},
    {0x1.7f85de8ad5c4fp+315, chars_format::general, 95,
        "1.0000000000000000202188791271559469885760963232143577411377768562080040049981643093586978275328e+95"},
    {0x1.df67562d8b362p+318, chars_format::general, 96,
        "999999999999999931290554592897108903273579836542044509826428632996050822694739791281414264061952"},
    {0x1.df67562d8b363p+318, chars_format::general, 96,
        "1.00000000000000004986165397190889301701026848543846215157489293061198839909930581538445901535642e+96"},
    {0x1.2ba095dc7701dp+322, chars_format::general, 97,
        "9999999999999998838621148412923952577789043769834774531270429139496757921329133816401963635441664"},
    {0x1.2ba095dc7701ep+322, chars_format::general, 97,
        "1.000000000000000073575873847711249839757606215217745679924585790135175914380219020205067965615309e+97"},
    {0x1.7688bb5394c25p+325, chars_format::general, 98,
        "99999999999999999769037024514370800696612547992403838920556863966097586548129676477911932478685184"},
    {0x1.7688bb5394c26p+325, chars_format::general, 98,
        "1.0000000000000001494613774502787916725490869505114529706436029406093759632791412756310166064437658e+98"},
    {0x1.d42aea2879f2ep+328, chars_format::general, 99,
        "999999999999999967336168804116691273849533185806555472917961779471295845921727862608739868455469056"},
    {0x1.d42aea2879f2fp+328, chars_format::general, 99,
        "1.00000000000000008875297456822475820631590236227648713806838922023001592416000347129025769378100019e+99"},
    {0x1.249ad2594c37cp+332, chars_format::general, 100,
        "9999999999999998216360018871870109548898901740426374747374488505608317520357971321909184780648316928"},
    {0x1.249ad2594c37dp+332, chars_format::general, 100,
        "1.00000000000000001590289110975991804683608085639452813897813275577478387721703810608134699858568151e+100"},
    {0x1.6dc186ef9f45cp+335, chars_format::general, 101,
        "99999999999999997704951326524533662844684271992415000612999597473199345218078991130326129448151154688"},
    {0x1.6dc186ef9f45dp+335, chars_format::general, 101,
        "1.000000000000000132463024643303662302003795265805662537522543098903155152325782690415604110898191401e+101"},
    {0x1.c931e8ab87173p+338, chars_format::general, 102,
        "999999999999999977049513265245336628446842719924150006129995974731993452180789911303261294481511546880"},
    {0x1.c931e8ab87174p+338, chars_format::general, 102,
        "1.00000000000000010138032236769199716729240475662936003124403367406892281229678413459313554761485543014e+102"},
    {0x1.1dbf316b346e7p+342, chars_format::general, 103,
        "9999999999999998029863805218200118740630558685368559709703431956602923480183979986974373400948301103104"},
    {0x1.1dbf316b346e8p+342, chars_format::general, 103,
        "1.000000000000000001915675085734668736215955127265192011152803514599379324203988755961236145108180323533e+"
        "103"},
    {0x1.652efdc6018a1p+345, chars_format::general, 104,
        "99999999999999984277223943460294324649363572028252317900683525944810974325551615015019710109750015295488"},
    {0x1.652efdc6018a2p+345, chars_format::general, 104,
        "1.0000000000000000019156750857346687362159551272651920111528035145993793242039887559612361451081803235328e+"
        "104"},
    {0x1.be7abd3781ecap+348, chars_format::general, 105,
        "999999999999999938258300825281978540327027364472124478294416212538871491824599713636820527503908255301632"},
    {0x1.be7abd3781ecbp+348, chars_format::general, 105,
        "1.00000000000000006557304934618735893210488289005825954401119081665988715658337779828565176271245239176397e+"
        "105"},
    {0x1.170cb642b133ep+352, chars_format::general, 106,
        "9999999999999998873324014169198263836158851542376704520077063708904652259210884797772880334204906007166976"},
    {0x1.170cb642b133fp+352, chars_format::general, 106,
        "1.000000000000000091035999050368435010460453995175486557154545737484090289535133415215418009754161219056435e+"
        "106"},
    {0x1.5ccfe3d35d80ep+355, chars_format::general, 107,
        "99999999999999996881384047029926983435371269061279689406644211752791525136670645395254002395395884805259264"},
    {0x1.5ccfe3d35d80fp+355, chars_format::general, 107,
        "1.0000000000000001317767185770581567358293677633630497781839136108028153022579424023030440050208953427243827e+"
        "107"},
    {0x1.b403dcc834e11p+358, chars_format::general, 108,
        "999999999999999903628689227595715073763450661512695740419453520217955231010212074612338431527184250183876608"},
    {0x1.b403dcc834e12p+358, chars_format::general, 108,
        "1."
        "00000000000000003399899171300282459494397471971289804771343071483787527172320083329274161638073344592130867e+"
        "108"},
    {0x1.108269fd210cbp+362, chars_format::general, 109,
        "999999999999999981850870718839980786471765096432817124795839836989907255438005329820580342439313767626335846"
        "4"},
    {0x1.108269fd210ccp+362, chars_format::general, 109,
        "1."
        "000000000000000190443354695491356020360603589553140816466203348381779320578787343709225438204992480806227149e+"
        "109"},
    {0x1.54a3047c694fdp+365, chars_format::general, 110,
        "9999999999999998566953803328491556461384620005606229097936217301547840163535361214873932849799065397184010649"
        "6"},
    {0x1.54a3047c694fep+365, chars_format::general, 110,
        "1."
        "0000000000000000235693675141702558332495327950568818631299125392682816684661617325983093615924495102623141069e"
        "+110"},
    {0x1.a9cbc59b83a3dp+368, chars_format::general, 111,
        "99999999999999995681977264164181575840510447725837828179539621562288260762111148815394293094743232204474889011"
        "2"},
    {0x1.a9cbc59b83a3ep+368, chars_format::general, 111,
        "1."
        "00000000000000009031896238669869590809396111285538544446442886291368072931121197704267579223746669847987932365"
        "e+111"},
    {0x1.0a1f5b8132466p+372, chars_format::general, 112,
        "99999999999999993011993469263043972846733315013897684926158968616472298328309139037619635868942544675772280340"
        "48"},
    {0x1.0a1f5b8132467p+372, chars_format::general, 112,
        "1."
        "00000000000000014371863828472144796796950376709418830953204192182999997798725217259816893675348044905393149706"
        "2e+112"},
    {0x1.4ca732617ed7fp+375, chars_format::general, 113,
        "99999999999999984468045325579403643266646490335689226515340879189861218540142707748740732746380344583923932594"
        "176"},
    {0x1.4ca732617ed80p+375, chars_format::general, 113,
        "1."
        "00000000000000001555941612946684302426820139692106143336977058043083378116475570326498538991504744767620628086"
        "78e+113"},
    {0x1.9fd0fef9de8dfp+378, chars_format::general, 114,
        "99999999999999987885624583052859775098681220206972609879668114960505650455409280264292293995405224620663271692"
        "6976"},
    {0x1.9fd0fef9de8e0p+378, chars_format::general, 114,
        "1."
        "00000000000000001555941612946684302426820139692106143336977058043083378116475570326498538991504744767620628086"
        "784e+114"},
    {0x1.03e29f5c2b18bp+382, chars_format::general, 115,
        "99999999999999979683434365116565058701797868515892489805282749110959013858769506226968546997745512532488857856"
        "24576"},
    {0x1.03e29f5c2b18cp+382, chars_format::general, 115,
        "1."
        "00000000000000001555941612946684302426820139692106143336977058043083378116475570326498538991504744767620628086"
        "784e+115"},
    {0x1.44db473335deep+385, chars_format::general, 116,
        "99999999999999984057935814682588907446802322751135220511621610897383886710310719046874545396497358979515211902"
        "353408"},
    {0x1.44db473335defp+385, chars_format::general, 116,
        "1."
        "00000000000000001555941612946684302426820139692106143336977058043083378116475570326498538991504744767620628086"
        "784e+116"},
    {0x1.961219000356ap+388, chars_format::general, 117,
        "99999999999999991057138133988227065438809449527523589641763789755663683272776659558724142834500313294757378376"
        "1256448"},
    {0x1.961219000356bp+388, chars_format::general, 117,
        "1."
        "00000000000000005055542772599503381422823703080300327902048147472223276397708540582423337710506221925241711323"
        "670118e+117"},
    {0x1.fb969f40042c5p+391, chars_format::general, 118,
        "99999999999999996656499989432737591832415150948634284945877532842287520522749411968203820784902676746951111555"
        "14343424"},
    {0x1.fb969f40042c6p+391, chars_format::general, 118,
        "1."
        "00000000000000007855223700321758644619626553790855675554105019015535195022694916787163176685707403651338577913"
        "1790131e+118"},
    {0x1.3d3e2388029bbp+395, chars_format::general, 119,
        "99999999999999994416755247254933381274972870380190006824232035607637985622760311004411949604741731366073618283"
        "536318464"},
    {0x1.3d3e2388029bcp+395, chars_format::general, 119,
        "1."
        "00000000000000012334713184677367065734511114927744231797396013484834264822673118714746919046029294413093564456"
        "39324467e+119"},
    {0x1.8c8dac6a0342ap+398, chars_format::general, 120,
        "99999999999999998000346834739420118166880519289700851818864831183077241462742872546478943492999243975477607518"
        "1077037056"},
    {0x1.8c8dac6a0342bp+398, chars_format::general, 120,
        "1."
        "00000000000000012334713184677367065734511114927744231797396013484834264822673118714746919046029294413093564456"
        "393244672e+120"},
    {0x1.efb1178484134p+401, chars_format::general, 121,
        "99999999999999992266600294764241339139828281034483499827452358262374432118770774079171753271787223800431224742"
        "79348731904"},
    {0x1.efb1178484135p+401, chars_format::general, 121,
        "1."
        "00000000000000003734093374714598897193932757544918203810277304103780050806714971013786133714211264150523990293"
        "4219200922e+121"},
    {0x1.35ceaeb2d28c0p+405, chars_format::general, 122,
        "99999999999999983092605830803955292696544699826135736641192401589249937168415416531480248917847991520357012302"
        "290741100544"},
    {0x1.35ceaeb2d28c1p+405, chars_format::general, 122,
        "1."
        "00000000000000001440594758724527385583111862242831263013712314935498927069126131626863257625726456080505437183"
        "29623353754e+122"},
    {0x1.83425a5f872f1p+408, chars_format::general, 123,
        "99999999999999997770996973140412967005798429759492157739208332266249129088983988607786655884150763168475752207"
        "0951350501376"},
    {0x1.83425a5f872f2p+408, chars_format::general, 123,
        "1."
        "00000000000000012449388115476870641315052159692848578837224262943248321009552560684093062850453534816594492111"
        "899528999731e+123"},
    {0x1.e412f0f768fadp+411, chars_format::general, 124,
        "99999999999999994835318744673121432143947683772820873519605146130849290704870274192525374490890208838852004226"
        "13425626021888"},
    {0x1.e412f0f768faep+411, chars_format::general, 124,
        "1."
        "00000000000000006578031658542287571591350667719506010398017890672448644241325131853570500063932426157346996149"
        "9777714198938e+124"},
    {0x1.2e8bd69aa19ccp+415, chars_format::general, 125,
        "99999999999999992486776161899288204254467086983483846143922597222529419997579302660316349376281765375153005841"
        "365553228283904"},
    {0x1.2e8bd69aa19cdp+415, chars_format::general, 125,
        "1."
        "00000000000000011275116824089954027370311861298180065149382988489088385655907074917988550293149313084744992919"
        "51517748376371e+125"},
    {0x1.7a2ecc414a03fp+418, chars_format::general, 126,
        "99999999999999992486776161899288204254467086983483846143922597222529419997579302660316349376281765375153005841"
        "3655532282839040"},
    {0x1.7a2ecc414a040p+418, chars_format::general, 126,
        "1."
        "00000000000000007517448691651820862747142906435240821348290910235776592524241520466454110109775803542826595503"
        "88525263266775e+126"},
    {0x1.d8ba7f519c84fp+421, chars_format::general, 127,
        "99999999999999995492910667849794735953002250873835241184796259825178854502911746221543901522980573008687723773"
        "86949310916067328"},
    {0x1.d8ba7f519c850p+421, chars_format::general, 127,
        "1."
        "00000000000000007517448691651820862747142906435240821348290910235776592524241520466454110109775803542826595503"
        "8852526326677504e+127"},
    {0x1.27748f9301d31p+425, chars_format::general, 128,
        "99999999999999988278187853568579059876517857536991893086699469578820211690113881674597776370903434688204400735"
        "860037395056427008"},
    {0x1.27748f9301d32p+425, chars_format::general, 128,
        "1."
        "00000000000000007517448691651820862747142906435240821348290910235776592524241520466454110109775803542826595503"
        "8852526326677504e+128"},
    {0x1.7151b377c247ep+428, chars_format::general, 129,
        "99999999999999999821744356418524141598892886875941250043654333972994040190590464949711576614226856000977717596"
        "6751665376232210432"},
    {0x1.7151b377c247fp+428, chars_format::general, 129,
        "1."
        "00000000000000015213153026885117583895392925994540392652927486498559144857892575983196643605324751084675473411"
        "095338727712279757e+129"},
    {0x1.cda62055b2d9dp+431, chars_format::general, 130,
        "99999999999999993665180888231886764680292871228501592999945072962767998323669620536317549817787697967498615270"
        "90709766158759755776"},
    {0x1.cda62055b2d9ep+431, chars_format::general, 130,
        "1."
        "00000000000000005978307824605161518517492902523380907087363594983220082057511309363105603410666014034456819922"
        "4432354136588445286e+130"},
    {0x1.2087d4358fc82p+435, chars_format::general, 131,
        "99999999999999991202555500957231813912852864969525730182461368558677581576901282770959939099212034754106974340"
        "599870111173348163584"},
    {0x1.2087d4358fc83p+435, chars_format::general, 131,
        "1."
        "00000000000000010903558599154471420052372915041332632722331003791400915551047984893820824847817340461240101783"
        "05769051448734331699e+131"},
    {0x1.68a9c942f3ba3p+438, chars_format::general, 132,
        "99999999999999999082956740236127656368660884998248491198409222651766915166559963620104293398654157036960225317"
        "5829982724989462249472"},
    {0x1.68a9c942f3ba4p+438, chars_format::general, 132,
        "1."
        "00000000000000014843759218793919341280276925055694013230304930837945582345877325318393001997538401602666727271"
        "549254595150142347674e+132"},
    {0x1.c2d43b93b0a8bp+441, chars_format::general, 133,
        "99999999999999989626475253101452645421691260963781177979271797740059714858969546601131068239323610297536324145"
        "20324447890822855131136"},
    {0x1.c2d43b93b0a8cp+441, chars_format::general, 133,
        "1."
        "00000000000000002235117235947685993350984093009737595604788364289002648602423435959762035118431005950101525708"
        "3762495370291854494925e+133"},
    {0x1.19c4a53c4e697p+445, chars_format::general, 134,
        "99999999999999992148203649670699315007549827372972461504375111049848301607660324472857261615145089428049364457"
        "837845490532419930947584"},
    {0x1.19c4a53c4e698p+445, chars_format::general, 134,
        "1."
        "00000000000000012322030822224672671694418358646502729705201617528156995597186547446666808621716922472153686958"
        "91465358352595096803738e+134"},
    {0x1.6035ce8b6203dp+448, chars_format::general, 135,
        "99999999999999996182969084181493986344923533627678515144540412345510040405565569067619171016459456036870228958"
        "0532071091311261383655424"},
    {0x1.6035ce8b6203ep+448, chars_format::general, 135,
        "1."
        "00000000000000012322030822224672671694418358646502729705201617528156995597186547446666808621716922472153686958"
        "914653583525950968037376e+135"},
    {0x1.b843422e3a84cp+451, chars_format::general, 136,
        "99999999999999992955156736572858249275024568623913672232408171308980649367241373391809643495407962749813537357"
        "88091781425216117243117568"},
    {0x1.b843422e3a84dp+451, chars_format::general, 136,
        "1."
        "00000000000000005866406127007401197554620428638973043880937135455098213520538156095047753579613935898040303758"
        "5700749937680210361686426e+136"},
    {0x1.132a095ce492fp+455, chars_format::general, 137,
        "99999999999999982626157224225223890651347880611866174913584999992086598044603947229219155428043184231232124237"
        "329592070639473281441202176"},
    {0x1.132a095ce4930p+455, chars_format::general, 137,
        "1."
        "00000000000000003284156248920492607898701256635961169551231342625874700689878799554400131562772741268394950478"
        "43224355786484906342114918e+137"},
    {0x1.57f48bb41db7bp+458, chars_format::general, 138,
        "99999999999999986757757029164277634100818555816685173841114268518844218573658917694255350654989095638664689485"
        "5501223680845484378371915776"},
    {0x1.57f48bb41db7cp+458, chars_format::general, 138,
        "1."
        "00000000000000003284156248920492607898701256635961169551231342625874700689878799554400131562772741268394950478"
        "432243557864849063421149184e+138"},
    {0x1.adf1aea12525ap+461, chars_format::general, 139,
        "99999999999999990063036873115520628860395095980540372983137683340250314996902894066284306836545824764610741684"
        "12654660604060856295398309888"},
    {0x1.adf1aea12525bp+461, chars_format::general, 139,
        "1."
        "00000000000000003284156248920492607898701256635961169551231342625874700689878799554400131562772741268394950478"
        "432243557864849063421149184e+139"},
    {0x1.0cb70d24b7378p+465, chars_format::general, 140,
        "99999999999999984774589122793531837245072631718372054355900219626000560719712531871037976946055058163097058166"
        "404267825310912362767116664832"},
    {0x1.0cb70d24b7379p+465, chars_format::general, 140,
        "1."
        "00000000000000005928380124081487003706362488767045328864850074482999577828473980652023296508018124569151792237"
        "29338294822969716351458240102e+140"},
    {0x1.4fe4d06de5056p+468, chars_format::general, 141,
        "99999999999999984774589122793531837245072631718372054355900219626000560719712531871037976946055058163097058166"
        "4042678253109123627671166648320"},
    {0x1.4fe4d06de5057p+468, chars_format::general, 141,
        "1."
        "00000000000000001697621923823895970414104517357310673963060103511599774406721690895826232595625511287940845423"
        "115559923645940203365089253786e+141"},
    {0x1.a3de04895e46cp+471, chars_format::general, 142,
        "99999999999999991543802243205677490512685385973947502198764173180240246194516195480953279205883239413034573069"
        "08878466464492349900630570041344"},
    {0x1.a3de04895e46dp+471, chars_format::general, 142,
        "1."
        "00000000000000005082228484029968797047910894485098397884492080288719617144123522700783883725539601912909602874"
        "4578183433129457714846837715763e+142"},
    {0x1.066ac2d5daec3p+475, chars_format::general, 143,
        "99999999999999980713061250546244445284504979165026785650181847493456749434830333705088795590158149413134549224"
        "793557721710505681023603243483136"},
    {0x1.066ac2d5daec4p+475, chars_format::general, 143,
        "1."
        "00000000000000002374543235865110535740865792782868218747346498867023742954202057256817762821608329412934596913"
        "38401160757934131698900815734374e+143"},
    {0x1.4805738b51a74p+478, chars_format::general, 144,
        "99999999999999985045357647610017663375777141888595072269614777768170148138704678415434589036448185413094558762"
        "5116484988842728082166842262552576"},
    {0x1.4805738b51a75p+478, chars_format::general, 144,
        "1."
        "00000000000000002374543235865110535740865792782868218747346498867023742954202057256817762821608329412934596913"
        "384011607579341316989008157343744e+144"},
    {0x1.9a06d06e26112p+481, chars_format::general, 145,
        "99999999999999998908706118214091961267848062604013589451800154647253023991102581488541128064576300612966589283"
        "20953898584032761523454337112604672"},
    {0x1.9a06d06e26113p+481, chars_format::general, 145,
        "1."
        "00000000000000012772054588818166259159918983319432106633985531526335899843500484561647667092704415812838619803"
        "9074294727963824222524025159968358e+145"},
    {0x1.00444244d7cabp+485, chars_format::general, 146,
        "99999999999999993363366729972462242111019694317846182578926003895619873650143420259298512453325054533017777074"
        "930382791057905692427399713177731072"},
    {0x1.00444244d7cacp+485, chars_format::general, 146,
        "1."
        "00000000000000015544724282938981118738333167462515810070422606902152475013980065176268974898330038852813025908"
        "04700757018759338365597434497099366e+146"},
    {0x1.405552d60dbd6p+488, chars_format::general, 147,
        "99999999999999997799638240565766017436482388946780108077225324496926393922910749242692604942326051396976826841"
        "5537077468838432306731146395363835904"},
    {0x1.405552d60dbd7p+488, chars_format::general, 147,
        "1."
        "00000000000000015544724282938981118738333167462515810070422606902152475013980065176268974898330038852813025908"
        "047007570187593383655974344970993664e+147"},
    {0x1.906aa78b912cbp+491, chars_format::general, 148,
        "99999999999999990701603823616479976915742077540485827279946411534835961486483022869262056959924456414642347214"
        "95638781756234316947997075736253956096"},
    {0x1.906aa78b912ccp+491, chars_format::general, 148,
        "1."
        "00000000000000004897672657515052057957222700353074388874504237459016826359338475616123152924727646379311306468"
        "1510276762053432918662585217102276198e+148"},
    {0x1.f485516e7577ep+494, chars_format::general, 149,
        "99999999999999993540817590396194393124038202103003539598857976719672134461054113418634276152885094407576139065"
        "595315789290943193957228310232077172736"},
    {0x1.f485516e7577fp+494, chars_format::general, 149,
        "1."
        "00000000000000004897672657515052057957222700353074388874504237459016826359338475616123152924727646379311306468"
        "15102767620534329186625852171022761984e+149"},
    {0x1.38d352e5096afp+498, chars_format::general, 150,
        "99999999999999998083559617243737459057312001403031879309116481015410011220367858297629826861622115196270206026"
        "6176005440567032331208403948233373515776"},
    {0x1.38d352e5096b0p+498, chars_format::general, 150,
        "1."
        "00000000000000016254527724633909722790407198603145238150150498198361518257622837813612029696570198351046473870"
        "706739563119743389775288733188378066944e+150"},
    {0x1.8708279e4bc5ap+501, chars_format::general, 151,
        "99999999999999987180978752809634100817454883082963864004496070705639106998014870588040505160653265303404445320"
        "16411713261887913912817139180431292235776"},
    {0x1.8708279e4bc5bp+501, chars_format::general, 151,
        "1."
        "00000000000000001717753238721771911803931040843054551077323284452000312627818854200826267428611731827225459595"
        "4354283478693112644517300624963454946509e+151"},
    {0x1.e8ca3185deb71p+504, chars_format::general, 152,
        "99999999999999992995688547174489225212045346187000138833626956204183589249936464033154810067836651912932851030"
        "272641618719051989257594860081125951275008"},
    {0x1.e8ca3185deb72p+504, chars_format::general, 152,
        "1."
        "00000000000000004625108135904199474001226272395072688491888727201272553753779650923383419882203425131989662450"
        "4896905909193976895164417966347520091095e+152"},
    {0x1.317e5ef3ab327p+508, chars_format::general, 153,
        "99999999999999999973340300412315374485553901911843668628584018802436967952242376167291975956456715844366937882"
        "4028710020392594094129030220133015859757056"},
    {0x1.317e5ef3ab328p+508, chars_format::general, 153,
        "1."
        "00000000000000018580411642379851772548243383844759748081802852397779311158391475191657751659443552994857836154"
        "750149357559812529827058120499103278510899e+153"},
    {0x1.7dddf6b095ff0p+511, chars_format::general, 154,
        "99999999999999988809097495231793535647940212752094020956652718645231562028552916752672510534664613554072398918"
        "99450398872692753716440996292182057045458944"},
    {0x1.7dddf6b095ff1p+511, chars_format::general, 154,
        "1."
        "00000000000000003694754568805822654098091798298426884519227785521505436593472195972165131097054083274465117536"
        "8723266731433700334957340417104619244827443e+154"},
    {0x1.dd55745cbb7ecp+514, chars_format::general, 155,
        "99999999999999988809097495231793535647940212752094020956652718645231562028552916752672510534664613554072398918"
        "994503988726927537164409962921820570454589440"},
    {0x1.dd55745cbb7edp+514, chars_format::general, 155,
        "1."
        "00000000000000000717623154091016830408061481189160311806712772146250661680488340128266606984576189330386573813"
        "29676213626008153422946922595273365367711334e+155"},
    {0x1.2a5568b9f52f4p+518, chars_format::general, 156,
        "99999999999999998335918022319172171456037227501747053636700761446046841750101255453147787694593874175123738834"
        "4363105067534507348164573733465510370326085632"},
    {0x1.2a5568b9f52f5p+518, chars_format::general, 156,
        "1."
        "00000000000000017389559076493929443072231257001053118996796847047677401193197932854098342014452395417226418665"
        "31992354280649713012055219419601197018864681e+156"},
    {0x1.74eac2e8727b1p+521, chars_format::general, 157,
        "99999999999999998335918022319172171456037227501747053636700761446046841750101255453147787694593874175123738834"
        "43631050675345073481645737334655103703260856320"},
    {0x1.74eac2e8727b2p+521, chars_format::general, 157,
        "1."
        "00000000000000013578830865658977988748992451101191905924777629927351289304578597373908231150480691168805882699"
        "1432009355958878510597332300261197835574391603e+157"},
    {0x1.d22573a28f19dp+524, chars_format::general, 158,
        "99999999999999995287335453651211007997446182781858083179085387749785952239205787068995699003416510776387310061"
        "494932420984963311567802202010637287727642443776"},
    {0x1.d22573a28f19ep+524, chars_format::general, 158,
        "1."
        "00000000000000007481665728323055661831810361661413965009546882534829510282787660605604053768125964371333025153"
        "26044476405891300456242288735429228494750692147e+158"},
    {0x1.2357684599702p+528, chars_format::general, 159,
        "99999999999999992848469398716842077230573347005946906812993088792777240630489412361674028050474620057398167043"
        "1418299523701733729688780649419062882836695482368"},
    {0x1.2357684599703p+528, chars_format::general, 159,
        "1."
        "00000000000000012359397838191793523365556033213236317741731480448846933500220410020247395674009745809311311189"
        "96664970128849288176027116149175428383545271255e+159"},
    {0x1.6c2d4256ffcc2p+531, chars_format::general, 160,
        "99999999999999985044098022926861498776580272523031142441497732130349363482597013298244681001060569756632909384"
        "41190205280284556945232082632196709006295628251136"},
    {0x1.6c2d4256ffcc3p+531, chars_format::general, 160,
        "1."
        "00000000000000000652840774506822655684566421488862671184488445455205117778381811425103375099888670358163424701"
        "8717578519375011764854353035618454865043828139622e+160"},
    {0x1.c73892ecbfbf3p+534, chars_format::general, 161,
        "99999999999999991287595123558845961539774732109363753938694017460291665200910932548988158640591809997245115511"
        "395844372456707812265566617217918448639526895091712"},
    {0x1.c73892ecbfbf4p+534, chars_format::general, 161,
        "1."
        "00000000000000003774589324822814887066163651282028976933086588120176268637538771050475113919654290478469527765"
        "36372901176443229789205819900982116579266812025242e+161"},
    {0x1.1c835bd3f7d78p+538, chars_format::general, 162,
        "99999999999999993784993963811639746645052515943896798537572531592268585888236500249285549696404306093489997962"
        "1894213003182527093908649335762989920701551401238528"},
    {0x1.1c835bd3f7d79p+538, chars_format::general, 162,
        "1."
        "00000000000000013764184685833990027487274786620161155328600644648083951386841041851664678142904274863449057568"
        "538036723210611886393251464443343339515181100380979e+162"},
    {0x1.63a432c8f5cd6p+541, chars_format::general, 163,
        "99999999999999993784993963811639746645052515943896798537572531592268585888236500249285549696404306093489997962"
        "18942130031825270939086493357629899207015514012385280"},
    {0x1.63a432c8f5cd7p+541, chars_format::general, 163,
        "1."
        "00000000000000009768346541429519971318830332484908283970395022036920878287120133531188852453604281109457245647"
        "2683136386321400509927741582699344700261759083295539e+163"},
    {0x1.bc8d3f7b3340bp+544, chars_format::general, 164,
        "99999999999999987391652932764487656775541389327492204364443535414407668928683046936524228593524316087103098888"
        "157864364992697772750101243698844800887746832841572352"},
    {0x1.bc8d3f7b3340cp+544, chars_format::general, 164,
        "1."
        "00000000000000000178334994858791836514563642560301392710701527770129502847789953562046870799284296099876897036"
        "22097823564380764603162862345375318325256344740613325e+164"},
    {0x1.15d847ad00087p+548, chars_format::general, 165,
        "99999999999999989948989345183348492723345839974054042033695133885552035712504428261628757034676312089657858517"
        "7704871391229197474064067196498264773607101557544845312"},
    {0x1.15d847ad00088p+548, chars_format::general, 165,
        "1."
        "00000000000000010407680644534235180305781445146548743387707921654706969983075478862464984563892280110095935554"
        "671469332164695544656850527257679889144416739057781965e+165"},
    {0x1.5b4e5998400a9p+551, chars_format::general, 166,
        "99999999999999994040727605053525830239832961008552982304497691439383022566618638381796002540519505693745473925"
        "15068357773127490685649548117139715971745147241514401792"},
    {0x1.5b4e5998400aap+551, chars_format::general, 166,
        "1."
        "00000000000000010407680644534235180305781445146548743387707921654706969983075478862464984563892280110095935554"
        "6714693321646955446568505272576798891444167390577819648e+166"},
    {0x1.b221effe500d3p+554, chars_format::general, 167,
        "99999999999999990767336997157383960226643264180953830087855645396318233083327270285662206135844950810475381599"
        "246526426844590779296424471954140613832058419086616428544"},
    {0x1.b221effe500d4p+554, chars_format::general, 167,
        "1."
        "00000000000000003860899428741951440279402051491350438954423829568577391016492742670197391754543170343555750902"
        "86315503039132728953670850882316679737363063240072678605e+167"},
    {0x1.0f5535fef2084p+558, chars_format::general, 168,
        "99999999999999993386049483474297456237195021643033151861169282230770064669960364762569243259584594717091455459"
        "9698521475539380813444812793279458505403728617494385000448"},
    {0x1.0f5535fef2085p+558, chars_format::general, 168,
        "1."
        "00000000000000014335749374009605424321609081339667726047678376906384717363025120577825540249501745970020046345"
        "756457913228716497728935738318387744206888403052015072051e+168"},
    {0x1.532a837eae8a5p+561, chars_format::general, 169,
        "99999999999999993386049483474297456237195021643033151861169282230770064669960364762569243259584594717091455459"
        "96985214755393808134448127932794585054037286174943850004480"},
    {0x1.532a837eae8a6p+561, chars_format::general, 169,
        "1."
        "00000000000000010145809395902543830704726269400340811210376557971261786824412169414774280851518315719434328168"
        "5991367600937608144520448465202993654735852947914997576499e+169"},
    {0x1.a7f5245e5a2cep+564, chars_format::general, 170,
        "99999999999999990034097500988648181343688772091571619991327827082671720239070003832128235741197850516622880918"
        "243995225045973534722968565889475147553730375141026248523776"},
    {0x1.a7f5245e5a2cfp+564, chars_format::general, 170,
        "1."
        "00000000000000003441905430931245280917713770297417747470693647675065097962631447553892265814744827318497179085"
        "14742291507783172120901941964335795950030032157467525460787e+170"},
    {0x1.08f936baf85c1p+568, chars_format::general, 171,
        "99999999999999995397220672965687021173298771373910070983074155319629071328494581320833847770616641237372600185"
        "0053663010587168093173889073910282723323583537144858509574144"},
    {0x1.08f936baf85c2p+568, chars_format::general, 171,
        "1."
        "00000000000000016849713360873842380491738768503263874950059468267458475686192891275656295888291804120371477252"
        "050850605109689907695070273397240771446870268008324260691968e+171"},
    {0x1.4b378469b6731p+571, chars_format::general, 172,
        "99999999999999991106722135384055949309610771948039310189677092730063190456954919329869358147081608660772824771"
        "59626944024852218964185263418978577250945597085571816901050368"},
    {0x1.4b378469b6732p+571, chars_format::general, 172,
        "1."
        "00000000000000008268716285710580236764362769651522353363265343088326713943113567293727316641221738967171926425"
        "2326568834893006683439977269947557718010655022907888967981466e+172"},
    {0x1.9e056584240fdp+574, chars_format::general, 173,
        "99999999999999987674323305318751091818660372407342701554959442658410485759723189737097766448253582599493004440"
        "868991951600366493901423615628791772651134064568704023452975104"},
    {0x1.9e056584240fep+574, chars_format::general, 173,
        "1."
        "00000000000000001403918625579970521782461970570129136093830042945021304548650108108184133243565686844612285763"
        "77810190619298927686313968987276777208442168971676060568308941e+173"},
    {0x1.02c35f729689ep+578, chars_format::general, 174,
        "99999999999999984928404241266507205825900052774785414647185322601088322001937806062880493089191161750469148176"
        "2871699606818419373090804007799965727644765395390927070069522432"},
    {0x1.02c35f729689fp+578, chars_format::general, 174,
        "1."
        "00000000000000006895756753684458293767982609835243709909378283059665632064220875456618679961690528542659998292"
        "94174588803003839004782611957035817185773673977598323857513513e+174"},
    {0x1.4374374f3c2c6p+581, chars_format::general, 175,
        "99999999999999993715345246233687641002733075598968732752062506784519246026851033820375767838190908467345488222"
        "94900033162112051840457868829614121240178061963384891963422539776"},
    {0x1.4374374f3c2c7p+581, chars_format::general, 175,
        "1."
        "00000000000000011289227256168048511356399121247335368961816875151381094076677489335366317336190401901098168316"
        "2726610734996776805955752633284304916763887798233613448887717069e+175"},
    {0x1.945145230b377p+584, chars_format::general, 176,
        "99999999999999986685792442259943292861266657339622078268160759437774506806920451614379548038991111093844416185"
        "619536034869697653528180058283225500691937355558043949532406874112"},
    {0x1.945145230b378p+584, chars_format::general, 176,
        "1."
        "00000000000000000744898050207431989144199493858315387235964254131263985246781616026371987637390705840846560260"
        "27846462837254338328097731830905692411162388370965388973604392141e+176"},
    {0x1.f965966bce055p+587, chars_format::general, 177,
        "99999999999999989497613563849441032117853224643360740061721458376472402494892684496778035958671030043244845000"
        "5513217535702667994787395102883917853758746611883659375731342835712"},
    {0x1.f965966bce056p+587, chars_format::general, 177,
        "1."
        "00000000000000000744898050207431989144199493858315387235964254131263985246781616026371987637390705840846560260"
        "278464628372543383280977318309056924111623883709653889736043921408e+177"},
    {0x1.3bdf7e0360c35p+591, chars_format::general, 178,
        "99999999999999987248156666577842840712583970800369810626872899225514085944514898190859245622927094883724501948"
        "60589317860981148271829194868425875762872481668410834714055235600384"},
    {0x1.3bdf7e0360c36p+591, chars_format::general, 178,
        "1."
        "00000000000000005243811844750628371954738001544297246105661372433180618347537188638209568308878576159887246364"
        "1693217782934540168018724415173229796059235727181690706012077765427e+178"},
    {0x1.8ad75d8438f43p+594, chars_format::general, 179,
        "99999999999999998045549773481514159457876389246726271914145983150114005386328272459269439234497983649422148597"
        "943950338419997003168440244384097290815044070304544781216945608327168"},
    {0x1.8ad75d8438f44p+594, chars_format::general, 179,
        "1."
        "00000000000000012442073916019742584451599613841868220297176761716247231308746104817149697383259168670352344130"
        "39469321816691103043530463865054866839680307513179335998546994475827e+179"},
    {0x1.ed8d34e547313p+597, chars_format::general, 180,
        "99999999999999989407635287958577104461642454489641102884327516010434069832877573044541284345241272636864031278"
        "4735046105718485868083216078242264642659886674081956339558310064685056"},
    {0x1.ed8d34e547314p+597, chars_format::general, 180,
        "1."
        "00000000000000000924854601989159844456621034165754661590752138863340650570811838930845490864250220653608187704"
        "434098914369379808621813123237387566331395871269994496970650475613389e+180"},
    {0x1.3478410f4c7ecp+601, chars_format::general, 181,
        "99999999999999991711079150764693652460638170424863814625612440581015385980464426221802125649043062240212862563"
        "66562347133135483117101991090685868467907010818055540655879490029748224"},
    {0x1.3478410f4c7edp+601, chars_format::general, 181,
        "1."
        "00000000000000010138630053213626036452603897906645508555891837145665915161159251639888856079457379067003512845"
        "2025743574074047860726063355679164479837216343594335873825060509292954e+181"},
    {0x1.819651531f9e7p+604, chars_format::general, 182,
        "99999999999999991711079150764693652460638170424863814625612440581015385980464426221802125649043062240212862563"
        "665623471331354831171019910906858684679070108180555406558794900297482240"},
    {0x1.819651531f9e8p+604, chars_format::general, 182,
        "1."
        "00000000000000006453119872723839559654210752410289169769835957832735809325020286556271509993374515701645382788"
        "89518418019219479509228905063570489532279132912365795121776382080293274e+182"},
    {0x1.e1fbe5a7e7861p+607, chars_format::general, 183,
        "99999999999999994659487295156522833899352686821948885654457144031359470649375598288696002517909352932499366608"
        "7115356131035228239552737388526279268078143523691759154905886843985723392"},
    {0x1.e1fbe5a7e7862p+607, chars_format::general, 183,
        "1."
        "00000000000000006453119872723839559654210752410289169769835957832735809325020286556271509993374515701645382788"
        "895184180192194795092289050635704895322791329123657951217763820802932736e+183"},
    {0x1.2d3d6f88f0b3cp+611, chars_format::general, 184,
        "99999999999999982865854717589206108144494621233608601539078330229983131973730910021120495042444190163353350428"
        "52788704601485085281825842706955095829283737561469387976341354799421194240"},
    {0x1.2d3d6f88f0b3dp+611, chars_format::general, 184,
        "1."
        "00000000000000001735666841696912869352267526174953056123684432312185273854762411249241307003188450593986976316"
        "8217247533567260066374829259224741079168005384218651369268937662411885773e+184"},
    {0x1.788ccb6b2ce0cp+614, chars_format::general, 185,
        "99999999999999997961704416875371517110712945186684165206763211895744845478556111003617144611039598507860251139"
        "162957211888350975873638026151889477992007905860430885494197722591793250304"},
    {0x1.788ccb6b2ce0dp+614, chars_format::general, 185,
        "1."
        "00000000000000013057554116161536926076931269139759728874448093561506558983381311986113794179635006852367151849"
        "79802737776185109892901762523422799769117843610616789122498189718937455821e+185"},
    {0x1.d6affe45f818fp+617, chars_format::general, 186,
        "99999999999999997961704416875371517110712945186684165206763211895744845478556111003617144611039598507860251139"
        "1629572118883509758736380261518894779920079058604308854941977225917932503040"},
    {0x1.d6affe45f8190p+617, chars_format::general, 186,
        "1."
        "00000000000000010038384176304303844283687604349144616140911117228354216282416271789614464265915925183465771707"
        "671013344587151074317941705417760293751344330057020490078825062269858296627e+186"},
    {0x1.262dfeebbb0f9p+621, chars_format::general, 187,
        "99999999999999990715696561218012120806928149689207894646274468696179222996240014532018752818113802502496938798"
        "05812353226907091680705581859236698853640605134247712274342131878495422251008"},
    {0x1.262dfeebbb0fap+621, chars_format::general, 187,
        "1."
        "00000000000000010038384176304303844283687604349144616140911117228354216282416271789614464265915925183465771707"
        "6710133445871510743179417054177602937513443300570204900788250622698582966272e+187"},
    {0x1.6fb97ea6a9d37p+624, chars_format::general, 188,
        "99999999999999986851159038200753776111576258757220550347347138989744224339004763080499610528553377966303172216"
        "135545569805454885304878641227288327493418395599568449276340570087973407686656"},
    {0x1.6fb97ea6a9d38p+624, chars_format::general, 188,
        "1."
        "00000000000000002309309130269787154892983822485169927543056457815484218967945768886576179686795076111078238543"
        "82585741965991901131358735068760297166536901857120314314466356487589666698035e+188"},
    {0x1.cba7de5054485p+627, chars_format::general, 189,
        "99999999999999989942789056614560451867857771502810425786489002754892223264792964241714924360201717595258185481"
        "6736079397763477105066203831193512563278085201938953880500051690455580595453952"},
    {0x1.cba7de5054486p+627, chars_format::general, 189,
        "1."
        "00000000000000002309309130269787154892983822485169927543056457815484218967945768886576179686795076111078238543"
        "825857419659919011313587350687602971665369018571203143144663564875896666980352e+189"},
    {0x1.1f48eaf234ad3p+631, chars_format::general, 190,
        "99999999999999987469485041883515111262832561306338525435175511742773824124162403312742673294883045892094174869"
        "24315804379963345034522698960570091326029642051843383703107348987949033805840384"},
    {0x1.1f48eaf234ad4p+631, chars_format::general, 190,
        "1."
        "00000000000000007255917159731877836103034242878113728245683439839721017249206890744520681817432419517406259768"
        "6867572116133475316363741377149036578003932179221262451825269232080321099543347e+190"},
    {0x1.671b25aec1d88p+634, chars_format::general, 191,
        "99999999999999991426771465453187656230872897620693565997277097362163262749171300799098274999392920617156591849"
        "131877877362376266603456419227541462168315779999172318661364176545198692437590016"},
    {0x1.671b25aec1d89p+634, chars_format::general, 191,
        "1."
        "00000000000000007255917159731877836103034242878113728245683439839721017249206890744520681817432419517406259768"
        "68675721161334753163637413771490365780039321792212624518252692320803210995433472e+191"},
    {0x1.c0e1ef1a724eap+637, chars_format::general, 192,
        "99999999999999991426771465453187656230872897620693565997277097362163262749171300799098274999392920617156591849"
        "1318778773623762666034564192275414621683157799991723186613641765451986924375900160"},
    {0x1.c0e1ef1a724ebp+637, chars_format::general, 192,
        "1."
        "00000000000000004090088020876139800128601973826629695796002171344209466349199772755436200453824519737356326184"
        "775781344763153278629790594017431218673977730337535459878294373875465426450985779e+192"},
    {0x1.188d357087712p+641, chars_format::general, 193,
        "99999999999999986361444843284006798671781267138319114077787067769344781309159912016563104817620280969076698114"
        "87431649040206546179292274931158555956605099986382706217459209761309199883223171072"},
    {0x1.188d357087713p+641, chars_format::general, 193,
        "1."
        "00000000000000006622751331960730228908147789067816921755747186140618707069205467146703785544710839561396273051"
        "9045620382433086810350574289754091699751101204052080881216804133415187732536649318e+193"},
    {0x1.5eb082cca94d7p+644, chars_format::general, 194,
        "99999999999999994465967438754696170766327875910118237148971115117854351613178134068619377108456504406004528089"
        "686414709538562749489776621177115003729674648080379472553427423904462708600804999168"},
    {0x1.5eb082cca94d8p+644, chars_format::general, 194,
        "1."
        "00000000000000010675012629696074914955421093453716483291339209814873492221214578172731921690128951279860188039"
        "31061114781155732488348436490817389205692194451348429331109807648720412813795157606e+194"},
    {0x1.b65ca37fd3a0dp+647, chars_format::general, 195,
        "99999999999999997707776476942971919604146519418837886377444734057258179734785422889441886024790993780775660079"
        "6112539971931616645685181699233267813951241073670004367049615544210109925082343145472"},
    {0x1.b65ca37fd3a0ep+647, chars_format::general, 195,
        "1."
        "00000000000000010675012629696074914955421093453716483291339209814873492221214578172731921690128951279860188039"
        "310611147811557324883484364908173892056921944513484293311098076487204128137951576064e+195"},
    {0x1.11f9e62fe4448p+651, chars_format::general, 196,
        "99999999999999995114329246392351320533891604611862166994665838905735117237499591832783878891723402280958754487"
        "67138256706948253250552493092635735926276453993770366538373425000777236538229086224384"},
    {0x1.11f9e62fe4449p+651, chars_format::general, 196,
        "1."
        "00000000000000015861907090797316113095930923067667922056897000117919617215786240286047935956264134279493999223"
        "1903540080589155890094708429021127363216410793720778359535526853136813823898384806707e+196"},
    {0x1.56785fbbdd55ap+654, chars_format::general, 197,
        "99999999999999995114329246392351320533891604611862166994665838905735117237499591832783878891723402280958754487"
        "671382567069482532505524930926357359262764539937703665383734250007772365382290862243840"},
    {0x1.56785fbbdd55bp+654, chars_format::general, 197,
        "1."
        "00000000000000011712391521916323154583523059376506771044450767875482717220128910595395124543355987879786950276"
        "08655971986102897770868166050696166090986577148520300183958899825249957898832895698534e+197"},
    {0x1.ac1677aad4ab0p+657, chars_format::general, 198,
        "99999999999999988475104336182762586914039022706004325374751867317836077244447864327739380631070368041427476172"
        "3053117059528639544242622390941156386039240473187039308013923507098814799398756243472384"},
    {0x1.ac1677aad4ab1p+657, chars_format::general, 198,
        "1."
        "00000000000000001753554156601940054153744186517720008614579810493634157230551319337828377152376436520490032803"
        "037453428186101110586787622758599079921605032556703399966076149305663250824706100140442e+198"},
    {0x1.0b8e0acac4eaep+661, chars_format::general, 199,
        "99999999999999988475104336182762586914039022706004325374751867317836077244447864327739380631070368041427476172"
        "30531170595286395442426223909411563860392404731870393080139235070988147993987562434723840"},
    {0x1.0b8e0acac4eafp+661, chars_format::general, 199,
        "1."
        "00000000000000009720624048853446534497567284804749418558476576399113005222213392343881775065160077607927566781"
        "4767384615260434042843028529572891447122136236995030814648864284631323133556043856163635e+199"},
    {0x1.4e718d7d7625ap+664, chars_format::general, 200,
        "99999999999999996973312221251036165947450327545502362648241750950346848435554075534196338404706251868027512415"
        "973882408182135734368278484639385041047239877871023591066789981811181813306167128854888448"},
    {0x1.4e718d7d7625bp+664, chars_format::general, 200,
        "1."
        "00000000000000013969727991387583324014272937224498437195221518215368390817766497947110253951978019521227584903"
        "31102381264067929425631097572992384593387153897566291159758524401378248003875013787018854e+200"},
    {0x1.a20df0dcd3af0p+667, chars_format::general, 201,
        "99999999999999990174745913196417302720721283673903932829449844044338231482669106569030772185797544806747483421"
        "0390258463987183104130654882031695190925872134291678628544718769301415466131339252487684096"},
    {0x1.a20df0dcd3af1p+667, chars_format::general, 201,
        "1."
        "00000000000000003771878529305655029174179371417100792467033657856355465388439044499361904623614958929307541410"
        "908738969965553158323491481075600563001892542312879319279108086692222079999200332461008486e+201"},
    {0x1.0548b68a044d6p+671, chars_format::general, 202,
        "99999999999999990174745913196417302720721283673903932829449844044338231482669106569030772185797544806747483421"
        "03902584639871831041306548820316951909258721342916786285447187693014154661313392524876840960"},
    {0x1.0548b68a044d7p+671, chars_format::general, 202,
        "1."
        "00000000000000011930158098971197665046254224063018908249583946143565805731901007257560584086305407402843576204"
        "8305668441056540670697470767990591893474757396431061931338898125494704000308401767883525325e+202"},
    {0x1.469ae42c8560cp+674, chars_format::general, 203,
        "99999999999999998876910787506329447650934459829549922997503484884029261182361866844442696946000689845185920534"
        "555642245481492613075738123641525387194542623914743194966239051177873087980216425864602058752"},
    {0x1.469ae42c8560dp+674, chars_format::general, 203,
        "1."
        "00000000000000016281240536126153737511360812140841903333610766563411320581747387395266546466406979922062794761"
        "58887504364704121840108339451823712339845344488589385918977339967333617071438142709626935706e+203"},
    {0x1.98419d37a6b8fp+677, chars_format::general, 204,
        "99999999999999998876910787506329447650934459829549922997503484884029261182361866844442696946000689845185920534"
        "5556422454814926130757381236415253871945426239147431949662390511778730879802164258646020587520"},
    {0x1.98419d37a6b90p+677, chars_format::general, 204,
        "1."
        "00000000000000012800374586402188879539275541678583507266389310227534908701870283285101776562325721906687419916"
        "182228484013931497336014340342894776157671280691663726345066529974243554167548426849935897395e+204"},
    {0x1.fe52048590672p+680, chars_format::general, 205,
        "99999999999999990522832508168813788517929810720129772436171989677925872670656816980047249176205670608285020905"
        "57969050236202928251957239362070375381666542984859087613894256390005080826781722527340175556608"},
    {0x1.fe52048590673p+680, chars_format::general, 205,
        "1."
        "00000000000000001661603547285501334028602676199356639851280649952730390686263550132574512869265696257486220410"
        "8809594931879803899277933669817992649871683552701273012420045469371471812176828260616688264806e+205"},
    {0x1.3ef342d37a407p+684, chars_format::general, 206,
        "99999999999999986067324092522138770313660664528439025470128525568004065464414123719036343698981660348604541103"
        "459182906031648839556284004276265549348464259679976306097717770685212259087870984958094927200256"},
    {0x1.3ef342d37a408p+684, chars_format::general, 206,
        "1."
        "00000000000000003889357755108838843130737249295202013334302382007691294289384896763079965607877701387326460311"
        "94121329135317061140943756165401836722126894035443458626261694354456645580765594621932224066355e+206"},
    {0x1.8eb0138858d09p+687, chars_format::general, 207,
        "99999999999999989631730825039478784877075981481791623042963296855941511229408278327845068080760868556348924945"
        "1555889830959531939269147157518161129230251958148679621306976052570830984318279772103403898929152"},
    {0x1.8eb0138858d0ap+687, chars_format::general, 207,
        "1."
        "00000000000000003889357755108838843130737249295202013334302382007691294289384896763079965607877701387326460311"
        "941213291353170611409437561654018367221268940354434586262616943544566455807655946219322240663552e+207"},
    {0x1.f25c186a6f04cp+690, chars_format::general, 208,
        "99999999999999998186306983081094819829272742169837857217766747946991381065394249388986006597030968254935446165"
        "22696356805028364441642842329313746550197144253860793660984920822957311285732475861572950035529728"},
    {0x1.f25c186a6f04dp+690, chars_format::general, 208,
        "1."
        "00000000000000009592408527136582866432201756420566169450838016068391207513375544137173924618724434519717474458"
        "6554630146560575784024467000148992689405664381702612359153846788595597987579871338229149809718067e+208"},
    {0x1.37798f428562fp+694, chars_format::general, 209,
        "99999999999999989061425747836704382546929530769255207431309733449871519907009213590435672179676195243109823530"
        "484164010765664497227613801915728022751095446033285297165420831725583764136794858449981115862089728"},
    {0x1.37798f4285630p+694, chars_format::general, 209,
        "1."
        "00000000000000007311188218325485257111615953570420507004223762444111242223779285187536341014385741266761068799"
        "96976312533490279160524304467054690825284743904393057605427758473356246157785465878147788484850483e+209"},
    {0x1.8557f31326bbbp+697, chars_format::general, 210,
        "99999999999999992711378241934460557459866815329488267345892539248719464370363227909855805946618104447840072584"
        "3812838336795121561031396504666917998514458446354143529431921823271795036250068185162804696593727488"},
    {0x1.8557f31326bbcp+697, chars_format::general, 210,
        "1."
        "00000000000000007311188218325485257111615953570420507004223762444111242223779285187536341014385741266761068799"
        "969763125334902791605243044670546908252847439043930576054277584733562461577854658781477884848504832e+210"},
    {0x1.e6adefd7f06aap+700, chars_format::general, 211,
        "99999999999999995631340237212665497390216642977674715277558783887797819941046439365391912960171631811624271827"
        "49897969201059028320356032930746282153172616351711759756540926280845609521557638656931995269719916544"},
    {0x1.e6adefd7f06abp+700, chars_format::general, 211,
        "1."
        "00000000000000007311188218325485257111615953570420507004223762444111242223779285187536341014385741266761068799"
        "969763125334902791605243044670546908252847439043930576054277584733562461577854658781477884848504832e+211"},
    {0x1.302cb5e6f642ap+704, chars_format::general, 212,
        "99999999999999990959401044767537593501656918740576398586892792465272451027953301036534141738485988029569553038"
        "510666318680865279842887243162229186843277653306392406169861934038413548670665077684456779836676898816"},
    {0x1.302cb5e6f642bp+704, chars_format::general, 212,
        "1."
        "00000000000000009647157814548049209055895815688969665349556758155373926680325854351965226625228563157788428194"
        "46391981199976529328557958774316372559707169414929317175205124911858373485031031322390947127876596531e+212"},
    {0x1.7c37e360b3d35p+707, chars_format::general, 213,
        "99999999999999998434503752679742239723352477519933705291958378741313041288902322362706575693183018080857103100"
        "8919677160084252852199641809946030023447952696435527124027376600704816231425231719002378564135125254144"},
    {0x1.7c37e360b3d36p+707, chars_format::general, 213,
        "1."
        "00000000000000013384709168504151532166743595078648318702089551293394221810800365015051443602577078183432203225"
        "654570510663545295974118056659350633347830502317873324868489112134617772086239360331800009567183778611e+213"},
    {0x1.db45dc38e0c82p+710, chars_format::general, 214,
        "99999999999999995444462669514860381234674254008190782609932144230896805184522713832237602111304206060342083075"
        "93944715707740128306913340586165347614418822310868858990958736965765439335377993421392542578277827477504"},
    {0x1.db45dc38e0c83p+710, chars_format::general, 214,
        "1."
        "00000000000000007404627002174387815189387148055162473338037082272561749602041147954113496438819454142402163175"
        "7495293928014972916724565063934515809466164092481450798821885313089633125087528849591751483057152773325e+214"},
    {0x1.290ba9a38c7d1p+714, chars_format::general, 215,
        "99999999999999990660396936451049407652789096389402106318690169014230827417515340183487244380298106827518051036"
        "015414262787762879627804165648934234223216948652905993920546904997130825691790753915825536773603473752064"},
    {0x1.290ba9a38c7d2p+714, chars_format::general, 215,
        "1."
        "00000000000000009796659868706293301980329726864556811483658069880894738485544834778488675304322503758814179195"
        "71154583994631649339312112649981120190710204647603637787670876363922509633974747510822509281030267784397e+"
        "215"},
    {0x1.734e940c6f9c5p+717, chars_format::general, 216,
        "99999999999999986833144350000000628787280970294371165285696588840898045203909441264486958195493227441258825404"
        "0761879473560521568747407734787588406864399290882799171293145332687119715621994096773456255662636329336832"},
    {0x1.734e940c6f9c6p+717, chars_format::general, 216,
        "1."
        "00000000000000002142154695804195744249313474674494929417670909534229174058333036940488102934712744986295727931"
        "833093209082895047886994342159460414833548007346784224294244020182387388080564786631265270395622996207206e+"
        "216"},
    {0x1.d022390f8b837p+720, chars_format::general, 217,
        "99999999999999996018550557482517698064500472922445423764881181256896722516563598670087645039024937968280966920"
        "73033110439215789148209291468717978517470477604338250142827222541691722147321863584969741246387925089779712"},
    {0x1.d022390f8b838p+720, chars_format::general, 217,
        "1."
        "00000000000000008265758834125873790434126476426544435070460637811561625600102475210888560830400552004310488942"
        "9358553137736322042918957696317410444923912386501859471602158149478575546879109374128331283273667415166157e+"
        "217"},
    {0x1.221563a9b7322p+724, chars_format::general, 218,
        "99999999999999988670225591496504042642724870819986016981533507324097780666440272745607095564199569546663253707"
        "407016578763273303796211201720443029584092898479300433989106071698353021544403254911815982945786756526505984"},
    {0x1.221563a9b7323p+724, chars_format::general, 218,
        "1."
        "00000000000000008265758834125873790434126476426544435070460637811561625600102475210888560830400552004310488942"
        "93585531377363220429189576963174104449239123865018594716021581494785755468791093741283312832736674151661568e+"
        "218"},
    {0x1.6a9abc9424febp+727, chars_format::general, 219,
        "99999999999999996508438888548251941759285513062609384217104359519083318639905153731719681670679962529722147801"
        "618552072767416863994485028884962235547412234547654639257549968998154834801806327912222841098418750522549862"
        "4"},
    {0x1.6a9abc9424fecp+727, chars_format::general, 219,
        "1."
        "00000000000000012184865482651747739992406797547856118688246063909054394586834915703944853883640748495839935990"
        "041623060775703984391032683214000647474050906684363049794437763597758461316612473913036557403682738514637619e+"
        "219"},
    {0x1.c5416bb92e3e6p+730, chars_format::general, 220,
        "99999999999999999643724207368951101405909769959658731111332700397077533829291106126164716113272119722945705439"
        "3031662703690742880737945597507699179327399689749963213649275279180755601047675571123855843594715481209674137"
        "6"},
    {0x1.c5416bb92e3e7p+730, chars_format::general, 220,
        "1."
        "00000000000000012184865482651747739992406797547856118688246063909054394586834915703944853883640748495839935990"
        "0416230607757039843910326832140006474740509066843630497944377635977584613166124739130365574036827385146376192e"
        "+220"},
    {0x1.1b48e353bce6fp+734, chars_format::general, 221,
        "99999999999999984594354677029595135102113336853821866019036664182705300920238534632828550788829765195472628778"
        "41701812188111865249310881159489304248316684372375624724951524510245607865055365695160441670641811964856316723"
        "2"},
    {0x1.1b48e353bce70p+734, chars_format::general, 221,
        "1."
        "00000000000000004660180717482069756840508580994937686142098045801868278132308629957276771221419571232103397659"
        "59854898653172616660068980913606220974926434405874301273673162218994872058950552383264597357715602427843549594"
        "e+221"},
    {0x1.621b1c28ac20bp+737, chars_format::general, 222,
        "99999999999999988607519885120090059449792385682045030043648940506537896362652553697718194875347726402798782554"
        "65332429481124015531462501110312687593638634379075360034695852051995460703834403032781272808056570057453763297"
        "28"},
    {0x1.621b1c28ac20cp+737, chars_format::general, 222,
        "1."
        "00000000000000004660180717482069756840508580994937686142098045801868278132308629957276771221419571232103397659"
        "59854898653172616660068980913606220974926434405874301273673162218994872058950552383264597357715602427843549593"
        "6e+222"},
    {0x1.baa1e332d728ep+740, chars_format::general, 223,
        "99999999999999991818052051592485998927935624744623561263338761565603972716583768949629910144562095368659705575"
        "64236923315533735757183797070971394269896194384435148282491314085395342974857632902877937717988376531531720556"
        "544"},
    {0x1.baa1e332d728fp+740, chars_format::general, 223,
        "1."
        "00000000000000004660180717482069756840508580994937686142098045801868278132308629957276771221419571232103397659"
        "59854898653172616660068980913606220974926434405874301273673162218994872058950552383264597357715602427843549593"
        "6e+223"},
    {0x1.14a52dffc6799p+744, chars_format::general, 224,
        "99999999999999996954903517948319502092964807244749211214842475260109694882873713352688654575305085714037182409"
        "22484113450589288118337870608025324951908290393010809478964053338835154608494800695032601573879266890056452171"
        "3664"},
    {0x1.14a52dffc679ap+744, chars_format::general, 224,
        "1."
        "00000000000000017502309383371653514753081537245251811020857330038132583548033490964923632298277047095547089743"
        "55472873990811497562954164756241047679956674427313454264855010352594401143043471863651256997442828324155378630"
        "656e+224"},
    {0x1.59ce797fb817fp+747, chars_format::general, 225,
        "99999999999999992845422344863652699560941461244648691253639504304505117149841757830241659030710693437735200942"
        "35886361342544846229414611778382180406298613586150280521785861936083305301585066461308870489166554603236666879"
        "50848"},
    {0x1.59ce797fb8180p+747, chars_format::general, 225,
        "1."
        "00000000000000009283347037202319909689034845245050771098451388126923428081969579920029641209088262542943126809"
        "82277369774722613785107647096954758588737320813592396350498627547090702529224003396203794828017403750515808046"
        "9402e+225"},
    {0x1.b04217dfa61dfp+750, chars_format::general, 226,
        "99999999999999996133007283331386141586560138044729107222601881068988779336267322248199255466386207258776786115"
        "85164563028980399740553218842096696042786355031638703687528415058284784747112853848287855356936724432692495112"
        "994816"},
    {0x1.b04217dfa61e0p+750, chars_format::general, 226,
        "1."
        "00000000000000009283347037202319909689034845245050771098451388126923428081969579920029641209088262542943126809"
        "82277369774722613785107647096954758588737320813592396350498627547090702529224003396203794828017403750515808046"
        "94016e+226"},
    {0x1.0e294eebc7d2bp+754, chars_format::general, 227,
        "99999999999999988242803431008825880725075313724536108897092176834227990088845967645101024020764974088276981699"
        "46896878981535071313820561889181858515215775562466488089746287565001234077846164119538291674288316841998507352"
        "6276096"},
    {0x1.0e294eebc7d2cp+754, chars_format::general, 227,
        "1."
        "00000000000000009283347037202319909689034845245050771098451388126923428081969579920029641209088262542943126809"
        "82277369774722613785107647096954758588737320813592396350498627547090702529224003396203794828017403750515808046"
        "94016e+227"},
    {0x1.51b3a2a6b9c76p+757, chars_format::general, 228,
        "99999999999999992450912152247524686517867220028639041337364019092767077687470690100086747458429631779210210721"
        "53972977140172579808077978930736438529920084612691669741896755561419127768121731974871392305034134223701967491"
        "49011968"},
    {0x1.51b3a2a6b9c77p+757, chars_format::general, 228,
        "1."
        "00000000000000009283347037202319909689034845245050771098451388126923428081969579920029641209088262542943126809"
        "82277369774722613785107647096954758588737320813592396350498627547090702529224003396203794828017403750515808046"
        "94016e+228"},
    {0x1.a6208b5068394p+760, chars_format::general, 229,
        "99999999999999999183886106229442775786334270115203733241798966706429617845270246028063904958693084084703377156"
        "85294734193992593398889846197223766553446979093051960385337504355687757672562640543404353314227442034427503713"
        "670135808"},
    {0x1.a6208b5068395p+760, chars_format::general, 229,
        "1."
        "00000000000000012649834014193278954323268370288333117050668861933754698160869357884018219959219988695689710027"
        "47938248301632620580513580730198422600500768053772541672219001944225017481444457680470275332614057655878576158"
        "03016806e+229"},
    {0x1.07d457124123cp+764, chars_format::general, 230,
        "99999999999999988411127779858373832956786989976700226194703050524569553592790956543300452958271560395914310860"
        "35179922907880571653590858570844041715803947924475495355832306284857949825457186833751615699518149537266645758"
        "1821100032"},
    {0x1.07d457124123dp+764, chars_format::general, 230,
        "1."
        "00000000000000009956644432600511718615881550253707240288894882888289682097749535512827356959114607773492443453"
        "35409545480104615144188833823603491391090010261628425414842702426517565519668094253057090928936734531588361669"
        "158161613e+230"},
    {0x1.49c96cd6d16cbp+767, chars_format::general, 231,
        "99999999999999988411127779858373832956786989976700226194703050524569553592790956543300452958271560395914310860"
        "35179922907880571653590858570844041715803947924475495355832306284857949825457186833751615699518149537266645758"
        "18211000320"},
    {0x1.49c96cd6d16ccp+767, chars_format::general, 231,
        "1."
        "00000000000000005647541102052084141484062638198305837470056516415545656396757819718921976158945998297976816934"
        "75363620965659806446069238773051601456032797794197839403040623198185642380825912769195995883053017532724018486"
        "9629512909e+231"},
    {0x1.9c3bc80c85c7ep+770, chars_format::general, 232,
        "99999999999999991858410444297115894662242119621021348449773743702764774153584329178424757598406447976326812075"
        "23216662519436418612086534611285553663849717898419964165273969667523488336530932020840491736225123136358120303"
        "938278260736"},
    {0x1.9c3bc80c85c7fp+770, chars_format::general, 232,
        "1."
        "00000000000000005647541102052084141484062638198305837470056516415545656396757819718921976158945998297976816934"
        "75363620965659806446069238773051601456032797794197839403040623198185642380825912769195995883053017532724018486"
        "96295129088e+232"},
    {0x1.01a55d07d39cfp+774, chars_format::general, 233,
        "99999999999999997374062707399103193390970327051935144057886852787877127050853725394623645022622268104986814019"
        "04075445897925773745679616275991972780722949856731114260380631079788349954248924320182693394956280894904479577"
        "1481474727936"},
    {0x1.01a55d07d39d0p+774, chars_format::general, 233,
        "1."
        "00000000000000019436671759807052388305883156775590326490339289128326538639931310259419194719485548619626821794"
        "27510579411883194280051942934817649248215877689975714640807276728847796425120893517551500029880911929089916669"
        "987624321024e+233"},
    {0x1.420eb449c8842p+777, chars_format::general, 234,
        "99999999999999984136497275954333676442022629217742034598415390983607480097407174475746315204504299796202809353"
        "90014365789551321425056220280696566900227193156784354032124643690352682071725742801761409414001502274393217321"
        "44446136385536"},
    {0x1.420eb449c8843p+777, chars_format::general, 234,
        "1."
        "00000000000000001786584517880693032373952892996666180544377340055967009368669242367582754961994924207914815574"
        "08762472600717257852554081607757108074221535423380034336465960209600239248423318159656454721941207101741566995"
        "7160428424397e+234"},
    {0x1.9292615c3aa53p+780, chars_format::general, 235,
        "99999999999999991196532172724877418814794734729311692976800170612551291805912001632480891107500549560887611841"
        "97513608514017695996055364811520783369824930063422626153861170298051704942404772944919427537177384205332557191"
        "153093955289088"},
    {0x1.9292615c3aa54p+780, chars_format::general, 235,
        "1."
        "00000000000000005316601966265964903560338945752451009733569729870438915222921655945950042913493049090257216818"
        "12512093962950445138053653873169216309020403876699170397334223513449750683762833231235463783529148067211236930"
        "57035913815654e+235"},
    {0x1.f736f9b3494e8p+783, chars_format::general, 236,
        "99999999999999994020546131433094915763903576933939556328154082464128816489313932495174721468699049466761532837"
        "20513305603804245824455022623850469957664024826077935002555780941131314090676385002182634786447736977708293139"
        "0365469918625792"},
    {0x1.f736f9b3494e9p+783, chars_format::general, 236,
        "1."
        "00000000000000005316601966265964903560338945752451009733569729870438915222921655945950042913493049090257216818"
        "12512093962950445138053653873169216309020403876699170397334223513449750683762833231235463783529148067211236930"
        "570359138156544e+236"},
    {0x1.3a825c100dd11p+787, chars_format::general, 237,
        "99999999999999994020546131433094915763903576933939556328154082464128816489313932495174721468699049466761532837"
        "20513305603804245824455022623850469957664024826077935002555780941131314090676385002182634786447736977708293139"
        "03654699186257920"},
    {0x1.3a825c100dd12p+787, chars_format::general, 237,
        "1."
        "00000000000000012094235467165686896238200167043557881776819118314224974463086290016415235780369448864354627206"
        "67711366978438164726212832622760464119834231307071911634201289056840812639614702168667161181777994720913003205"
        "4906464259329229e+237"},
    {0x1.8922f31411455p+790, chars_format::general, 238,
        "99999999999999990405808264286576519669044258912015891238421075294109584894559460990926618606364969587242913963"
        "31073693328877462044103460624068471125229983529879139676226679317989414380888721568885729507381685429067351125"
        "745727105048510464"},
    {0x1.8922f31411456p+790, chars_format::general, 238,
        "1."
        "00000000000000004864759732872650104048481530999710551597353103974186511273577347007919030055701289105317389458"
        "88832142428584597165509708623196466454966148714674320981543085810557013220039375302073350623645891623631119178"
        "90900665230478541e+238"},
    {0x1.eb6bafd91596bp+793, chars_format::general, 239,
        "99999999999999999081179145438220670296706622164632687453780292502155740721970192601122065475966761298087599260"
        "65728762788701743116947209423545268323071682640756248459416523213529973684379113808798302177140209145805611957"
        "6436948334022754304"},
    {0x1.eb6bafd91596cp+793, chars_format::general, 239,
        "1."
        "00000000000000010648340320307079537800256439834788415740925915446217281825184501414715994635435816912547179657"
        "11935522068467451214072207822847664586860614788592393503669648407584052755699636795348399070151574101456626400"
        "174318471207295386e+239"},
    {0x1.33234de7ad7e2p+797, chars_format::general, 240,
        "99999999999999982887153500621818255791736877426414667851776420380469583177470160262090564652710083437844186705"
        "61039299797029751780972211664521913553767177633785645397462147941854262984530381627628166526924298207894191738"
        "10082174047524749312"},
    {0x1.33234de7ad7e3p+797, chars_format::general, 240,
        "1."
        "00000000000000001394611380411992443797416585698663833111209417090968048942613054363840851307860572420979515339"
        "94970114644654884736372209103405747575829469070323477468267148252340789498643218406108321555742482136935814846"
        "1498195609632794214e+240"},
    {0x1.7fec216198ddbp+800, chars_format::general, 241,
        "99999999999999990290136652537887930994008760735314333955549619064668969483527317902790679314770279031098318159"
        "34611625736079804963132210640075447162592094208400778225784148066048873590175516339020228538451571779510840981"
        "320420868670460264448"},
    {0x1.7fec216198ddcp+800, chars_format::general, 241,
        "1."
        "00000000000000005096102956370027281398552527353113666163096016433067742095641633184190908638890670217606581066"
        "81756277614179911327452208591182514380241927357631043882428148314438094801465785761804352561506118922744139467"
        "7596191250608858071e+241"},
    {0x1.dfe729b9ff152p+803, chars_format::general, 242,
        "99999999999999993251329913304315801074917514058874200397058898538348724005950180959070725179594357268399970740"
        "84040556111699826235996210230296860606122060838246831357112948115726717832433570223577053343062481208157500678"
        "6082605199485453729792"},
    {0x1.dfe729b9ff153p+803, chars_format::general, 242,
        "1."
        "00000000000000005096102956370027281398552527353113666163096016433067742095641633184190908638890670217606581066"
        "81756277614179911327452208591182514380241927357631043882428148314438094801465785761804352561506118922744139467"
        "759619125060885807104e+242"},
    {0x1.2bf07a143f6d3p+807, chars_format::general, 243,
        "99999999999999988513420696078031208945463508741178414090644051380461116770073600069022651795875832088717326610"
        "44954267510707792199413810885942599096474114230493146346986868036242167044820684008286133655685026122322845162"
        "94771707790360919932928"},
    {0x1.2bf07a143f6d4p+807, chars_format::general, 243,
        "1."
        "00000000000000007465057564983169577463279530011961559316303440012011545713579923629214945330749932807447903132"
        "01299421914675928345743408263359645135065900661507886387491188354180370195272228869449812405194846465661467225"
        "589890846083353893929e+243"},
    {0x1.76ec98994f488p+810, chars_format::general, 244,
        "99999999999999992303748069859058882649026712995335043135775929106771202558774864781061110502850652232463441914"
        "76223298391501419428679730361426008304192471516696094355087732099829807674910992980518869405586990190990569575"
        "476151831539558138249216"},
    {0x1.76ec98994f489p+810, chars_format::general, 244,
        "1."
        "00000000000000007465057564983169577463279530011961559316303440012011545713579923629214945330749932807447903132"
        "01299421914675928345743408263359645135065900661507886387491188354180370195272228869449812405194846465661467225"
        "58989084608335389392896e+244"},
    {0x1.d4a7bebfa31aap+813, chars_format::general, 245,
        "99999999999999992303748069859058882649026712995335043135775929106771202558774864781061110502850652232463441914"
        "76223298391501419428679730361426008304192471516696094355087732099829807674910992980518869405586990190990569575"
        "4761518315395581382492160"},
    {0x1.d4a7bebfa31abp+813, chars_format::general, 245,
        "1."
        "00000000000000004432795665958347438500428966608636256080197937830963477082618911859584178365170076692451010888"
        "56284197210041026562330672682972917768891214832545527981010497103310257691199981691663623805273275210727287695"
        "567143043174594742793011e+245"},
    {0x1.24e8d737c5f0ap+817, chars_format::general, 246,
        "99999999999999987452129031419343460308465811550014557958007125617094292749237245949651883357922882448468414325"
        "24198938864085576575219353432807244518312974190356320904718626098437627668395397496060967645712476183095882327"
        "43975534688554349643169792"},
    {0x1.24e8d737c5f0bp+817, chars_format::general, 246,
        "1."
        "00000000000000006858605185178205149670709417331296498669082339575801931987387721275288791937633961584448524683"
        "32296376973748947989060861147282299661830963495715414706195050104006347694457779433892574685210532214674631319"
        "5853412855016020637017702e+246"},
    {0x1.6e230d05b76cdp+820, chars_format::general, 247,
        "99999999999999995214719492922888136053363253862527334242437211200577348444497436079906646789807314102860458468"
        "47437914107950925140755956518597266575720169912499958425309195700665115678820350271193610461511698595727381924"
        "297989722331966923339726848"},
    {0x1.6e230d05b76cep+820, chars_format::general, 247,
        "1."
        "00000000000000010739900415929977487543158138487552886811297382367543459835017816340416173653576177411644546754"
        "93915864595681622271829162690177310690534561356787233466490334905120091699670255821458896093110143420990381118"
        "0144584732248137771557847e+247"},
    {0x1.c9abd04725480p+823, chars_format::general, 248,
        "99999999999999992109683308321470265755404276937522223728665176967184126166393360027804741417053541441103640811"
        "18142324010404785714541315284281257752757291623642503417072967859774120474650369161140553335192009630674782085"
        "5546959721533975525765152768"},
    {0x1.c9abd04725481p+823, chars_format::general, 248,
        "1."
        "00000000000000004529828046727141746947240184637542665783753313900757015278809664236212362908068632088130911440"
        "35324684400589343419399880221545293044608804779072323450017879223338101291330293601352781840470765490885181440"
        "527870972867675035629361562e+248"},
    {0x1.1e0b622c774d0p+827, chars_format::general, 249,
        "99999999999999992109683308321470265755404276937522223728665176967184126166393360027804741417053541441103640811"
        "18142324010404785714541315284281257752757291623642503417072967859774120474650369161140553335192009630674782085"
        "55469597215339755257651527680"},
    {0x1.1e0b622c774d1p+827, chars_format::general, 249,
        "1."
        "00000000000000011981914889770544635662341729257554931016806196060900748746259446761256935802677686476347273817"
        "85634100634700078042315019183903714219719712672330215469784826041476489781338248265480118943638019007011421053"
        "5117759732962415254610693325e+249"},
    {0x1.658e3ab795204p+830, chars_format::general, 250,
        "99999999999999992109683308321470265755404276937522223728665176967184126166393360027804741417053541441103640811"
        "18142324010404785714541315284281257752757291623642503417072967859774120474650369161140553335192009630674782085"
        "554695972153397552576515276800"},
    {0x1.658e3ab795205p+830, chars_format::general, 250,
        "1."
        "00000000000000008007468573480729761680954238793548389559177992242157424230286229414566496925552857469298547216"
        "52135745309841019576760278403979222926327228462592673059242454405136015920000672444612205821948817131744093259"
        "92035997306767273088415852134e+250"},
    {0x1.bef1c9657a685p+833, chars_format::general, 251,
        "99999999999999992109683308321470265755404276937522223728665176967184126166393360027804741417053541441103640811"
        "18142324010404785714541315284281257752757291623642503417072967859774120474650369161140553335192009630674782085"
        "5546959721533975525765152768000"},
    {0x1.bef1c9657a686p+833, chars_format::general, 251,
        "1."
        "00000000000000004827911520448877862495844246422343156393075429187162764617507655537214145823852994263659565935"
        "45337061049953772804316485780039629891613241094802639130808557096063636830930611787917875324597455631530231025"
        "047227172884817695222629872435e+251"},
    {0x1.17571ddf6c813p+837, chars_format::general, 252,
        "99999999999999989566037665895988746407316283040558037195783126523188398476170500925922860535693650876592455786"
        "32703376602494988296586281185129583324986101729410476274325850012516217203394320635785088937310920430503692297"
        "65618973200711352404729235767296"},
    {0x1.17571ddf6c814p+837, chars_format::general, 252,
        "1."
        "00000000000000009915202805299840901192020234216271529458839530075154219997953373740977907586572775392681935985"
        "16214955865773367640226553978342978747155620883266693416302792790579443373442708838628804120359634031872410600"
        "8442396531773857522810757106893e+252"},
    {0x1.5d2ce55747a18p+840, chars_format::general, 253,
        "99999999999999993635870693776759177364257073275700735648394407233581562780527075488933869945869475779810351826"
        "09405692455150664165314335743772262409420005560181719702721238568128862437403998276353831973920663150777435958"
        "293799716241167969694049028276224"},
    {0x1.5d2ce55747a19p+840, chars_format::general, 253,
        "1."
        "00000000000000009915202805299840901192020234216271529458839530075154219997953373740977907586572775392681935985"
        "16214955865773367640226553978342978747155620883266693416302792790579443373442708838628804120359634031872410600"
        "84423965317738575228107571068928e+253"},
    {0x1.b4781ead1989ep+843, chars_format::general, 254,
        "99999999999999993635870693776759177364257073275700735648394407233581562780527075488933869945869475779810351826"
        "09405692455150664165314335743772262409420005560181719702721238568128862437403998276353831973920663150777435958"
        "2937997162411679696940490282762240"},
    {0x1.b4781ead1989fp+843, chars_format::general, 254,
        "1."
        "00000000000000006659336382995224556426467602028157370696750505506839688554468114090569100058432115470107619153"
        "34853103183648826945244110331428835479608497818649698673586481946089327186234966726173809691071839855653415672"
        "334151665790142195763670374206669e+254"},
    {0x1.10cb132c2ff63p+847, chars_format::general, 255,
        "99999999999999998845256969464145328989141284776683389667736846542884813090103490929587961990894531655929258756"
        "99584656746549929277286245578834891637495402463568911291067335919313048336936385656281823060781133832727827843"
        "90994049606075766012189756664840192"},
    {0x1.10cb132c2ff64p+847, chars_format::general, 255,
        "1."
        "00000000000000019682802072213689935488678130780614005745106603780097814328409152692204330170994755160404886480"
        "60300513912146989725173884919085408549796990077117677644451725324049791935065935175993787408223016560529395386"
        "3745036153391164218332917201371136e+255"},
    {0x1.54fdd7f73bf3bp+850, chars_format::general, 256,
        "99999999999999986342729907814418565089419177174325020021314992200557012347120093872018141082834397553243882122"
        "83155142447191693008553661974684581490114449895439651479036702276471002178058655944454644452316004196046887318"
        "431202624493742403095061074555174912"},
    {0x1.54fdd7f73bf3cp+850, chars_format::general, 256,
        "1."
        "00000000000000003012765990014054250289048653977469512883210797990327413337764623282111235626914576356824384301"
        "71727828179669341366863773446884995019955719986278664561744213800260397056562295560224215930269510378288141352"
        "40285311991642941246417639734614426e+256"},
    {0x1.aa3d4df50af0ap+853, chars_format::general, 257,
        "99999999999999989676737124254345702129345072534953918593694153358511092545248999754036759991650433313959982558"
        "60869679593687222680215684269124664196082703913607454095578204581228881153759383867608558747906705432495138125"
        "2255327235782798049688841391133687808"},
    {0x1.aa3d4df50af0bp+853, chars_format::general, 257,
        "1."
        "00000000000000003012765990014054250289048653977469512883210797990327413337764623282111235626914576356824384301"
        "71727828179669341366863773446884995019955719986278664561744213800260397056562295560224215930269510378288141352"
        "402853119916429412464176397346144256e+257"},
    {0x1.0a6650b926d66p+857, chars_format::general, 258,
        "99999999999999984342325577950462282865463639957947680877887495505784564228242750342806969737544776096814221861"
        "36526420159294375205556448598020531866533497484538969909111800893616274792638219190562295874961583454177936834"
        "35460456504301996197076723582025859072"},
    {0x1.0a6650b926d67p+857, chars_format::general, 258,
        "1."
        "00000000000000005679971763165995959920989370265972631741114126916690677496267747987726130753967404965397264650"
        "33899457896865765104193391282437061184730323200812906654977415644066700237122877898747347366742071367446741997"
        "838317199184059333963234848992699351e+258"},
    {0x1.4cffe4e7708c0p+860, chars_format::general, 259,
        "99999999999999992877384052036675753687673932081157661223178148070147009535452749400774634144113827644247438976"
        "95475635254322931165011225671787143593812227771048544607458046793796444970432082673836316471673778619485458899"
        "748089618699435710767754281089234894848"},
    {0x1.4cffe4e7708c1p+860, chars_format::general, 259,
        "1."
        "00000000000000009947501000209102695332094516327577621913759453198871900149872747516709962957251930739113873208"
        "13374065444380043083920779819320367048369688344067694004150538594156785326019809640384357665098168950100503030"
        "5350597260122672083617283716271875031e+259"},
    {0x1.a03fde214caf0p+863, chars_format::general, 260,
        "99999999999999992877384052036675753687673932081157661223178148070147009535452749400774634144113827644247438976"
        "95475635254322931165011225671787143593812227771048544607458046793796444970432082673836316471673778619485458899"
        "7480896186994357107677542810892348948480"},
    {0x1.a03fde214caf1p+863, chars_format::general, 260,
        "1."
        "00000000000000006533477610574617307003210399478293629775643192173126922026988747893522897194624310120140586361"
        "89794379406368620700138868989813722357458196229463864124812040234084717254902264247074749426413290883977494204"
        "377665704549700908842933553519596981453e+260"},
    {0x1.0427ead4cfed6p+867, chars_format::general, 261,
        "99999999999999992877384052036675753687673932081157661223178148070147009535452749400774634144113827644247438976"
        "95475635254322931165011225671787143593812227771048544607458046793796444970432082673836316471673778619485458899"
        "74808961869943571076775428108923489484800"},
    {0x1.0427ead4cfed7p+867, chars_format::general, 261,
        "1."
        "00000000000000014727133745697382238992532279916575210907122218634914869521910346989171855024930599605676474792"
        "86385625897596034421215454980629669615645777304513055835224436298257680625584373191017809199256998242672715387"
        "1554113560598600276880411169778142334157e+261"},
    {0x1.4531e58a03e8bp+870, chars_format::general, 262,
        "99999999999999984137484174572393159565730592946990641349600519844239865540869710365415745791787118859675824650"
        "59111638997013689862529533948250133185078807957662740116351490992011950708371166466963719380640490770210556304"
        "785160923755265983999639546733803159420928"},
    {0x1.4531e58a03e8cp+870, chars_format::general, 262,
        "1."
        "00000000000000001617283929500958347809617271215324681096755776296054153530035788436133522496440536428819053303"
        "31839631511632172467492917395324154002545647584434349098564602595580939232492998880708913562707066468760361494"
        "71101831364360543753586901544466663027507e+262"},
    {0x1.967e5eec84e2ep+873, chars_format::general, 263,
        "99999999999999987633444125558106197214507928600657449299031571134602723138702925979559301132717802373504470381"
        "13657237499937386383522210637664937348572175883017061912794113312725748413195532949712758217053805909920517342"
        "7703324017329338747068854404759758535917568"},
    {0x1.967e5eec84e2fp+873, chars_format::general, 263,
        "1."
        "00000000000000001617283929500958347809617271215324681096755776296054153530035788436133522496440536428819053303"
        "31839631511632172467492917395324154002545647584434349098564602595580939232492998880708913562707066468760361494"
        "711018313643605437535869015444666630275072e+263"},
    {0x1.fc1df6a7a61bap+876, chars_format::general, 264,
        "99999999999999993226980047135247057452551665646524342018121253199183295295236070962188989678206895995630303550"
        "00930195104615300817110493340728624010161564563583976787102309025867824740914519322111220355315110133456455003"
        "54660676649720249983847887046345216426508288"},
    {0x1.fc1df6a7a61bbp+876, chars_format::general, 264,
        "1."
        "00000000000000004414051890289528777928639139738258127456300617328344439608302360927448366769185083239881969887"
        "75476110313971129684287058746855997333340341924717806535718700452151977396352492066908144631837718580528330325"
        "099155496025739750101665730438404785611735e+264"},
    {0x1.3d92ba28c7d14p+880, chars_format::general, 265,
        "99999999999999988752151309873534369262116676009830827842849507547518837570009554976085238841815621097929637014"
        "91111829020872969270239867178277674680890053619130444887655752455354163678739330224192450644706066754627704874"
        "925587274685787599733204126473471115726422016"},
    {0x1.3d92ba28c7d15p+880, chars_format::general, 265,
        "1."
        "00000000000000006651466258920385122023856634556604884543936490154176668470915618920500242187380720688732303155"
        "30385293355842295457722371828081471997976097396944572485441978737408807927440086615867529487142240269942705389"
        "40966524193144720015430310243339530988106547e+265"},
    {0x1.8cf768b2f9c59p+883, chars_format::general, 266,
        "99999999999999988752151309873534369262116676009830827842849507547518837570009554976085238841815621097929637014"
        "91111829020872969270239867178277674680890053619130444887655752455354163678739330224192450644706066754627704874"
        "9255872746857875997332041264734711157264220160"},
    {0x1.8cf768b2f9c5ap+883, chars_format::general, 266,
        "1."
        "00000000000000003071603269111014971471508642847250073203719093632845102290734406131617241518267700770571769927"
        "22530600488848430220225870898120712534558888641381746965884733480997879077699935337532513718655005566879705286"
        "512849648482315280070083307241410471050136781e+266"},
    {0x1.f03542dfb8370p+886, chars_format::general, 267,
        "99999999999999997343822485416022730587751856112282375059371259198714596402444465669404440447686868901514916762"
        "29963091901658245840231469410183497393091354632481226134593141070740392918115693292196488489075430041978905121"
        "87794469896370420793533163493423472892065087488"},
    {0x1.f03542dfb8371p+886, chars_format::general, 267,
        "1."
        "00000000000000008799384052806007212355265429582217771348066928066975608179024346593830042588848532639628623092"
        "15098109076038614600220272386057927676026422650282267797176325891255365237284177382868538948234581091780505451"
        "1477545980009263522048349795485862131796226867e+267"},
    {0x1.362149cbd3226p+890, chars_format::general, 268,
        "99999999999999997343822485416022730587751856112282375059371259198714596402444465669404440447686868901514916762"
        "29963091901658245840231469410183497393091354632481226134593141070740392918115693292196488489075430041978905121"
        "877944698963704207935331634934234728920650874880"},
    {0x1.362149cbd3227p+890, chars_format::general, 268,
        "1."
        "00000000000000015672720993239997901415773573664179009121284329387932215244972275148485403873545530882496846890"
        "06179119380666835856213554171582585845787463460962892794726236783564348628785267837271769223730071721661465648"
        "70964053742325963876653698631719710373500577382e+268"},
    {0x1.83a99c3ec7eafp+893, chars_format::general, 269,
        "99999999999999990012263082286432662256543169091523721434606031123027548865433341877772055077343404109122144711"
        "19476680910054809833838635505623862012012911101088559470539902785610810633847863474166376195213573370105880911"
        "1452663635798820356028494943810497789949089153024"},
    {0x1.83a99c3ec7eb0p+893, chars_format::general, 269,
        "1."
        "00000000000000004675381888545612798918960543133041028684136487274401643939455589461036825818030333693907688813"
        "40449502893261681846624303314743132774169798163873892798646379355869975202383523110226600782937286713851929332"
        "610623034347526380267813775487419678846392834458e+269"},
    {0x1.e494034e79e5bp+896, chars_format::general, 270,
        "99999999999999992944886843538268689589026643899827182884512122353302367880237791394425009225480790026079253531"
        "63671245306696184236395769067447716164444288513645626136161198099662643547554995401378421112758316038855090595"
        "43833769773341090453584235060232375896520569913344"},
    {0x1.e494034e79e5cp+896, chars_format::general, 270,
        "1."
        "00000000000000004675381888545612798918960543133041028684136487274401643939455589461036825818030333693907688813"
        "40449502893261681846624303314743132774169798163873892798646379355869975202383523110226600782937286713851929332"
        "6106230343475263802678137754874196788463928344576e+270"},
    {0x1.2edc82110c2f9p+900, chars_format::general, 271,
        "99999999999999995290985852539737511455013423746469952044436995337522223092081351007747372543990698759644940587"
        "99026896824009283758441475916906799486389390443691279468658234350904109878520700943148057046794110173854458342"
        "872794765056233999682236635579342942941443126198272"},
    {0x1.2edc82110c2fap+900, chars_format::general, 271,
        "1."
        "00000000000000014059777924551488086382907662519612105323835979211281064786829827914326279092069968628170437038"
        "81872108962514079934807130712579466061950205884056506128634524360835840526246345277305144519080463253849400322"
        "34845130363881876085339091539549641475134254271693e+271"},
    {0x1.7a93a2954f3b7p+903, chars_format::general, 272,
        "99999999999999991537227438137387396469434575991841521388557198562770454753131655626431591234374844785939841297"
        "82457854396308324523168344957772266171277227355618234136662976348917763748975572076316639552336839557855469946"
        "9776634573397170474480057796161122485794632428945408"},
    {0x1.7a93a2954f3b8p+903, chars_format::general, 272,
        "1."
        "00000000000000006552261095746787856411749967010355244012076385661777528108930437151694716472838260680760238458"
        "48734024107112161464260868794310399431725879707910415464644008356863148267156087543642309530165922021851423530"
        "558188688205784856384929203469035026027382776109466e+272"},
    {0x1.d9388b3aa30a5p+906, chars_format::general, 273,
        "99999999999999994540234169659267488457897654195544265913261035982571869424291411931484216282067527964903920729"
        "95713088338469091911386849725079892823366957826076670402259182750506840652611675169781773547902656050654660663"
        "69376850351293060923539046438669680406904714953752576"},
    {0x1.d9388b3aa30a6p+906, chars_format::general, 273,
        "1."
        "00000000000000006552261095746787856411749967010355244012076385661777528108930437151694716472838260680760238458"
        "48734024107112161464260868794310399431725879707910415464644008356863148267156087543642309530165922021851423530"
        "5581886882057848563849292034690350260273827761094656e+273"},
    {0x1.27c35704a5e67p+910, chars_format::general, 274,
        "99999999999999992137828784441763414867127191632582070293497966046730737687363606887442116243913381421732657184"
        "25108901184740478000812045911233791501695173449709921389782217629235579129702792695009666351450002856415308090"
        "320884466574359759805482716570229159677380024223137792"},
    {0x1.27c35704a5e68p+910, chars_format::general, 274,
        "1."
        "00000000000000011357071866181796003593290892136279635251602525533459791582786047239778916549146553767102765549"
        "89942398414569389285410476422002602075069448460643913489597938599405671312973852493186523923071228410330128677"
        "30395676208292655524474469910197031481071702673824154e+274"},
    {0x1.71b42cc5cf601p+913, chars_format::general, 275,
        "99999999999999995981677400789769932612359931733321583285118877944076548466448094957909476304960015890806678857"
        "38075600630706260257731732013387553616370028451896719809745361823269597566357004654645037865774247967198272207"
        "7174989256760731188933351130765773907040474247261585408"},
    {0x1.71b42cc5cf602p+913, chars_format::general, 275,
        "1."
        "00000000000000011357071866181796003593290892136279635251602525533459791582786047239778916549146553767102765549"
        "89942398414569389285410476422002602075069448460643913489597938599405671312973852493186523923071228410330128677"
        "303956762082926555244744699101970314810717026738241536e+275"},
    {0x1.ce2137f743381p+916, chars_format::general, 276,
        "99999999999999992906598507711364718416173739652729972891822148426199899843180504501535588256122708315547461518"
        "87702241073933634452195983131664543924630144450147281073774846468042382817033635086936740654314851878571900913"
        "80020735839470243162305319587149880588271350432374194176"},
    {0x1.ce2137f743382p+916, chars_format::general, 276,
        "1."
        "00000000000000005206914080024985575200918507975096414465009066497706494336250866327031140451471938616584330872"
        "89195679301024137674338978658556582691589680457145036017656907888951241814327113357769929500152436233077386089"
        "4693736275201851807041808646918131451680491859334083379e+276"},
    {0x1.20d4c2fa8a030p+920, chars_format::general, 277,
        "99999999999999980606282935397743861631428971330363531318635230354693305350110142676040036060773478014510592164"
        "86208802846843131230052987604772505157670608443149526129892785047133523819740156816103551808477267524066415738"
        "131041089269219682541925527051184466597377822714075545600"},
    {0x1.20d4c2fa8a031p+920, chars_format::general, 277,
        "1."
        "00000000000000000286787851099537232487020600646149837835734299269103856539022721596832919573332246496169583131"
        "28598304010187936385481780447799767184805866054345934040104083320587698215409722049436653961817402491275192019"
        "20170711986999208107172979716368740945391491328954177946e+277"},
    {0x1.6909f3b92c83dp+923, chars_format::general, 278,
        "99999999999999996350686867959178558315902274782992576532314485486221746301240205812674342870820492799837784938"
        "00120403777518975354396021879194314779378814532106652458061823665896863336275809002770033531149375497833436762"
        "9875739137498376013657689431411868208826074951744485326848"},
    {0x1.6909f3b92c83ep+923, chars_format::general, 278,
        "1."
        "00000000000000012095090800520613255000375578235621621745993740617750187252370268949308649680867507585164977711"
        "14032004708194819478739056153616124401087020621063778786230862284660202852811461189436515253821483471600457787"
        "84410673823045552018961235923118917516783716763482151977e+278"},
    {0x1.c34c70a777a4cp+926, chars_format::general, 279,
        "99999999999999993201806081446891618979007614092466767489578634459916058111014193185347481508811089842772346383"
        "37338083591383806529527415024309952855037173314315227192428015942144195432968678565436737186614953903080032558"
        "01626734885371401760100025992318635002556156068237393526784"},
    {0x1.c34c70a777a4dp+926, chars_format::general, 279,
        "1."
        "00000000000000005797329227496039376326586256854570003660522038565138810871918243694654926956848701671034100601"
        "88467364335924481829001842443847400552403738185480928254963246837154867046197200314769922564752640282093649377"
        "9014936084382083526600749927951882334537452986506723249357e+279"},
    {0x1.1a0fc668aac6fp+930, chars_format::general, 280,
        "99999999999999983125387564607573413100944699882784178552823911175737855902290952777901525150381000380162943008"
        "56434658995751266289947873088679994697143921417382666342399831226135658142385861165970188884104804799869139102"
        "108086341186118549553740473625584843283014570307735223533568"},
    {0x1.1a0fc668aac70p+930, chars_format::general, 280,
        "1."
        "00000000000000003278224598286209824857070528302149356426333357744094260319737433592793437867241179305381749758"
        "18241508187016346769106956959939911012930425211247788042456200658152732723551495964903285489125103006290926013"
        "92444835652130948564826004622078785676810855105701264700211e+280"},
    {0x1.6093b802d578bp+933, chars_format::general, 281,
        "99999999999999987155954971343300695452169865566657214127525800489409136785780248940879907693753036165206704358"
        "48796028834004282385779689862931977960301222176155690682411105112539073058618988125756808205108864441153496484"
        "4713587442531567367726443881446254459800333664575907082272768"},
    {0x1.6093b802d578cp+933, chars_format::general, 281,
        "1."
        "00000000000000003278224598286209824857070528302149356426333357744094260319737433592793437867241179305381749758"
        "18241508187016346769106956959939911012930425211247788042456200658152732723551495964903285489125103006290926013"
        "924448356521309485648260046220787856768108551057012647002112e+281"},
    {0x1.b8b8a6038ad6ep+936, chars_format::general, 282,
        "99999999999999990380408896731882521333149998113755642587287311940346161492571685871262613728450664793241713438"
        "42685124704606695262445143282333564570827062783174110154420124221661804991605489693586103661912112154180982390"
        "36197666670678728654776751975985792813764840337747509598224384"},
    {0x1.b8b8a6038ad6fp+936, chars_format::general, 282,
        "1."
        "00000000000000003278224598286209824857070528302149356426333357744094260319737433592793437867241179305381749758"
        "18241508187016346769106956959939911012930425211247788042456200658152732723551495964903285489125103006290926013"
        "924448356521309485648260046220787856768108551057012647002112e+282"},
    {0x1.137367c236c65p+940, chars_format::general, 283,
        "99999999999999995539535177353613442742718210189113128122905730261845401023437984959874943383966870598097727966"
        "32907678097570555865109868753376103147668407754403581309634554796258176084383892202112976392797308495024959839"
        "786965342632596166187964530344229899589832462449290116390191104"},
    {0x1.137367c236c66p+940, chars_format::general, 283,
        "1."
        "00000000000000016176040299840537128380991058490543070265379403547842359146903181314324262006031693817521786077"
        "93797891669425998275768770637546257455033787639321465930492277094643660455497502236220467316338093858400869637"
        "48692004633583168474875257268171778539856869873655019802198016e+283"},
    {0x1.585041b2c477ep+943, chars_format::general, 284,
        "99999999999999991412234152856228705615063640528827139694410995604646009398744945688985079659553905954212916344"
        "00729635383199467382978088376542072286195331777420004385463010336581079210161170195291478208089151422349777880"
        "2469744018919490624758069218767323224280852151918381000638332928"},
    {0x1.585041b2c477fp+943, chars_format::general, 284,
        "1."
        "00000000000000007921438250845767654125681919169971093408389934233443575897517102772544534557205764529752162833"
        "29441806240683821311505209883878195732087635685354312082149188175289466707052058222577470946921779713050505718"
        "406938164854537477324437355746722631075074204221646165369264538e+284"},
    {0x1.ae64521f7595ep+946, chars_format::general, 285,
        "99999999999999998015915792052044285019310951985284721180002571056165035998253808522408861618614649384428614939"
        "72214503726193208954388936979476521664552253340593727464137481472064434208917525406205875303622202738630069015"
        "51095990707698442841525909542472844588688081080376132618600579072"},
    {0x1.ae64521f7595fp+946, chars_format::general, 285,
        "1."
        "00000000000000011223279070443675443827805574898199884151185721959203089197271534189256425536736136244860012131"
        "15184240412180692097210634185345420421266096466941173621486423743031144206430235828034669494688305371190651286"
        "0389309174470551602941634425207206928044720020276077784303507866e+285"},
    {0x1.0cfeb353a97dap+950, chars_format::general, 286,
        "99999999999999982167079857982086894449117404489786525614582789972519372159432537722191784916868865151910938310"
        "00650819703008229183002900332433843156495641588976792075318750746904382211902272900011322274342879579557370290"
        "877394694632899550160573878909537749585771381335145583492791795712"},
    {0x1.0cfeb353a97dbp+950, chars_format::general, 286,
        "1."
        "00000000000000003298861103408696748542708801150450786368475831417380257277860898789147887185863244128601173816"
        "29402398400588202211517615861824081167237790591132705927077058380451118207922609574937392980048643791654301923"
        "72214831122501272116682083426312534465391728729329990708374378906e+286"},
    {0x1.503e602893dd1p+953, chars_format::general, 287,
        "99999999999999990619792356152730836086553963154052229916140006550463726206803882148974225824466616742587032512"
        "52151451182040218394408786544189938360792501189839157616022073800323076610310407569981750556625185264396142944"
        "0152961412697448185630726610509727876130297437184073129291725930496"},
    {0x1.503e602893dd2p+953, chars_format::general, 287,
        "1."
        "00000000000000007525217352494018719361427080482583638519254439706352434301546571002539107639662119923939220917"
        "55152714140104196817220558967702128769386220391563888697428719907160465407126676909922607121189796634073688250"
        "291099034543435355368070225333842863667546468484930771801934187725e+287"},
    {0x1.a44df832b8d45p+956, chars_format::general, 288,
        "99999999999999987238707356884473259431579339688345948195517119919285984587855344378261249461427516106316594831"
        "51551198590427422709846432059487500279073757349494211399740744578955598850947153701993579243712262990460633882"
        "76013556261500671120207314819439877240212639876510262115462027411456"},
    {0x1.a44df832b8d46p+956, chars_format::general, 288,
        "1."
        "00000000000000000763047353957503566051477833551171075078008666443996951063649495461113154913583918651398345555"
        "53952208956878605448095849998297252605948732710873996264866061464425509888400169173946264495363952086202670127"
        "7807778772339591406460711996206948332457397785783213882528295498547e+288"},
    {0x1.06b0bb1fb384bp+960, chars_format::general, 289,
        "99999999999999984533839357469867198107599640915780922819018810614343791292696514161690868370996235597300244686"
        "71070996517137186162196548471725549813698762277218254426715681201861616643456550607603042193381925171312226633"
        "756007099691216225313273537909139560233403722802458867734978418966528"},
    {0x1.06b0bb1fb384cp+960, chars_format::general, 289,
        "1."
        "00000000000000006172783352786715688699437231096301125831005285053881337653967155894253917094446479669431045845"
        "14912613103459078543395617173821153536698722855425910210916188218613474303381375362727338596024627724499484625"
        "78903480308154011242367042019121325758318513050360889509211326015078e+289"},
    {0x1.485ce9e7a065ep+963, chars_format::general, 290,
        "99999999999999988861628156533236896225967158951884963421416105502251300564950642508203478115686284411726404918"
        "39839319834401564638436362212144670558298754392859785583555782605211988175441515558627901473910465681949678232"
        "1626126403692810027353529143655542997033600043426888732064053872033792"},
    {0x1.485ce9e7a065fp+963, chars_format::general, 290,
        "1."
        "00000000000000006172783352786715688699437231096301125831005285053881337653967155894253917094446479669431045845"
        "14912613103459078543395617173821153536698722855425910210916188218613474303381375362727338596024627724499484625"
        "789034803081540112423670420191213257583185130503608895092113260150784e+290"},
    {0x1.9a742461887f6p+966, chars_format::general, 291,
        "99999999999999995786090235034628413215355187809651428385251777322903315400557247862623653707190362514808261289"
        "09868637142024570200420064196815263749658741777886235434499944850572582626617459480267676322756130498969600789"
        "61318150545418464661067991669581788285529005480705688196068853638234112"},
    {0x1.9a742461887f7p+966, chars_format::general, 291,
        "1."
        "00000000000000009635014392037411447194131245525184358312923120964207345071770458571464004890198518720971974030"
        "49927271757270581324387468166156450132378716547939135136388269341293771528969347323547226020447460133009445904"
        "514319235623991934361333921356345049159150155735792899469254834740265e+291"},
    {0x1.008896bcf54f9p+970, chars_format::general, 292,
        "99999999999999979167381246631288772440823918551011912472046164953338479795101395012015232287580575067411805999"
        "41798275603729356851659179433605840090394772053822755792233955461707155943795194068332216685526534938121786651"
        "731816229250415901309895111103185283290657933692573660950408978352832512"},
    {0x1.008896bcf54fap+970, chars_format::general, 292,
        "1."
        "00000000000000001325659897835741626806865610895864600356320314779424927269042532146159794180393624997273746385"
        "65892090988122974650007025784551738302746731685907395315255274646861058187558214617579496201832662352585538835"
        "57363659752210756171094151856002874937683409517855128896411505572551066e+292"},
    {0x1.40aabc6c32a38p+973, chars_format::general, 293,
        "99999999999999992462348437353960485060448933957923525202610654848990348279466077292501969423268405025328970231"
        "16254564834365527530667887244173379017805947833073539506046746972799497290053006397880584395310211386800037962"
        "0369084502134308975505229555772913629423636305841602377586326247764393984"},
    {0x1.40aabc6c32a39p+973, chars_format::general, 293,
        "1."
        "00000000000000010188971358317522768553282287833805675510029974709859506258618986999817618937518844969218522540"
        "15529617141880421769346164324930097587687515538741251124463802320922619085063422837278408008355113318371039709"
        "110364744830784225871360081542766135811304559772942340169597486674581914e+293"},
    {0x1.90d56b873f4c6p+976, chars_format::general, 294,
        "99999999999999992462348437353960485060448933957923525202610654848990348279466077292501969423268405025328970231"
        "16254564834365527530667887244173379017805947833073539506046746972799497290053006397880584395310211386800037962"
        "03690845021343089755052295557729136294236363058416023775863262477643939840"},
    {0x1.90d56b873f4c7p+976, chars_format::general, 294,
        "1."
        "00000000000000006643646774124810311854715617058629245448546110737685674662788405058354489034668756980440612078"
        "35674606680377442921610508908778753873711201997607708800780391251297994726061339549398843285746132932056839359"
        "6956734859073135602071926563496711812375163739351859196874045142949534106e+294"},
    {0x1.f50ac6690f1f8p+979, chars_format::general, 295,
        "99999999999999998134867772062300415778155607198205813300984837204468478832795008398842977267828545807373626970"
        "04022581572770293687044935910015528960168049498887207223940204684198896264456339658487887951484580004902758521"
        "100414464490983962613190835886243290260424727924570510530141380583845003264"},
    {0x1.f50ac6690f1f9p+979, chars_format::general, 295,
        "1."
        "00000000000000009479906441478980277213568953678770389497733201915424739939452870611524992956948827371462940447"
        "79558615049579825999799033241699828844892252830514542659727120106997694213263006179702495063833317241108199639"
        "22742649304609009273852659650414714489654692260539105607315889219865621299e+295"},
    {0x1.3926bc01a973bp+983, chars_format::general, 296,
        "99999999999999998134867772062300415778155607198205813300984837204468478832795008398842977267828545807373626970"
        "04022581572770293687044935910015528960168049498887207223940204684198896264456339658487887951484580004902758521"
        "1004144644909839626131908358862432902604247279245705105301413805838450032640"},
    {0x1.3926bc01a973cp+983, chars_format::general, 296,
        "1."
        "00000000000000016286929643128988194074816961567109135215782220741998496603447587939134202370420996309916528534"
        "44880235135665545387451491640710408775726774829490943921199269360676972982547006092431259331242559582831464310"
        "103633710179153770813728052874889457678220239413883383398969399167542938829e+296"},
    {0x1.87706b0213d09p+986, chars_format::general, 297,
        "99999999999999987243630649422287748800158794576863820152106407081950468170403460674668242206273075505847886031"
        "39507989435033142666801002471598601070832814300524965205584765878312050233601939798121865123629792258145535047"
        "69848291707808207769286850569305558980974742103098278680884456943362624192512"},
    {0x1.87706b0213d0ap+986, chars_format::general, 297,
        "1."
        "00000000000000001765280146275637971437487878071986477683944313911974482386925524306901222288347035907882207282"
        "92194112285349344027126247056154504923279794565007954563392017619494511608074472945276562227436175920488499678"
        "901058313628617924253298279283972523743983830222433085103906984300584590377e+297"},
    {0x1.e94c85c298c4cp+989, chars_format::general, 298,
        "99999999999999995956620347534297882382556244673937414671209151179964876700316698854008030255517451747068478782"
        "31119663145222863482996149222332143382301002459214758820269116923021527058285459686414683385913622455551313826"
        "420028155008403585629126369847605750170289266545852965785882018353801250996224"},
    {0x1.e94c85c298c4dp+989, chars_format::general, 298,
        "1."
        "00000000000000007573939945016978060492419511470035540696679476643984088073534349759794414321176620068695935783"
        "53268561425475824571256344889976866464258586670801150306514918315967496157863486204138441068958729385425685531"
        "3820884722488322628774701887203392973176783938990132044219319502473679297577e+298"},
    {0x1.31cfd3999f7afp+993, chars_format::general, 299,
        "99999999999999986662764669548153739894665631237058913850832890808749507601742578129378923002990117089766513181"
        "33400544521020494612387992688216364916734935089945645631272475808664751778623038472235677239477536911651816462"
        "4503799012160606438304513147494189124523779646633247748770420728389479079870464"},
    {0x1.31cfd3999f7b0p+993, chars_format::general, 299,
        "1."
        "00000000000000005250476025520442024870446858110815915491585411551180245798890819578637137508044786404370444383"
        "28838781769425232353604305756447921847867069828483872009265758037378302337947880900593689532349707999450811190"
        "389676408800746527427801424945792587888200568428381156694721963868654594005402e+299"},
    {0x1.7e43c8800759bp+996, chars_format::general, 300,
        "99999999999999990380306940742611396889821876611810314178983394957235655241172226419230565904001050952687299421"
        "72488191970701442160631255301862676302961362037653290906871132254407461890488006957907279698051971129211615408"
        "03823920273299782054992133678869364753954248541633605124057805104488924519071744"},
    {0x1.7e43c8800759cp+996, chars_format::general, 300,
        "1."
        "00000000000000005250476025520442024870446858110815915491585411551180245798890819578637137508044786404370444383"
        "28838781769425232353604305756447921847867069828483872009265758037378302337947880900593689532349707999450811190"
        "3896764088007465274278014249457925878882005684283811566947219638686545940054016e+300"},
    {0x1.ddd4baa009302p+999, chars_format::general, 301,
        "99999999999999993354340757698177522485946872911611434441503798276024573352715945051111880224809798043023928414"
        "03758309930446200199225865392779725411942503595819407127350057411001629979979981746444561664911518503259454564"
        "508526643946547561925497354420113435609274102018745072331406833609642314953654272"},
    {0x1.ddd4baa009303p+999, chars_format::general, 301,
        "1."
        "00000000000000005250476025520442024870446858110815915491585411551180245798890819578637137508044786404370444383"
        "28838781769425232353604305756447921847867069828483872009265758037378302337947880900593689532349707999450811190"
        "3896764088007465274278014249457925878882005684283811566947219638686545940054016e+301"},
    {0x1.2aa4f4a405be1p+1003, chars_format::general, 302,
        "99999999999999988595886650569271721532146878831929642021471152965962304374245995240101777311515802698485322026"
        "33726121194854587337474489247312446837572677102753621174583777160450961036792822084784910517936242704782911914"
        "1560667380048679757245757262098417746977035154548906385860807815060374033329553408"},
    {0x1.2aa4f4a405be2p+1003, chars_format::general, 302,
        "1."
        "00000000000000007629703079084894925347346855150656811701601734206211380288125794484142188964691784076639747577"
        "13854876137221038784479993829181561135051983075016764985648898162653636809541460731423515105837345898689082515"
        "565906361771586320528262239050928418343985861710308373567384989920457049815751066e+302"},
    {0x1.754e31cd072d9p+1006, chars_format::general, 303,
        "99999999999999984789123364866147080769106883568184208085445036717912489191470035391293694980880606422854436916"
        "17700370206381297048073388330938623978076815908300992412370752960010425882243094355457189600356022066001677793"
        "87409881325152430676383842364162444596844704620380709158981993982315347403639619584"},
    {0x1.754e31cd072dap+1006, chars_format::general, 303,
        "1."
        "00000000000000000016176507678645643821266864623165943829549501710111749922573874786526024303421391525377977356"
        "81803374160274458205677791996433915416060260686111507461222849761772566500442005272768073270676904621126614275"
        "0019705122648989826067876339144937608854729232081412795748633065546891912226327757e+303"},
    {0x1.d2a1be4048f90p+1009, chars_format::general, 304,
        "99999999999999993925355250553646218600402872201173249531907715713232045630132339028433092574405077484368561180"
        "56162172578717193742636030530235798840866882774987301441682011041067710253162440905843719802548551599076639682"
        "550821832659549112269607949805346034918662572406407604380845959862074904348138143744"},
    {0x1.d2a1be4048f91p+1009, chars_format::general, 304,
        "1."
        "00000000000000006106997764803645069042130857045158638127191287706991454215015410544618956032437705566387393533"
        "07444575741831722668719553462632031991253638597235713480763688482477422747721569639692426738805257643176588867"
        "45311919187024885294396731802364148685228327400987495476888065324730347809712740762e+304"},
    {0x1.23a516e82d9bap+1013, chars_format::general, 305,
        "99999999999999993925355250553646218600402872201173249531907715713232045630132339028433092574405077484368561180"
        "56162172578717193742636030530235798840866882774987301441682011041067710253162440905843719802548551599076639682"
        "5508218326595491122696079498053460349186625724064076043808459598620749043481381437440"},
    {0x1.23a516e82d9bbp+1013, chars_format::general, 305,
        "1."
        "00000000000000013415983273353644379307167647951549871284361430903247099365945253454330474107257282415598692944"
        "58214017639700440024369667222069771881485692090584760704212694947323250244457046880001650900559281269636558378"
        "394497607396668697348582938954618758012455694971955365001701469278440622346520965939e+305"},
    {0x1.6c8e5ca239028p+1016, chars_format::general, 306,
        "99999999999999986129104041433646954317696961901022600830926229637226024135807173258074139961264195511876508474"
        "95341434554323895229942575853502209624619359048748317736669737478565494256644598516180547363344259730852672204"
        "21335152276470127823801795414563694568114532338018850013250375609552861714878501486592"},
    {0x1.6c8e5ca239029p+1016, chars_format::general, 306,
        "1."
        "00000000000000001721606459673645482883108782501323898232889201789238067124457504798792045187545959456860613886"
        "16982910603110492255329485206969388057114406501226285146694284603569926249680283295506892241752843467300607160"
        "8882921425543969463011979454650551241561798214326267086291881636286211915474912726221e+306"},
    {0x1.c7b1f3cac7433p+1019, chars_format::general, 307,
        "99999999999999998603105976025645777170026418381263638752496607358835658526727438490648464142289606667863792803"
        "92654615393353172850252103336275952370615397010730691664689375178569039851073146339641623266071126720011020169"
        "553304018596457812688561947201171488461172921822139066929851282122002676667750021070848"},
    {0x1.c7b1f3cac7434p+1019, chars_format::general, 307,
        "1."
        "00000000000000011077107910617644600022355874861504676674066985080445292917647703723222788323315017823851077132"
        "89967796232382450470561630819049695116611434972713065592709012878572585445501694163102699168797993709169368134"
        "89325651442821434713910594025670603124120052026408963372719880814847673618671502727578e+307"},
    {0x1.1ccf385ebc89fp+1023, chars_format::general, 308,
        "99999999999999981139503267596847425176765179308926185662298078548582170379439067165044410288854031049481594743"
        "36416162218712184181818764860392712526220943863955368165461882398564076018873179386796117002253512935189333018"
        "0773705244319986644578003569234231285691342840034082734135647456849389933411990123839488"},
    {0x1.1ccf385ebc8a0p+1023, chars_format::general, 308,
        "1."
        "00000000000000001097906362944045541740492309677311846336810682903157585404911491537163328978494688899061249669"
        "72117251561159028374314008832830700919814604603127166450293302718569748969958855904333838446616500117842689762"
        "621294517762809119578670745812278397017178441510529180289320787327297488571543022311834e+308"},
    {0x1.a36e2eb1c432cp-14, chars_format::general, 309,
        "9.99999999999999912396464463171241732197813689708709716796875e-05"},
    {0x1.a36e2eb1c432dp-14, chars_format::general, 309,
        "0.000100000000000000004792173602385929598312941379845142364501953125"},
    {0x1.0624dd2f1a9fbp-10, chars_format::general, 309,
        "0.00099999999999999980397624721462079833145253360271453857421875"},
    {0x1.0624dd2f1a9fcp-10, chars_format::general, 309,
        "0.001000000000000000020816681711721685132943093776702880859375"},
    {0x1.47ae147ae147ap-7, chars_format::general, 309, "0.0099999999999999984734433411404097569175064563751220703125"},
    {0x1.47ae147ae147bp-7, chars_format::general, 309, "0.01000000000000000020816681711721685132943093776702880859375"},
    {0x1.9999999999999p-4, chars_format::general, 309, "0.09999999999999999167332731531132594682276248931884765625"},
    {0x1.999999999999ap-4, chars_format::general, 309, "0.1000000000000000055511151231257827021181583404541015625"},
    {0x1.fffffffffffffp-1, chars_format::general, 309, "0.99999999999999988897769753748434595763683319091796875"},
    {0x1.0000000000000p+0, chars_format::general, 309, "1"},
    {0x1.3ffffffffffffp+3, chars_format::general, 309, "9.9999999999999982236431605997495353221893310546875"},
    {0x1.4000000000000p+3, chars_format::general, 309, "10"},
    {0x1.8ffffffffffffp+6, chars_format::general, 309, "99.9999999999999857891452847979962825775146484375"},
    {0x1.9000000000000p+6, chars_format::general, 309, "100"},
    {0x1.f3fffffffffffp+9, chars_format::general, 309, "999.9999999999998863131622783839702606201171875"},
    {0x1.f400000000000p+9, chars_format::general, 309, "1000"},
    {0x1.387ffffffffffp+13, chars_format::general, 309, "9999.999999999998181010596454143524169921875"},
    {0x1.3880000000000p+13, chars_format::general, 309, "10000"},
    {0x1.869ffffffffffp+16, chars_format::general, 309, "99999.999999999985448084771633148193359375"},
    {0x1.86a0000000000p+16, chars_format::general, 309, "100000"},
    {0x1.e847fffffffffp+19, chars_format::general, 309, "999999.999999999883584678173065185546875"},
    {0x1.e848000000000p+19, chars_format::general, 309, "1000000"},
    {0x1.312cfffffffffp+23, chars_format::general, 309, "9999999.99999999813735485076904296875"},
    {0x1.312d000000000p+23, chars_format::general, 309, "10000000"},
    {0x1.7d783ffffffffp+26, chars_format::general, 309, "99999999.99999998509883880615234375"},
    {0x1.7d78400000000p+26, chars_format::general, 309, "100000000"},
    {0x1.dcd64ffffffffp+29, chars_format::general, 309, "999999999.99999988079071044921875"},
    {0x1.dcd6500000000p+29, chars_format::general, 309, "1000000000"},
    {0x1.2a05f1fffffffp+33, chars_format::general, 309, "9999999999.9999980926513671875"},
    {0x1.2a05f20000000p+33, chars_format::general, 309, "10000000000"},
    {0x1.74876e7ffffffp+36, chars_format::general, 309, "99999999999.9999847412109375"},
    {0x1.74876e8000000p+36, chars_format::general, 309, "100000000000"},
    {0x1.d1a94a1ffffffp+39, chars_format::general, 309, "999999999999.9998779296875"},
    {0x1.d1a94a2000000p+39, chars_format::general, 309, "1000000000000"},
    {0x1.2309ce53fffffp+43, chars_format::general, 309, "9999999999999.998046875"},
    {0x1.2309ce5400000p+43, chars_format::general, 309, "10000000000000"},
    {0x1.6bcc41e8fffffp+46, chars_format::general, 309, "99999999999999.984375"},
    {0x1.6bcc41e900000p+46, chars_format::general, 309, "100000000000000"},
    {0x1.c6bf52633ffffp+49, chars_format::general, 309, "999999999999999.875"},
    {0x1.c6bf526340000p+49, chars_format::general, 309, "1000000000000000"},
    {0x1.1c37937e07fffp+53, chars_format::general, 309, "9999999999999998"},
    {0x1.1c37937e08000p+53, chars_format::general, 309, "10000000000000000"},
    {0x1.6345785d89fffp+56, chars_format::general, 309, "99999999999999984"},
    {0x1.6345785d8a000p+56, chars_format::general, 309, "100000000000000000"},
    {0x1.bc16d674ec7ffp+59, chars_format::general, 309, "999999999999999872"},
    {0x1.bc16d674ec800p+59, chars_format::general, 309, "1000000000000000000"},
    {0x1.158e460913cffp+63, chars_format::general, 309, "9999999999999997952"},
    {0x1.158e460913d00p+63, chars_format::general, 309, "10000000000000000000"},
    {0x1.5af1d78b58c3fp+66, chars_format::general, 309, "99999999999999983616"},
    {0x1.5af1d78b58c40p+66, chars_format::general, 309, "100000000000000000000"},
    {0x1.b1ae4d6e2ef4fp+69, chars_format::general, 309, "999999999999999868928"},
    {0x1.b1ae4d6e2ef50p+69, chars_format::general, 309, "1000000000000000000000"},
    {0x1.0f0cf064dd591p+73, chars_format::general, 309, "9999999999999997902848"},
    {0x1.0f0cf064dd592p+73, chars_format::general, 309, "10000000000000000000000"},
    {0x1.52d02c7e14af6p+76, chars_format::general, 309, "99999999999999991611392"},
    {0x1.52d02c7e14af7p+76, chars_format::general, 309, "100000000000000008388608"},
    {0x1.a784379d99db4p+79, chars_format::general, 309, "999999999999999983222784"},
    {0x1.a784379d99db5p+79, chars_format::general, 309, "1000000000000000117440512"},
    {0x1.08b2a2c280290p+83, chars_format::general, 309, "9999999999999998758486016"},
    {0x1.08b2a2c280291p+83, chars_format::general, 309, "10000000000000000905969664"},
    {0x1.4adf4b7320334p+86, chars_format::general, 309, "99999999999999987584860160"},
    {0x1.4adf4b7320335p+86, chars_format::general, 309, "100000000000000004764729344"},
    {0x1.9d971e4fe8401p+89, chars_format::general, 309, "999999999999999875848601600"},
    {0x1.9d971e4fe8402p+89, chars_format::general, 309, "1000000000000000013287555072"},
    {0x1.027e72f1f1281p+93, chars_format::general, 309, "9999999999999999583119736832"},
    {0x1.027e72f1f1282p+93, chars_format::general, 309, "10000000000000001782142992384"},
    {0x1.431e0fae6d721p+96, chars_format::general, 309, "99999999999999991433150857216"},
    {0x1.431e0fae6d722p+96, chars_format::general, 309, "100000000000000009025336901632"},
    {0x1.93e5939a08ce9p+99, chars_format::general, 309, "999999999999999879147136483328"},
    {0x1.93e5939a08ceap+99, chars_format::general, 309, "1000000000000000019884624838656"},
    {0x1.f8def8808b024p+102, chars_format::general, 309, "9999999999999999635896294965248"},
    {0x1.f8def8808b025p+102, chars_format::general, 309, "10000000000000000761796201807872"},
    {0x1.3b8b5b5056e16p+106, chars_format::general, 309, "99999999999999987351763694911488"},
    {0x1.3b8b5b5056e17p+106, chars_format::general, 309, "100000000000000005366162204393472"},
    {0x1.8a6e32246c99cp+109, chars_format::general, 309, "999999999999999945575230987042816"},
    {0x1.8a6e32246c99dp+109, chars_format::general, 309, "1000000000000000089690419062898688"},
    {0x1.ed09bead87c03p+112, chars_format::general, 309, "9999999999999999455752309870428160"},
    {0x1.ed09bead87c04p+112, chars_format::general, 309, "10000000000000000608673814477275136"},
    {0x1.3426172c74d82p+116, chars_format::general, 309, "99999999999999996863366107917975552"},
    {0x1.3426172c74d83p+116, chars_format::general, 309, "100000000000000015310110181627527168"},
    {0x1.812f9cf7920e2p+119, chars_format::general, 309, "999999999999999894846684784341549056"},
    {0x1.812f9cf7920e3p+119, chars_format::general, 309, "1000000000000000042420637374017961984"},
    {0x1.e17b84357691bp+122, chars_format::general, 309, "9999999999999999538762658202121142272"},
    {0x1.e17b84357691cp+122, chars_format::general, 309, "10000000000000000719354278919532445696"},
    {0x1.2ced32a16a1b1p+126, chars_format::general, 309, "99999999999999997748809823456034029568"},
    {0x1.2ced32a16a1b2p+126, chars_format::general, 309, "100000000000000016638275754934614884352"},
    {0x1.78287f49c4a1dp+129, chars_format::general, 309, "999999999999999939709166371603178586112"},
    {0x1.78287f49c4a1ep+129, chars_format::general, 309, "1000000000000000090824893823431825424384"},
    {0x1.d6329f1c35ca4p+132, chars_format::general, 309, "9999999999999999094860208812374492184576"},
    {0x1.d6329f1c35ca5p+132, chars_format::general, 309, "10000000000000000303786028427003666890752"},
    {0x1.25dfa371a19e6p+136, chars_format::general, 309, "99999999999999981277195531206711524196352"},
    {0x1.25dfa371a19e7p+136, chars_format::general, 309, "100000000000000000620008645040778319495168"},
    {0x1.6f578c4e0a060p+139, chars_format::general, 309, "999999999999999890143207767403382423158784"},
    {0x1.6f578c4e0a061p+139, chars_format::general, 309, "1000000000000000044885712678075916785549312"},
    {0x1.cb2d6f618c878p+142, chars_format::general, 309, "9999999999999998901432077674033824231587840"},
    {0x1.cb2d6f618c879p+142, chars_format::general, 309, "10000000000000000139372116959414099130712064"},
    {0x1.1efc659cf7d4bp+146, chars_format::general, 309, "99999999999999989014320776740338242315878400"},
    {0x1.1efc659cf7d4cp+146, chars_format::general, 309, "100000000000000008821361405306422640701865984"},
    {0x1.66bb7f0435c9ep+149, chars_format::general, 309, "999999999999999929757289024535551219930759168"},
    {0x1.66bb7f0435c9fp+149, chars_format::general, 309, "1000000000000000088213614053064226407018659840"},
    {0x1.c06a5ec5433c6p+152, chars_format::general, 309, "9999999999999999931398190359470212947659194368"},
    {0x1.c06a5ec5433c7p+152, chars_format::general, 309, "10000000000000001199048790587699614444362399744"},
    {0x1.18427b3b4a05bp+156, chars_format::general, 309, "99999999999999984102174700855949311516153479168"},
    {0x1.18427b3b4a05cp+156, chars_format::general, 309, "100000000000000004384584304507619735463404765184"},
    {0x1.5e531a0a1c872p+159, chars_format::general, 309, "999999999999999881586566215862833963056037363712"},
    {0x1.5e531a0a1c873p+159, chars_format::general, 309, "1000000000000000043845843045076197354634047651840"},
    {0x1.b5e7e08ca3a8fp+162, chars_format::general, 309, "9999999999999999464902769475481793196872414789632"},
    {0x1.b5e7e08ca3a90p+162, chars_format::general, 309, "10000000000000000762976984109188700329496497094656"},
    {0x1.11b0ec57e6499p+166, chars_format::general, 309, "99999999999999986860582406952576489172979654066176"},
    {0x1.11b0ec57e649ap+166, chars_format::general, 309, "100000000000000007629769841091887003294964970946560"},
    {0x1.561d276ddfdc0p+169, chars_format::general, 309, "999999999999999993220948674361627976461708441944064"},
    {0x1.561d276ddfdc1p+169, chars_format::general, 309, "1000000000000000159374448147476112089437590976987136"},
    {0x1.aba4714957d30p+172, chars_format::general, 309, "9999999999999999932209486743616279764617084419440640"},
    {0x1.aba4714957d31p+172, chars_format::general, 309, "10000000000000001261437482528532152668424144699785216"},
    {0x1.0b46c6cdd6e3ep+176, chars_format::general, 309, "99999999999999999322094867436162797646170844194406400"},
    {0x1.0b46c6cdd6e3fp+176, chars_format::general, 309, "100000000000000020589742799994816764107083808679919616"},
    {0x1.4e1878814c9cdp+179, chars_format::general, 309, "999999999999999908150356944127012110618056584002011136"},
    {0x1.4e1878814c9cep+179, chars_format::general, 309, "1000000000000000078291540404596243842305360299886116864"},
    {0x1.a19e96a19fc40p+182, chars_format::general, 309, "9999999999999998741221202520331657642805958408251899904"},
    {0x1.a19e96a19fc41p+182, chars_format::general, 309, "10000000000000000102350670204085511496304388135324745728"},
    {0x1.05031e2503da8p+186, chars_format::general, 309, "99999999999999987412212025203316576428059584082518999040"},
    {0x1.05031e2503da9p+186, chars_format::general, 309, "100000000000000009190283508143378238084034459715684532224"},
    {0x1.4643e5ae44d12p+189, chars_format::general, 309, "999999999999999874122120252033165764280595840825189990400"},
    {0x1.4643e5ae44d13p+189, chars_format::general, 309, "1000000000000000048346692115553659057528394845890514255872"},
    {0x1.97d4df19d6057p+192, chars_format::general, 309, "9999999999999999438119489974413630815797154428513196965888"},
    {0x1.97d4df19d6058p+192, chars_format::general, 309, "10000000000000000831916064882577577161779546469035791089664"},
    {0x1.fdca16e04b86dp+195, chars_format::general, 309, "99999999999999997168788049560464200849936328366177157906432"},
    {0x1.fdca16e04b86ep+195, chars_format::general, 309,
        "100000000000000008319160648825775771617795464690357910896640"},
    {0x1.3e9e4e4c2f344p+199, chars_format::general, 309,
        "999999999999999949387135297074018866963645011013410073083904"},
    {0x1.3e9e4e4c2f345p+199, chars_format::general, 309,
        "1000000000000000127793096885319003999249391192200302120927232"},
    {0x1.8e45e1df3b015p+202, chars_format::general, 309,
        "9999999999999999493871352970740188669636450110134100730839040"},
    {0x1.8e45e1df3b016p+202, chars_format::general, 309,
        "10000000000000000921119045676700069727922419559629237113585664"},
    {0x1.f1d75a5709c1ap+205, chars_format::general, 309,
        "99999999999999992084218144295482124579792562202350734542897152"},
    {0x1.f1d75a5709c1bp+205, chars_format::general, 309,
        "100000000000000003502199685943161173046080317798311825604870144"},
    {0x1.3726987666190p+209, chars_format::general, 309,
        "999999999999999875170255276364105051932774599639662981181079552"},
    {0x1.3726987666191p+209, chars_format::general, 309,
        "1000000000000000057857959942726969827393378689175040438172647424"},
    {0x1.84f03e93ff9f4p+212, chars_format::general, 309,
        "9999999999999998751702552763641050519327745996396629811810795520"},
    {0x1.84f03e93ff9f5p+212, chars_format::general, 309,
        "10000000000000000213204190094543968723012578712679649467743338496"},
    {0x1.e62c4e38ff872p+215, chars_format::general, 309,
        "99999999999999999209038626283633850822756121694230455365568299008"},
    {0x1.e62c4e38ff873p+215, chars_format::general, 309,
        "100000000000000010901051724930857196452234783424494612613028642816"},
    {0x1.2fdbb0e39fb47p+219, chars_format::general, 309,
        "999999999999999945322333868247445125709646570021247924665841614848"},
    {0x1.2fdbb0e39fb48p+219, chars_format::general, 309,
        "1000000000000000132394543446603018655781305157705474440625207115776"},
    {0x1.7bd29d1c87a19p+222, chars_format::general, 309,
        "9999999999999999827367757839185598317239782875580932278577147150336"},
    {0x1.7bd29d1c87a1ap+222, chars_format::general, 309,
        "10000000000000001323945434466030186557813051577054744406252071157760"},
    {0x1.dac74463a989fp+225, chars_format::general, 309,
        "99999999999999995280522225138166806691251291352861698530421623488512"},
    {0x1.dac74463a98a0p+225, chars_format::general, 309,
        "100000000000000007253143638152923512615837440964652195551821015547904"},
    {0x1.28bc8abe49f63p+229, chars_format::general, 309,
        "999999999999999880969493773293127831364996015857874003175819882528768"},
    {0x1.28bc8abe49f64p+229, chars_format::general, 309,
        "1000000000000000072531436381529235126158374409646521955518210155479040"},
    {0x1.72ebad6ddc73cp+232, chars_format::general, 309,
        "9999999999999999192818822949403492903236716946156035936442979371188224"},
    {0x1.72ebad6ddc73dp+232, chars_format::general, 309,
        "10000000000000000725314363815292351261583744096465219555182101554790400"},
    {0x1.cfa698c95390bp+235, chars_format::general, 309,
        "99999999999999991928188229494034929032367169461560359364429793711882240"},
    {0x1.cfa698c95390cp+235, chars_format::general, 309,
        "100000000000000004188152556421145795899143386664033828314342771180699648"},
    {0x1.21c81f7dd43a7p+239, chars_format::general, 309,
        "999999999999999943801810948794571024057224129020550531544123892056457216"},
    {0x1.21c81f7dd43a8p+239, chars_format::general, 309,
        "1000000000000000139961240179628344893925643604260126034742731531557535744"},
    {0x1.6a3a275d49491p+242, chars_format::general, 309,
        "9999999999999999830336967949613257980309080240684656321838454199566729216"},
    {0x1.6a3a275d49492p+242, chars_format::general, 309,
        "10000000000000001399612401796283448939256436042601260347427315315575357440"},
    {0x1.c4c8b1349b9b5p+245, chars_format::general, 309,
        "99999999999999995164818811802792197885196090803013355167206819763650035712"},
    {0x1.c4c8b1349b9b6p+245, chars_format::general, 309,
        "100000000000000007719022282576153725556774937218346187371917708691719061504"},
    {0x1.1afd6ec0e1411p+249, chars_format::general, 309,
        "999999999999999926539781176481198923508803215199467887262646419780362305536"},
    {0x1.1afd6ec0e1412p+249, chars_format::general, 309,
        "1000000000000000127407036708854983366254064757844793202538020642629466718208"},
    {0x1.61bcca7119915p+252, chars_format::general, 309,
        "9999999999999998863663300700064420349597509066704028242075715752105414230016"},
    {0x1.61bcca7119916p+252, chars_format::general, 309,
        "10000000000000000470601344959054695891559601407866630764278709534898249531392"},
    {0x1.ba2bfd0d5ff5bp+255, chars_format::general, 309,
        "99999999999999998278261272554585856747747644714015897553975120217811154108416"},
    {0x1.ba2bfd0d5ff5cp+255, chars_format::general, 309,
        "100000000000000011133765626626508061083444383443316717731599070480153836519424"},
    {0x1.145b7e285bf98p+259, chars_format::general, 309,
        "999999999999999802805551768538947706777722104929947493053015898505313987330048"},
    {0x1.145b7e285bf99p+259, chars_format::general, 309,
        "1000000000000000008493621433689702976148869924598760615894999102702796905906176"},
    {0x1.59725db272f7fp+262, chars_format::general, 309,
        "9999999999999999673560075006595519222746403606649979913266024618633003221909504"},
    {0x1.59725db272f80p+262, chars_format::general, 309,
        "10000000000000001319064632327801561377715586164000484896001890252212866570518528"},
    {0x1.afcef51f0fb5ep+265, chars_format::general, 309,
        "99999999999999986862573406138718939297648940722396769236245052384850852127440896"},
    {0x1.afcef51f0fb5fp+265, chars_format::general, 309,
        "100000000000000000026609864708367276537402401181200809098131977453489758916313088"},
    {0x1.0de1593369d1bp+269, chars_format::general, 309,
        "999999999999999921281879895665782741935503249059183851809998224123064148429897728"},
    {0x1.0de1593369d1cp+269, chars_format::general, 309,
        "1000000000000000131906463232780156137771558616400048489600189025221286657051852800"},
    {0x1.5159af8044462p+272, chars_format::general, 309,
        "9999999999999999634067965630886574211027143225273567793680363843427086501542887424"},
    {0x1.5159af8044463p+272, chars_format::general, 309,
        "10000000000000001319064632327801561377715586164000484896001890252212866570518528000"},
    {0x1.a5b01b605557ap+275, chars_format::general, 309,
        "99999999999999989600692989521205793443517660497828009527517532799127744739526311936"},
    {0x1.a5b01b605557bp+275, chars_format::general, 309,
        "100000000000000003080666323096525690777025204007643346346089744069413985291331436544"},
    {0x1.078e111c3556cp+279, chars_format::general, 309,
        "999999999999999842087036560910778345101146430939018748000886482910132485188042620928"},
    {0x1.078e111c3556dp+279, chars_format::general, 309,
        "1000000000000000057766609898115896702437267127096064137098041863234712334016924614656"},
    {0x1.4971956342ac7p+282, chars_format::general, 309,
        "9999999999999998420870365609107783451011464309390187480008864829101324851880426209280"},
    {0x1.4971956342ac8p+282, chars_format::general, 309,
        "10000000000000000146306952306748730309700429878646550592786107871697963642511482159104"},
    {0x1.9bcdfabc13579p+285, chars_format::general, 309,
        "99999999999999987659576829486359728227492574232414601025643134376206526100066373992448"},
    {0x1.9bcdfabc1357ap+285, chars_format::general, 309,
        "100000000000000001463069523067487303097004298786465505927861078716979636425114821591040"},
    {0x1.0160bcb58c16cp+289, chars_format::general, 309,
        "999999999999999959416724456350362731491996089648451439669739009806703922950954425516032"},
    {0x1.0160bcb58c16dp+289, chars_format::general, 309,
        "1000000000000000180272607553648403929404183682513265918105226119259073688151729587093504"},
    {0x1.41b8ebe2ef1c7p+292, chars_format::general, 309,
        "9999999999999999594167244563503627314919960896484514396697390098067039229509544255160320"},
    {0x1.41b8ebe2ef1c8p+292, chars_format::general, 309,
        "10000000000000001361014309341887956898217461639403030224181286973685997351115745547780096"},
    {0x1.922726dbaae39p+295, chars_format::general, 309,
        "99999999999999999475366575191804932315794610450682175621941694731908308538307845136842752"},
    {0x1.922726dbaae3ap+295, chars_format::general, 309,
        "100000000000000013610143093418879568982174616394030302241812869736859973511157455477800960"},
    {0x1.f6b0f092959c7p+298, chars_format::general, 309,
        "999999999999999966484112715463900049825186092620125502979674597309179755437379230686511104"},
    {0x1.f6b0f092959c8p+298, chars_format::general, 309,
        "1000000000000000079562324861280497143156226140166910515938643997348793075220176113414176768"},
    {0x1.3a2e965b9d81cp+302, chars_format::general, 309,
        "9999999999999998986371854279739417938265620640920544952042929572854117635677011010499117056"},
    {0x1.3a2e965b9d81dp+302, chars_format::general, 309,
        "10000000000000000795623248612804971431562261401669105159386439973487930752201761134141767680"},
    {0x1.88ba3bf284e23p+305, chars_format::general, 309,
        "99999999999999989863718542797394179382656206409205449520429295728541176356770110104991170560"},
    {0x1.88ba3bf284e24p+305, chars_format::general, 309,
        "100000000000000004337729697461918607329029332495193931179177378933611681288968111094132375552"},
    {0x1.eae8caef261acp+308, chars_format::general, 309,
        "999999999999999927585207737302990649719308316264031458521789123695552773432097103028194115584"},
    {0x1.eae8caef261adp+308, chars_format::general, 309,
        "1000000000000000043377296974619186073290293324951939311791773789336116812889681110941323755520"},
    {0x1.32d17ed577d0bp+312, chars_format::general, 309,
        "9999999999999998349515363474500343108625203093137051759058013911831015418660298966976904036352"},
    {0x1.32d17ed577d0cp+312, chars_format::general, 309,
        "10000000000000000202188791271559469885760963232143577411377768562080040049981643093586978275328"},
    {0x1.7f85de8ad5c4ep+315, chars_format::general, 309,
        "99999999999999987200500490339121684640523551209383568895219648418808203449245677922989188841472"},
    {0x1.7f85de8ad5c4fp+315, chars_format::general, 309,
        "100000000000000002021887912715594698857609632321435774113777685620800400499816430935869782753280"},
    {0x1.df67562d8b362p+318, chars_format::general, 309,
        "999999999999999931290554592897108903273579836542044509826428632996050822694739791281414264061952"},
    {0x1.df67562d8b363p+318, chars_format::general, 309,
        "1000000000000000049861653971908893017010268485438462151574892930611988399099305815384459015356416"},
    {0x1.2ba095dc7701dp+322, chars_format::general, 309,
        "9999999999999998838621148412923952577789043769834774531270429139496757921329133816401963635441664"},
    {0x1.2ba095dc7701ep+322, chars_format::general, 309,
        "10000000000000000735758738477112498397576062152177456799245857901351759143802190202050679656153088"},
    {0x1.7688bb5394c25p+325, chars_format::general, 309,
        "99999999999999999769037024514370800696612547992403838920556863966097586548129676477911932478685184"},
    {0x1.7688bb5394c26p+325, chars_format::general, 309,
        "100000000000000014946137745027879167254908695051145297064360294060937596327914127563101660644376576"},
    {0x1.d42aea2879f2ep+328, chars_format::general, 309,
        "999999999999999967336168804116691273849533185806555472917961779471295845921727862608739868455469056"},
    {0x1.d42aea2879f2fp+328, chars_format::general, 309,
        "1000000000000000088752974568224758206315902362276487138068389220230015924160003471290257693781000192"},
    {0x1.249ad2594c37cp+332, chars_format::general, 309,
        "9999999999999998216360018871870109548898901740426374747374488505608317520357971321909184780648316928"},
    {0x1.249ad2594c37dp+332, chars_format::general, 309,
        "10000000000000000159028911097599180468360808563945281389781327557747838772170381060813469985856815104"},
    {0x1.6dc186ef9f45cp+335, chars_format::general, 309,
        "99999999999999997704951326524533662844684271992415000612999597473199345218078991130326129448151154688"},
    {0x1.6dc186ef9f45dp+335, chars_format::general, 309,
        "100000000000000013246302464330366230200379526580566253752254309890315515232578269041560411089819140096"},
    {0x1.c931e8ab87173p+338, chars_format::general, 309,
        "999999999999999977049513265245336628446842719924150006129995974731993452180789911303261294481511546880"},
    {0x1.c931e8ab87174p+338, chars_format::general, 309,
        "1000000000000000101380322367691997167292404756629360031244033674068922812296784134593135547614855430144"},
    {0x1.1dbf316b346e7p+342, chars_format::general, 309,
        "9999999999999998029863805218200118740630558685368559709703431956602923480183979986974373400948301103104"},
    {0x1.1dbf316b346e8p+342, chars_format::general, 309,
        "10000000000000000019156750857346687362159551272651920111528035145993793242039887559612361451081803235328"},
    {0x1.652efdc6018a1p+345, chars_format::general, 309,
        "99999999999999984277223943460294324649363572028252317900683525944810974325551615015019710109750015295488"},
    {0x1.652efdc6018a2p+345, chars_format::general, 309,
        "100000000000000000191567508573466873621595512726519201115280351459937932420398875596123614510818032353280"},
    {0x1.be7abd3781ecap+348, chars_format::general, 309,
        "999999999999999938258300825281978540327027364472124478294416212538871491824599713636820527503908255301632"},
    {0x1.be7abd3781ecbp+348, chars_format::general, 309,
        "1000000000000000065573049346187358932104882890058259544011190816659887156583377798285651762712452391763968"},
    {0x1.170cb642b133ep+352, chars_format::general, 309,
        "9999999999999998873324014169198263836158851542376704520077063708904652259210884797772880334204906007166976"},
    {0x1.170cb642b133fp+352, chars_format::general, 309,
        "10000000000000000910359990503684350104604539951754865571545457374840902895351334152154180097541612190564352"},
    {0x1.5ccfe3d35d80ep+355, chars_format::general, 309,
        "99999999999999996881384047029926983435371269061279689406644211752791525136670645395254002395395884805259264"},
    {0x1.5ccfe3d35d80fp+355, chars_format::general, 309,
        "100000000000000013177671857705815673582936776336304977818391361080281530225794240230304400502089534272438272"},
    {0x1.b403dcc834e11p+358, chars_format::general, 309,
        "999999999999999903628689227595715073763450661512695740419453520217955231010212074612338431527184250183876608"},
    {0x1.b403dcc834e12p+358, chars_format::general, 309,
        "100000000000000003399899171300282459494397471971289804771343071483787527172320083329274161638073344592130867"
        "2"},
    {0x1.108269fd210cbp+362, chars_format::general, 309,
        "999999999999999981850870718839980786471765096432817124795839836989907255438005329820580342439313767626335846"
        "4"},
    {0x1.108269fd210ccp+362, chars_format::general, 309,
        "1000000000000000190443354695491356020360603589553140816466203348381779320578787343709225438204992480806227148"
        "8"},
    {0x1.54a3047c694fdp+365, chars_format::general, 309,
        "9999999999999998566953803328491556461384620005606229097936217301547840163535361214873932849799065397184010649"
        "6"},
    {0x1.54a3047c694fep+365, chars_format::general, 309,
        "10000000000000000235693675141702558332495327950568818631299125392682816684661617325983093615924495102623141068"
        "8"},
    {0x1.a9cbc59b83a3dp+368, chars_format::general, 309,
        "99999999999999995681977264164181575840510447725837828179539621562288260762111148815394293094743232204474889011"
        "2"},
    {0x1.a9cbc59b83a3ep+368, chars_format::general, 309,
        "10000000000000000903189623866986959080939611128553854444644288629136807293112119770426757922374666984798793236"
        "48"},
    {0x1.0a1f5b8132466p+372, chars_format::general, 309,
        "99999999999999993011993469263043972846733315013897684926158968616472298328309139037619635868942544675772280340"
        "48"},
    {0x1.0a1f5b8132467p+372, chars_format::general, 309,
        "10000000000000001437186382847214479679695037670941883095320419218299999779872521725981689367534804490539314970"
        "624"},
    {0x1.4ca732617ed7fp+375, chars_format::general, 309,
        "99999999999999984468045325579403643266646490335689226515340879189861218540142707748740732746380344583923932594"
        "176"},
    {0x1.4ca732617ed80p+375, chars_format::general, 309,
        "10000000000000000155594161294668430242682013969210614333697705804308337811647557032649853899150474476762062808"
        "6784"},
    {0x1.9fd0fef9de8dfp+378, chars_format::general, 309,
        "99999999999999987885624583052859775098681220206972609879668114960505650455409280264292293995405224620663271692"
        "6976"},
    {0x1.9fd0fef9de8e0p+378, chars_format::general, 309,
        "10000000000000000155594161294668430242682013969210614333697705804308337811647557032649853899150474476762062808"
        "67840"},
    {0x1.03e29f5c2b18bp+382, chars_format::general, 309,
        "99999999999999979683434365116565058701797868515892489805282749110959013858769506226968546997745512532488857856"
        "24576"},
    {0x1.03e29f5c2b18cp+382, chars_format::general, 309,
        "10000000000000000155594161294668430242682013969210614333697705804308337811647557032649853899150474476762062808"
        "678400"},
    {0x1.44db473335deep+385, chars_format::general, 309,
        "99999999999999984057935814682588907446802322751135220511621610897383886710310719046874545396497358979515211902"
        "353408"},
    {0x1.44db473335defp+385, chars_format::general, 309,
        "10000000000000000155594161294668430242682013969210614333697705804308337811647557032649853899150474476762062808"
        "6784000"},
    {0x1.961219000356ap+388, chars_format::general, 309,
        "99999999999999991057138133988227065438809449527523589641763789755663683272776659558724142834500313294757378376"
        "1256448"},
    {0x1.961219000356bp+388, chars_format::general, 309,
        "10000000000000000505554277259950338142282370308030032790204814747222327639770854058242333771050622192524171132"
        "36701184"},
    {0x1.fb969f40042c5p+391, chars_format::general, 309,
        "99999999999999996656499989432737591832415150948634284945877532842287520522749411968203820784902676746951111555"
        "14343424"},
    {0x1.fb969f40042c6p+391, chars_format::general, 309,
        "10000000000000000785522370032175864461962655379085567555410501901553519502269491678716317668570740365133857791"
        "317901312"},
    {0x1.3d3e2388029bbp+395, chars_format::general, 309,
        "99999999999999994416755247254933381274972870380190006824232035607637985622760311004411949604741731366073618283"
        "536318464"},
    {0x1.3d3e2388029bcp+395, chars_format::general, 309,
        "10000000000000001233471318467736706573451111492774423179739601348483426482267311871474691904602929441309356445"
        "6393244672"},
    {0x1.8c8dac6a0342ap+398, chars_format::general, 309,
        "99999999999999998000346834739420118166880519289700851818864831183077241462742872546478943492999243975477607518"
        "1077037056"},
    {0x1.8c8dac6a0342bp+398, chars_format::general, 309,
        "10000000000000001233471318467736706573451111492774423179739601348483426482267311871474691904602929441309356445"
        "63932446720"},
    {0x1.efb1178484134p+401, chars_format::general, 309,
        "99999999999999992266600294764241339139828281034483499827452358262374432118770774079171753271787223800431224742"
        "79348731904"},
    {0x1.efb1178484135p+401, chars_format::general, 309,
        "10000000000000000373409337471459889719393275754491820381027730410378005080671497101378613371421126415052399029"
        "342192009216"},
    {0x1.35ceaeb2d28c0p+405, chars_format::general, 309,
        "99999999999999983092605830803955292696544699826135736641192401589249937168415416531480248917847991520357012302"
        "290741100544"},
    {0x1.35ceaeb2d28c1p+405, chars_format::general, 309,
        "10000000000000000144059475872452738558311186224283126301371231493549892706912613162686325762572645608050543718"
        "3296233537536"},
    {0x1.83425a5f872f1p+408, chars_format::general, 309,
        "99999999999999997770996973140412967005798429759492157739208332266249129088983988607786655884150763168475752207"
        "0951350501376"},
    {0x1.83425a5f872f2p+408, chars_format::general, 309,
        "10000000000000001244938811547687064131505215969284857883722426294324832100955256068409306285045353481659449211"
        "18995289997312"},
    {0x1.e412f0f768fadp+411, chars_format::general, 309,
        "99999999999999994835318744673121432143947683772820873519605146130849290704870274192525374490890208838852004226"
        "13425626021888"},
    {0x1.e412f0f768faep+411, chars_format::general, 309,
        "10000000000000000657803165854228757159135066771950601039801789067244864424132513185357050006393242615734699614"
        "997777141989376"},
    {0x1.2e8bd69aa19ccp+415, chars_format::general, 309,
        "99999999999999992486776161899288204254467086983483846143922597222529419997579302660316349376281765375153005841"
        "365553228283904"},
    {0x1.2e8bd69aa19cdp+415, chars_format::general, 309,
        "10000000000000001127511682408995402737031186129818006514938298848908838565590707491798855029314931308474499291"
        "9515177483763712"},
    {0x1.7a2ecc414a03fp+418, chars_format::general, 309,
        "99999999999999992486776161899288204254467086983483846143922597222529419997579302660316349376281765375153005841"
        "3655532282839040"},
    {0x1.7a2ecc414a040p+418, chars_format::general, 309,
        "10000000000000000751744869165182086274714290643524082134829091023577659252424152046645411010977580354282659550"
        "38852526326677504"},
    {0x1.d8ba7f519c84fp+421, chars_format::general, 309,
        "99999999999999995492910667849794735953002250873835241184796259825178854502911746221543901522980573008687723773"
        "86949310916067328"},
    {0x1.d8ba7f519c850p+421, chars_format::general, 309,
        "10000000000000000751744869165182086274714290643524082134829091023577659252424152046645411010977580354282659550"
        "388525263266775040"},
    {0x1.27748f9301d31p+425, chars_format::general, 309,
        "99999999999999988278187853568579059876517857536991893086699469578820211690113881674597776370903434688204400735"
        "860037395056427008"},
    {0x1.27748f9301d32p+425, chars_format::general, 309,
        "10000000000000000751744869165182086274714290643524082134829091023577659252424152046645411010977580354282659550"
        "3885252632667750400"},
    {0x1.7151b377c247ep+428, chars_format::general, 309,
        "99999999999999999821744356418524141598892886875941250043654333972994040190590464949711576614226856000977717596"
        "6751665376232210432"},
    {0x1.7151b377c247fp+428, chars_format::general, 309,
        "10000000000000001521315302688511758389539292599454039265292748649855914485789257598319664360532475108467547341"
        "10953387277122797568"},
    {0x1.cda62055b2d9dp+431, chars_format::general, 309,
        "99999999999999993665180888231886764680292871228501592999945072962767998323669620536317549817787697967498615270"
        "90709766158759755776"},
    {0x1.cda62055b2d9ep+431, chars_format::general, 309,
        "10000000000000000597830782460516151851749290252338090708736359498322008205751130936310560341066601403445681992"
        "244323541365884452864"},
    {0x1.2087d4358fc82p+435, chars_format::general, 309,
        "99999999999999991202555500957231813912852864969525730182461368558677581576901282770959939099212034754106974340"
        "599870111173348163584"},
    {0x1.2087d4358fc83p+435, chars_format::general, 309,
        "10000000000000001090355859915447142005237291504133263272233100379140091555104798489382082484781734046124010178"
        "3057690514487343316992"},
    {0x1.68a9c942f3ba3p+438, chars_format::general, 309,
        "99999999999999999082956740236127656368660884998248491198409222651766915166559963620104293398654157036960225317"
        "5829982724989462249472"},
    {0x1.68a9c942f3ba4p+438, chars_format::general, 309,
        "10000000000000001484375921879391934128027692505569401323030493083794558234587732531839300199753840160266672727"
        "15492545951501423476736"},
    {0x1.c2d43b93b0a8bp+441, chars_format::general, 309,
        "99999999999999989626475253101452645421691260963781177979271797740059714858969546601131068239323610297536324145"
        "20324447890822855131136"},
    {0x1.c2d43b93b0a8cp+441, chars_format::general, 309,
        "10000000000000000223511723594768599335098409300973759560478836428900264860242343595976203511843100595010152570"
        "837624953702918544949248"},
    {0x1.19c4a53c4e697p+445, chars_format::general, 309,
        "99999999999999992148203649670699315007549827372972461504375111049848301607660324472857261615145089428049364457"
        "837845490532419930947584"},
    {0x1.19c4a53c4e698p+445, chars_format::general, 309,
        "10000000000000001232203082222467267169441835864650272970520161752815699559718654744666680862171692247215368695"
        "8914653583525950968037376"},
    {0x1.6035ce8b6203dp+448, chars_format::general, 309,
        "99999999999999996182969084181493986344923533627678515144540412345510040405565569067619171016459456036870228958"
        "0532071091311261383655424"},
    {0x1.6035ce8b6203ep+448, chars_format::general, 309,
        "10000000000000001232203082222467267169441835864650272970520161752815699559718654744666680862171692247215368695"
        "89146535835259509680373760"},
    {0x1.b843422e3a84cp+451, chars_format::general, 309,
        "99999999999999992955156736572858249275024568623913672232408171308980649367241373391809643495407962749813537357"
        "88091781425216117243117568"},
    {0x1.b843422e3a84dp+451, chars_format::general, 309,
        "10000000000000000586640612700740119755462042863897304388093713545509821352053815609504775357961393589804030375"
        "857007499376802103616864256"},
    {0x1.132a095ce492fp+455, chars_format::general, 309,
        "99999999999999982626157224225223890651347880611866174913584999992086598044603947229219155428043184231232124237"
        "329592070639473281441202176"},
    {0x1.132a095ce4930p+455, chars_format::general, 309,
        "10000000000000000328415624892049260789870125663596116955123134262587470068987879955440013156277274126839495047"
        "8432243557864849063421149184"},
    {0x1.57f48bb41db7bp+458, chars_format::general, 309,
        "99999999999999986757757029164277634100818555816685173841114268518844218573658917694255350654989095638664689485"
        "5501223680845484378371915776"},
    {0x1.57f48bb41db7cp+458, chars_format::general, 309,
        "10000000000000000328415624892049260789870125663596116955123134262587470068987879955440013156277274126839495047"
        "84322435578648490634211491840"},
    {0x1.adf1aea12525ap+461, chars_format::general, 309,
        "99999999999999990063036873115520628860395095980540372983137683340250314996902894066284306836545824764610741684"
        "12654660604060856295398309888"},
    {0x1.adf1aea12525bp+461, chars_format::general, 309,
        "10000000000000000328415624892049260789870125663596116955123134262587470068987879955440013156277274126839495047"
        "843224355786484906342114918400"},
    {0x1.0cb70d24b7378p+465, chars_format::general, 309,
        "99999999999999984774589122793531837245072631718372054355900219626000560719712531871037976946055058163097058166"
        "404267825310912362767116664832"},
    {0x1.0cb70d24b7379p+465, chars_format::general, 309,
        "10000000000000000592838012408148700370636248876704532886485007448299957782847398065202329650801812456915179223"
        "7293382948229697163514582401024"},
    {0x1.4fe4d06de5056p+468, chars_format::general, 309,
        "99999999999999984774589122793531837245072631718372054355900219626000560719712531871037976946055058163097058166"
        "4042678253109123627671166648320"},
    {0x1.4fe4d06de5057p+468, chars_format::general, 309,
        "10000000000000000169762192382389597041410451735731067396306010351159977440672169089582623259562551128794084542"
        "31155599236459402033650892537856"},
    {0x1.a3de04895e46cp+471, chars_format::general, 309,
        "99999999999999991543802243205677490512685385973947502198764173180240246194516195480953279205883239413034573069"
        "08878466464492349900630570041344"},
    {0x1.a3de04895e46dp+471, chars_format::general, 309,
        "10000000000000000508222848402996879704791089448509839788449208028871961714412352270078388372553960191290960287"
        "445781834331294577148468377157632"},
    {0x1.066ac2d5daec3p+475, chars_format::general, 309,
        "99999999999999980713061250546244445284504979165026785650181847493456749434830333705088795590158149413134549224"
        "793557721710505681023603243483136"},
    {0x1.066ac2d5daec4p+475, chars_format::general, 309,
        "10000000000000000237454323586511053574086579278286821874734649886702374295420205725681776282160832941293459691"
        "3384011607579341316989008157343744"},
    {0x1.4805738b51a74p+478, chars_format::general, 309,
        "99999999999999985045357647610017663375777141888595072269614777768170148138704678415434589036448185413094558762"
        "5116484988842728082166842262552576"},
    {0x1.4805738b51a75p+478, chars_format::general, 309,
        "10000000000000000237454323586511053574086579278286821874734649886702374295420205725681776282160832941293459691"
        "33840116075793413169890081573437440"},
    {0x1.9a06d06e26112p+481, chars_format::general, 309,
        "99999999999999998908706118214091961267848062604013589451800154647253023991102581488541128064576300612966589283"
        "20953898584032761523454337112604672"},
    {0x1.9a06d06e26113p+481, chars_format::general, 309,
        "10000000000000001277205458881816625915991898331943210663398553152633589984350048456164766709270441581283861980"
        "390742947279638242225240251599683584"},
    {0x1.00444244d7cabp+485, chars_format::general, 309,
        "99999999999999993363366729972462242111019694317846182578926003895619873650143420259298512453325054533017777074"
        "930382791057905692427399713177731072"},
    {0x1.00444244d7cacp+485, chars_format::general, 309,
        "10000000000000001554472428293898111873833316746251581007042260690215247501398006517626897489833003885281302590"
        "8047007570187593383655974344970993664"},
    {0x1.405552d60dbd6p+488, chars_format::general, 309,
        "99999999999999997799638240565766017436482388946780108077225324496926393922910749242692604942326051396976826841"
        "5537077468838432306731146395363835904"},
    {0x1.405552d60dbd7p+488, chars_format::general, 309,
        "10000000000000001554472428293898111873833316746251581007042260690215247501398006517626897489833003885281302590"
        "80470075701875933836559743449709936640"},
    {0x1.906aa78b912cbp+491, chars_format::general, 309,
        "99999999999999990701603823616479976915742077540485827279946411534835961486483022869262056959924456414642347214"
        "95638781756234316947997075736253956096"},
    {0x1.906aa78b912ccp+491, chars_format::general, 309,
        "10000000000000000489767265751505205795722270035307438887450423745901682635933847561612315292472764637931130646"
        "815102767620534329186625852171022761984"},
    {0x1.f485516e7577ep+494, chars_format::general, 309,
        "99999999999999993540817590396194393124038202103003539598857976719672134461054113418634276152885094407576139065"
        "595315789290943193957228310232077172736"},
    {0x1.f485516e7577fp+494, chars_format::general, 309,
        "10000000000000000489767265751505205795722270035307438887450423745901682635933847561612315292472764637931130646"
        "8151027676205343291866258521710227619840"},
    {0x1.38d352e5096afp+498, chars_format::general, 309,
        "99999999999999998083559617243737459057312001403031879309116481015410011220367858297629826861622115196270206026"
        "6176005440567032331208403948233373515776"},
    {0x1.38d352e5096b0p+498, chars_format::general, 309,
        "10000000000000001625452772463390972279040719860314523815015049819836151825762283781361202969657019835104647387"
        "07067395631197433897752887331883780669440"},
    {0x1.8708279e4bc5ap+501, chars_format::general, 309,
        "99999999999999987180978752809634100817454883082963864004496070705639106998014870588040505160653265303404445320"
        "16411713261887913912817139180431292235776"},
    {0x1.8708279e4bc5bp+501, chars_format::general, 309,
        "10000000000000000171775323872177191180393104084305455107732328445200031262781885420082626742861173182722545959"
        "543542834786931126445173006249634549465088"},
    {0x1.e8ca3185deb71p+504, chars_format::general, 309,
        "99999999999999992995688547174489225212045346187000138833626956204183589249936464033154810067836651912932851030"
        "272641618719051989257594860081125951275008"},
    {0x1.e8ca3185deb72p+504, chars_format::general, 309,
        "10000000000000000462510813590419947400122627239507268849188872720127255375377965092338341988220342513198966245"
        "0489690590919397689516441796634752009109504"},
    {0x1.317e5ef3ab327p+508, chars_format::general, 309,
        "99999999999999999973340300412315374485553901911843668628584018802436967952242376167291975956456715844366937882"
        "4028710020392594094129030220133015859757056"},
    {0x1.317e5ef3ab328p+508, chars_format::general, 309,
        "10000000000000001858041164237985177254824338384475974808180285239777931115839147519165775165944355299485783615"
        "47501493575598125298270581204991032785108992"},
    {0x1.7dddf6b095ff0p+511, chars_format::general, 309,
        "99999999999999988809097495231793535647940212752094020956652718645231562028552916752672510534664613554072398918"
        "99450398872692753716440996292182057045458944"},
    {0x1.7dddf6b095ff1p+511, chars_format::general, 309,
        "10000000000000000369475456880582265409809179829842688451922778552150543659347219597216513109705408327446511753"
        "687232667314337003349573404171046192448274432"},
    {0x1.dd55745cbb7ecp+514, chars_format::general, 309,
        "99999999999999988809097495231793535647940212752094020956652718645231562028552916752672510534664613554072398918"
        "994503988726927537164409962921820570454589440"},
    {0x1.dd55745cbb7edp+514, chars_format::general, 309,
        "10000000000000000071762315409101683040806148118916031180671277214625066168048834012826660698457618933038657381"
        "3296762136260081534229469225952733653677113344"},
    {0x1.2a5568b9f52f4p+518, chars_format::general, 309,
        "99999999999999998335918022319172171456037227501747053636700761446046841750101255453147787694593874175123738834"
        "4363105067534507348164573733465510370326085632"},
    {0x1.2a5568b9f52f5p+518, chars_format::general, 309,
        "10000000000000001738955907649392944307223125700105311899679684704767740119319793285409834201445239541722641866"
        "53199235428064971301205521941960119701886468096"},
    {0x1.74eac2e8727b1p+521, chars_format::general, 309,
        "99999999999999998335918022319172171456037227501747053636700761446046841750101255453147787694593874175123738834"
        "43631050675345073481645737334655103703260856320"},
    {0x1.74eac2e8727b2p+521, chars_format::general, 309,
        "10000000000000001357883086565897798874899245110119190592477762992735128930457859737390823115048069116880588269"
        "914320093559588785105973323002611978355743916032"},
    {0x1.d22573a28f19dp+524, chars_format::general, 309,
        "99999999999999995287335453651211007997446182781858083179085387749785952239205787068995699003416510776387310061"
        "494932420984963311567802202010637287727642443776"},
    {0x1.d22573a28f19ep+524, chars_format::general, 309,
        "10000000000000000748166572832305566183181036166141396500954688253482951028278766060560405376812596437133302515"
        "3260444764058913004562422887354292284947506921472"},
    {0x1.2357684599702p+528, chars_format::general, 309,
        "99999999999999992848469398716842077230573347005946906812993088792777240630489412361674028050474620057398167043"
        "1418299523701733729688780649419062882836695482368"},
    {0x1.2357684599703p+528, chars_format::general, 309,
        "10000000000000001235939783819179352336555603321323631774173148044884693350022041002024739567400974580931131118"
        "99666497012884928817602711614917542838354527125504"},
    {0x1.6c2d4256ffcc2p+531, chars_format::general, 309,
        "99999999999999985044098022926861498776580272523031142441497732130349363482597013298244681001060569756632909384"
        "41190205280284556945232082632196709006295628251136"},
    {0x1.6c2d4256ffcc3p+531, chars_format::general, 309,
        "10000000000000000065284077450682265568456642148886267118448844545520511777838181142510337509988867035816342470"
        "187175785193750117648543530356184548650438281396224"},
    {0x1.c73892ecbfbf3p+534, chars_format::general, 309,
        "99999999999999991287595123558845961539774732109363753938694017460291665200910932548988158640591809997245115511"
        "395844372456707812265566617217918448639526895091712"},
    {0x1.c73892ecbfbf4p+534, chars_format::general, 309,
        "10000000000000000377458932482281488706616365128202897693308658812017626863753877105047511391965429047846952776"
        "5363729011764432297892058199009821165792668120252416"},
    {0x1.1c835bd3f7d78p+538, chars_format::general, 309,
        "99999999999999993784993963811639746645052515943896798537572531592268585888236500249285549696404306093489997962"
        "1894213003182527093908649335762989920701551401238528"},
    {0x1.1c835bd3f7d79p+538, chars_format::general, 309,
        "10000000000000001376418468583399002748727478662016115532860064464808395138684104185166467814290427486344905756"
        "85380367232106118863932514644433433395151811003809792"},
    {0x1.63a432c8f5cd6p+541, chars_format::general, 309,
        "99999999999999993784993963811639746645052515943896798537572531592268585888236500249285549696404306093489997962"
        "18942130031825270939086493357629899207015514012385280"},
    {0x1.63a432c8f5cd7p+541, chars_format::general, 309,
        "10000000000000000976834654142951997131883033248490828397039502203692087828712013353118885245360428110945724564"
        "726831363863214005099277415826993447002617590832955392"},
    {0x1.bc8d3f7b3340bp+544, chars_format::general, 309,
        "99999999999999987391652932764487656775541389327492204364443535414407668928683046936524228593524316087103098888"
        "157864364992697772750101243698844800887746832841572352"},
    {0x1.bc8d3f7b3340cp+544, chars_format::general, 309,
        "10000000000000000017833499485879183651456364256030139271070152777012950284778995356204687079928429609987689703"
        "6220978235643807646031628623453753183252563447406133248"},
    {0x1.15d847ad00087p+548, chars_format::general, 309,
        "99999999999999989948989345183348492723345839974054042033695133885552035712504428261628757034676312089657858517"
        "7704871391229197474064067196498264773607101557544845312"},
    {0x1.15d847ad00088p+548, chars_format::general, 309,
        "10000000000000001040768064453423518030578144514654874338770792165470696998307547886246498456389228011009593555"
        "46714693321646955446568505272576798891444167390577819648"},
    {0x1.5b4e5998400a9p+551, chars_format::general, 309,
        "99999999999999994040727605053525830239832961008552982304497691439383022566618638381796002540519505693745473925"
        "15068357773127490685649548117139715971745147241514401792"},
    {0x1.5b4e5998400aap+551, chars_format::general, 309,
        "10000000000000001040768064453423518030578144514654874338770792165470696998307547886246498456389228011009593555"
        "467146933216469554465685052725767988914441673905778196480"},
    {0x1.b221effe500d3p+554, chars_format::general, 309,
        "99999999999999990767336997157383960226643264180953830087855645396318233083327270285662206135844950810475381599"
        "246526426844590779296424471954140613832058419086616428544"},
    {0x1.b221effe500d4p+554, chars_format::general, 309,
        "10000000000000000386089942874195144027940205149135043895442382956857739101649274267019739175454317034355575090"
        "2863155030391327289536708508823166797373630632400726786048"},
    {0x1.0f5535fef2084p+558, chars_format::general, 309,
        "99999999999999993386049483474297456237195021643033151861169282230770064669960364762569243259584594717091455459"
        "9698521475539380813444812793279458505403728617494385000448"},
    {0x1.0f5535fef2085p+558, chars_format::general, 309,
        "10000000000000001433574937400960542432160908133966772604767837690638471736302512057782554024950174597002004634"
        "57564579132287164977289357383183877442068884030520150720512"},
    {0x1.532a837eae8a5p+561, chars_format::general, 309,
        "99999999999999993386049483474297456237195021643033151861169282230770064669960364762569243259584594717091455459"
        "96985214755393808134448127932794585054037286174943850004480"},
    {0x1.532a837eae8a6p+561, chars_format::general, 309,
        "10000000000000001014580939590254383070472626940034081121037655797126178682441216941477428085151831571943432816"
        "859913676009376081445204484652029936547358529479149975764992"},
    {0x1.a7f5245e5a2cep+564, chars_format::general, 309,
        "99999999999999990034097500988648181343688772091571619991327827082671720239070003832128235741197850516622880918"
        "243995225045973534722968565889475147553730375141026248523776"},
    {0x1.a7f5245e5a2cfp+564, chars_format::general, 309,
        "10000000000000000344190543093124528091771377029741774747069364767506509796263144755389226581474482731849717908"
        "5147422915077831721209019419643357959500300321574675254607872"},
    {0x1.08f936baf85c1p+568, chars_format::general, 309,
        "99999999999999995397220672965687021173298771373910070983074155319629071328494581320833847770616641237372600185"
        "0053663010587168093173889073910282723323583537144858509574144"},
    {0x1.08f936baf85c2p+568, chars_format::general, 309,
        "10000000000000001684971336087384238049173876850326387495005946826745847568619289127565629588829180412037147725"
        "20508506051096899076950702733972407714468702680083242606919680"},
    {0x1.4b378469b6731p+571, chars_format::general, 309,
        "99999999999999991106722135384055949309610771948039310189677092730063190456954919329869358147081608660772824771"
        "59626944024852218964185263418978577250945597085571816901050368"},
    {0x1.4b378469b6732p+571, chars_format::general, 309,
        "10000000000000000826871628571058023676436276965152235336326534308832671394311356729372731664122173896717192642"
        "523265688348930066834399772699475577180106550229078889679814656"},
    {0x1.9e056584240fdp+574, chars_format::general, 309,
        "99999999999999987674323305318751091818660372407342701554959442658410485759723189737097766448253582599493004440"
        "868991951600366493901423615628791772651134064568704023452975104"},
    {0x1.9e056584240fep+574, chars_format::general, 309,
        "10000000000000000140391862557997052178246197057012913609383004294502130454865010810818413324356568684461228576"
        "3778101906192989276863139689872767772084421689716760605683089408"},
    {0x1.02c35f729689ep+578, chars_format::general, 309,
        "99999999999999984928404241266507205825900052774785414647185322601088322001937806062880493089191161750469148176"
        "2871699606818419373090804007799965727644765395390927070069522432"},
    {0x1.02c35f729689fp+578, chars_format::general, 309,
        "10000000000000000689575675368445829376798260983524370990937828305966563206422087545661867996169052854265999829"
        "29417458880300383900478261195703581718577367397759832385751351296"},
    {0x1.4374374f3c2c6p+581, chars_format::general, 309,
        "99999999999999993715345246233687641002733075598968732752062506784519246026851033820375767838190908467345488222"
        "94900033162112051840457868829614121240178061963384891963422539776"},
    {0x1.4374374f3c2c7p+581, chars_format::general, 309,
        "10000000000000001128922725616804851135639912124733536896181687515138109407667748933536631733619040190109816831"
        "627266107349967768059557526332843049167638877982336134488877170688"},
    {0x1.945145230b377p+584, chars_format::general, 309,
        "99999999999999986685792442259943292861266657339622078268160759437774506806920451614379548038991111093844416185"
        "619536034869697653528180058283225500691937355558043949532406874112"},
    {0x1.945145230b378p+584, chars_format::general, 309,
        "10000000000000000074489805020743198914419949385831538723596425413126398524678161602637198763739070584084656026"
        "0278464628372543383280977318309056924111623883709653889736043921408"},
    {0x1.f965966bce055p+587, chars_format::general, 309,
        "99999999999999989497613563849441032117853224643360740061721458376472402494892684496778035958671030043244845000"
        "5513217535702667994787395102883917853758746611883659375731342835712"},
    {0x1.f965966bce056p+587, chars_format::general, 309,
        "10000000000000000074489805020743198914419949385831538723596425413126398524678161602637198763739070584084656026"
        "02784646283725433832809773183090569241116238837096538897360439214080"},
    {0x1.3bdf7e0360c35p+591, chars_format::general, 309,
        "99999999999999987248156666577842840712583970800369810626872899225514085944514898190859245622927094883724501948"
        "60589317860981148271829194868425875762872481668410834714055235600384"},
    {0x1.3bdf7e0360c36p+591, chars_format::general, 309,
        "10000000000000000524381184475062837195473800154429724610566137243318061834753718863820956830887857615988724636"
        "416932177829345401680187244151732297960592357271816907060120777654272"},
    {0x1.8ad75d8438f43p+594, chars_format::general, 309,
        "99999999999999998045549773481514159457876389246726271914145983150114005386328272459269439234497983649422148597"
        "943950338419997003168440244384097290815044070304544781216945608327168"},
    {0x1.8ad75d8438f44p+594, chars_format::general, 309,
        "10000000000000001244207391601974258445159961384186822029717676171624723130874610481714969738325916867035234413"
        "0394693218166911030435304638650548668396803075131793359985469944758272"},
    {0x1.ed8d34e547313p+597, chars_format::general, 309,
        "99999999999999989407635287958577104461642454489641102884327516010434069832877573044541284345241272636864031278"
        "4735046105718485868083216078242264642659886674081956339558310064685056"},
    {0x1.ed8d34e547314p+597, chars_format::general, 309,
        "10000000000000000092485460198915984445662103416575466159075213886334065057081183893084549086425022065360818770"
        "44340989143693798086218131232373875663313958712699944969706504756133888"},
    {0x1.3478410f4c7ecp+601, chars_format::general, 309,
        "99999999999999991711079150764693652460638170424863814625612440581015385980464426221802125649043062240212862563"
        "66562347133135483117101991090685868467907010818055540655879490029748224"},
    {0x1.3478410f4c7edp+601, chars_format::general, 309,
        "10000000000000001013863005321362603645260389790664550855589183714566591516115925163988885607945737906700351284"
        "520257435740740478607260633556791644798372163435943358738250605092929536"},
    {0x1.819651531f9e7p+604, chars_format::general, 309,
        "99999999999999991711079150764693652460638170424863814625612440581015385980464426221802125649043062240212862563"
        "665623471331354831171019910906858684679070108180555406558794900297482240"},
    {0x1.819651531f9e8p+604, chars_format::general, 309,
        "10000000000000000645311987272383955965421075241028916976983595783273580932502028655627150999337451570164538278"
        "8895184180192194795092289050635704895322791329123657951217763820802932736"},
    {0x1.e1fbe5a7e7861p+607, chars_format::general, 309,
        "99999999999999994659487295156522833899352686821948885654457144031359470649375598288696002517909352932499366608"
        "7115356131035228239552737388526279268078143523691759154905886843985723392"},
    {0x1.e1fbe5a7e7862p+607, chars_format::general, 309,
        "10000000000000000645311987272383955965421075241028916976983595783273580932502028655627150999337451570164538278"
        "88951841801921947950922890506357048953227913291236579512177638208029327360"},
    {0x1.2d3d6f88f0b3cp+611, chars_format::general, 309,
        "99999999999999982865854717589206108144494621233608601539078330229983131973730910021120495042444190163353350428"
        "52788704601485085281825842706955095829283737561469387976341354799421194240"},
    {0x1.2d3d6f88f0b3dp+611, chars_format::general, 309,
        "10000000000000000173566684169691286935226752617495305612368443231218527385476241124924130700318845059398697631"
        "682172475335672600663748292592247410791680053842186513692689376624118857728"},
    {0x1.788ccb6b2ce0cp+614, chars_format::general, 309,
        "99999999999999997961704416875371517110712945186684165206763211895744845478556111003617144611039598507860251139"
        "162957211888350975873638026151889477992007905860430885494197722591793250304"},
    {0x1.788ccb6b2ce0dp+614, chars_format::general, 309,
        "10000000000000001305755411616153692607693126913975972887444809356150655898338131198611379417963500685236715184"
        "9798027377761851098929017625234227997691178436106167891224981897189374558208"},
    {0x1.d6affe45f818fp+617, chars_format::general, 309,
        "99999999999999997961704416875371517110712945186684165206763211895744845478556111003617144611039598507860251139"
        "1629572118883509758736380261518894779920079058604308854941977225917932503040"},
    {0x1.d6affe45f8190p+617, chars_format::general, 309,
        "10000000000000001003838417630430384428368760434914461614091111722835421628241627178961446426591592518346577170"
        "76710133445871510743179417054177602937513443300570204900788250622698582966272"},
    {0x1.262dfeebbb0f9p+621, chars_format::general, 309,
        "99999999999999990715696561218012120806928149689207894646274468696179222996240014532018752818113802502496938798"
        "05812353226907091680705581859236698853640605134247712274342131878495422251008"},
    {0x1.262dfeebbb0fap+621, chars_format::general, 309,
        "10000000000000001003838417630430384428368760434914461614091111722835421628241627178961446426591592518346577170"
        "767101334458715107431794170541776029375134433005702049007882506226985829662720"},
    {0x1.6fb97ea6a9d37p+624, chars_format::general, 309,
        "99999999999999986851159038200753776111576258757220550347347138989744224339004763080499610528553377966303172216"
        "135545569805454885304878641227288327493418395599568449276340570087973407686656"},
    {0x1.6fb97ea6a9d38p+624, chars_format::general, 309,
        "10000000000000000230930913026978715489298382248516992754305645781548421896794576888657617968679507611107823854"
        "3825857419659919011313587350687602971665369018571203143144663564875896666980352"},
    {0x1.cba7de5054485p+627, chars_format::general, 309,
        "99999999999999989942789056614560451867857771502810425786489002754892223264792964241714924360201717595258185481"
        "6736079397763477105066203831193512563278085201938953880500051690455580595453952"},
    {0x1.cba7de5054486p+627, chars_format::general, 309,
        "10000000000000000230930913026978715489298382248516992754305645781548421896794576888657617968679507611107823854"
        "38258574196599190113135873506876029716653690185712031431446635648758966669803520"},
    {0x1.1f48eaf234ad3p+631, chars_format::general, 309,
        "99999999999999987469485041883515111262832561306338525435175511742773824124162403312742673294883045892094174869"
        "24315804379963345034522698960570091326029642051843383703107348987949033805840384"},
    {0x1.1f48eaf234ad4p+631, chars_format::general, 309,
        "10000000000000000725591715973187783610303424287811372824568343983972101724920689074452068181743241951740625976"
        "868675721161334753163637413771490365780039321792212624518252692320803210995433472"},
    {0x1.671b25aec1d88p+634, chars_format::general, 309,
        "99999999999999991426771465453187656230872897620693565997277097362163262749171300799098274999392920617156591849"
        "131877877362376266603456419227541462168315779999172318661364176545198692437590016"},
    {0x1.671b25aec1d89p+634, chars_format::general, 309,
        "10000000000000000725591715973187783610303424287811372824568343983972101724920689074452068181743241951740625976"
        "8686757211613347531636374137714903657800393217922126245182526923208032109954334720"},
    {0x1.c0e1ef1a724eap+637, chars_format::general, 309,
        "99999999999999991426771465453187656230872897620693565997277097362163262749171300799098274999392920617156591849"
        "1318778773623762666034564192275414621683157799991723186613641765451986924375900160"},
    {0x1.c0e1ef1a724ebp+637, chars_format::general, 309,
        "10000000000000000409008802087613980012860197382662969579600217134420946634919977275543620045382451973735632618"
        "47757813447631532786297905940174312186739777303375354598782943738754654264509857792"},
    {0x1.188d357087712p+641, chars_format::general, 309,
        "99999999999999986361444843284006798671781267138319114077787067769344781309159912016563104817620280969076698114"
        "87431649040206546179292274931158555956605099986382706217459209761309199883223171072"},
    {0x1.188d357087713p+641, chars_format::general, 309,
        "10000000000000000662275133196073022890814778906781692175574718614061870706920546714670378554471083956139627305"
        "190456203824330868103505742897540916997511012040520808812168041334151877325366493184"},
    {0x1.5eb082cca94d7p+644, chars_format::general, 309,
        "99999999999999994465967438754696170766327875910118237148971115117854351613178134068619377108456504406004528089"
        "686414709538562749489776621177115003729674648080379472553427423904462708600804999168"},
    {0x1.5eb082cca94d8p+644, chars_format::general, 309,
        "10000000000000001067501262969607491495542109345371648329133920981487349222121457817273192169012895127986018803"
        "9310611147811557324883484364908173892056921944513484293311098076487204128137951576064"},
    {0x1.b65ca37fd3a0dp+647, chars_format::general, 309,
        "99999999999999997707776476942971919604146519418837886377444734057258179734785422889441886024790993780775660079"
        "6112539971931616645685181699233267813951241073670004367049615544210109925082343145472"},
    {0x1.b65ca37fd3a0ep+647, chars_format::general, 309,
        "10000000000000001067501262969607491495542109345371648329133920981487349222121457817273192169012895127986018803"
        "93106111478115573248834843649081738920569219445134842933110980764872041281379515760640"},
    {0x1.11f9e62fe4448p+651, chars_format::general, 309,
        "99999999999999995114329246392351320533891604611862166994665838905735117237499591832783878891723402280958754487"
        "67138256706948253250552493092635735926276453993770366538373425000777236538229086224384"},
    {0x1.11f9e62fe4449p+651, chars_format::general, 309,
        "10000000000000001586190709079731611309593092306766792205689700011791961721578624028604793595626413427949399922"
        "319035400805891558900947084290211273632164107937207783595355268531368138238983848067072"},
    {0x1.56785fbbdd55ap+654, chars_format::general, 309,
        "99999999999999995114329246392351320533891604611862166994665838905735117237499591832783878891723402280958754487"
        "671382567069482532505524930926357359262764539937703665383734250007772365382290862243840"},
    {0x1.56785fbbdd55bp+654, chars_format::general, 309,
        "10000000000000001171239152191632315458352305937650677104445076787548271722012891059539512454335598787978695027"
        "6086559719861028977708681660506961660909865771485203001839588998252499578988328956985344"},
    {0x1.ac1677aad4ab0p+657, chars_format::general, 309,
        "99999999999999988475104336182762586914039022706004325374751867317836077244447864327739380631070368041427476172"
        "3053117059528639544242622390941156386039240473187039308013923507098814799398756243472384"},
    {0x1.ac1677aad4ab1p+657, chars_format::general, 309,
        "10000000000000000175355415660194005415374418651772000861457981049363415723055131933782837715237643652049003280"
        "30374534281861011105867876227585990799216050325567033999660761493056632508247061001404416"},
    {0x1.0b8e0acac4eaep+661, chars_format::general, 309,
        "99999999999999988475104336182762586914039022706004325374751867317836077244447864327739380631070368041427476172"
        "30531170595286395442426223909411563860392404731870393080139235070988147993987562434723840"},
    {0x1.0b8e0acac4eafp+661, chars_format::general, 309,
        "10000000000000000972062404885344653449756728480474941855847657639911300522221339234388177506516007760792756678"
        "147673846152604340428430285295728914471221362369950308146488642846313231335560438561636352"},
    {0x1.4e718d7d7625ap+664, chars_format::general, 309,
        "99999999999999996973312221251036165947450327545502362648241750950346848435554075534196338404706251868027512415"
        "973882408182135734368278484639385041047239877871023591066789981811181813306167128854888448"},
    {0x1.4e718d7d7625bp+664, chars_format::general, 309,
        "10000000000000001396972799138758332401427293722449843719522151821536839081776649794711025395197801952122758490"
        "3311023812640679294256310975729923845933871538975662911597585244013782480038750137870188544"},
    {0x1.a20df0dcd3af0p+667, chars_format::general, 309,
        "99999999999999990174745913196417302720721283673903932829449844044338231482669106569030772185797544806747483421"
        "0390258463987183104130654882031695190925872134291678628544718769301415466131339252487684096"},
    {0x1.a20df0dcd3af1p+667, chars_format::general, 309,
        "10000000000000000377187852930565502917417937141710079246703365785635546538843904449936190462361495892930754141"
        "09087389699655531583234914810756005630018925423128793192791080866922220799992003324610084864"},
    {0x1.0548b68a044d6p+671, chars_format::general, 309,
        "99999999999999990174745913196417302720721283673903932829449844044338231482669106569030772185797544806747483421"
        "03902584639871831041306548820316951909258721342916786285447187693014154661313392524876840960"},
    {0x1.0548b68a044d7p+671, chars_format::general, 309,
        "10000000000000001193015809897119766504625422406301890824958394614356580573190100725756058408630540740284357620"
        "483056684410565406706974707679905918934747573964310619313388981254947040003084017678835253248"},
    {0x1.469ae42c8560cp+674, chars_format::general, 309,
        "99999999999999998876910787506329447650934459829549922997503484884029261182361866844442696946000689845185920534"
        "555642245481492613075738123641525387194542623914743194966239051177873087980216425864602058752"},
    {0x1.469ae42c8560dp+674, chars_format::general, 309,
        "10000000000000001628124053612615373751136081214084190333361076656341132058174738739526654646640697992206279476"
        "1588875043647041218401083394518237123398453444885893859189773399673336170714381427096269357056"},
    {0x1.98419d37a6b8fp+677, chars_format::general, 309,
        "99999999999999998876910787506329447650934459829549922997503484884029261182361866844442696946000689845185920534"
        "5556422454814926130757381236415253871945426239147431949662390511778730879802164258646020587520"},
    {0x1.98419d37a6b90p+677, chars_format::general, 309,
        "10000000000000001280037458640218887953927554167858350726638931022753490870187028328510177656232572190668741991"
        "61822284840139314973360143403428947761576712806916637263450665299742435541675484268499358973952"},
    {0x1.fe52048590672p+680, chars_format::general, 309,
        "99999999999999990522832508168813788517929810720129772436171989677925872670656816980047249176205670608285020905"
        "57969050236202928251957239362070375381666542984859087613894256390005080826781722527340175556608"},
    {0x1.fe52048590673p+680, chars_format::general, 309,
        "10000000000000000166160354728550133402860267619935663985128064995273039068626355013257451286926569625748622041"
        "088095949318798038992779336698179926498716835527012730124200454693714718121768282606166882648064"},
    {0x1.3ef342d37a407p+684, chars_format::general, 309,
        "99999999999999986067324092522138770313660664528439025470128525568004065464414123719036343698981660348604541103"
        "459182906031648839556284004276265549348464259679976306097717770685212259087870984958094927200256"},
    {0x1.3ef342d37a408p+684, chars_format::general, 309,
        "10000000000000000388935775510883884313073724929520201333430238200769129428938489676307996560787770138732646031"
        "1941213291353170611409437561654018367221268940354434586262616943544566455807655946219322240663552"},
    {0x1.8eb0138858d09p+687, chars_format::general, 309,
        "99999999999999989631730825039478784877075981481791623042963296855941511229408278327845068080760868556348924945"
        "1555889830959531939269147157518161129230251958148679621306976052570830984318279772103403898929152"},
    {0x1.8eb0138858d0ap+687, chars_format::general, 309,
        "10000000000000000388935775510883884313073724929520201333430238200769129428938489676307996560787770138732646031"
        "19412132913531706114094375616540183672212689403544345862626169435445664558076559462193222406635520"},
    {0x1.f25c186a6f04cp+690, chars_format::general, 309,
        "99999999999999998186306983081094819829272742169837857217766747946991381065394249388986006597030968254935446165"
        "22696356805028364441642842329313746550197144253860793660984920822957311285732475861572950035529728"},
    {0x1.f25c186a6f04dp+690, chars_format::general, 309,
        "10000000000000000959240852713658286643220175642056616945083801606839120751337554413717392461872443451971747445"
        "865546301465605757840244670001489926894056643817026123591538467885955979875798713382291498097180672"},
    {0x1.37798f428562fp+694, chars_format::general, 309,
        "99999999999999989061425747836704382546929530769255207431309733449871519907009213590435672179676195243109823530"
        "484164010765664497227613801915728022751095446033285297165420831725583764136794858449981115862089728"},
    {0x1.37798f4285630p+694, chars_format::general, 309,
        "10000000000000000731118821832548525711161595357042050700422376244411124222377928518753634101438574126676106879"
        "9969763125334902791605243044670546908252847439043930576054277584733562461577854658781477884848504832"},
    {0x1.8557f31326bbbp+697, chars_format::general, 309,
        "99999999999999992711378241934460557459866815329488267345892539248719464370363227909855805946618104447840072584"
        "3812838336795121561031396504666917998514458446354143529431921823271795036250068185162804696593727488"},
    {0x1.8557f31326bbcp+697, chars_format::general, 309,
        "10000000000000000731118821832548525711161595357042050700422376244411124222377928518753634101438574126676106879"
        "99697631253349027916052430446705469082528474390439305760542775847335624615778546587814778848485048320"},
    {0x1.e6adefd7f06aap+700, chars_format::general, 309,
        "99999999999999995631340237212665497390216642977674715277558783887797819941046439365391912960171631811624271827"
        "49897969201059028320356032930746282153172616351711759756540926280845609521557638656931995269719916544"},
    {0x1.e6adefd7f06abp+700, chars_format::general, 309,
        "10000000000000000731118821832548525711161595357042050700422376244411124222377928518753634101438574126676106879"
        "996976312533490279160524304467054690825284743904393057605427758473356246157785465878147788484850483200"},
    {0x1.302cb5e6f642ap+704, chars_format::general, 309,
        "99999999999999990959401044767537593501656918740576398586892792465272451027953301036534141738485988029569553038"
        "510666318680865279842887243162229186843277653306392406169861934038413548670665077684456779836676898816"},
    {0x1.302cb5e6f642bp+704, chars_format::general, 309,
        "10000000000000000964715781454804920905589581568896966534955675815537392668032585435196522662522856315778842819"
        "4463919811999765293285579587743163725597071694149293171752051249118583734850310313223909471278765965312"},
    {0x1.7c37e360b3d35p+707, chars_format::general, 309,
        "99999999999999998434503752679742239723352477519933705291958378741313041288902322362706575693183018080857103100"
        "8919677160084252852199641809946030023447952696435527124027376600704816231425231719002378564135125254144"},
    {0x1.7c37e360b3d36p+707, chars_format::general, 309,
        "10000000000000001338470916850415153216674359507864831870208955129339422181080036501505144360257707818343220322"
        "56545705106635452959741180566593506333478305023178733248684891121346177720862393603318000095671837786112"},
    {0x1.db45dc38e0c82p+710, chars_format::general, 309,
        "99999999999999995444462669514860381234674254008190782609932144230896805184522713832237602111304206060342083075"
        "93944715707740128306913340586165347614418822310868858990958736965765439335377993421392542578277827477504"},
    {0x1.db45dc38e0c83p+710, chars_format::general, 309,
        "10000000000000000740462700217438781518938714805516247333803708227256174960204114795411349643881945414240216317"
        "574952939280149729167245650639345158094661640924814507988218853130896331250875288495917514830571527733248"},
    {0x1.290ba9a38c7d1p+714, chars_format::general, 309,
        "99999999999999990660396936451049407652789096389402106318690169014230827417515340183487244380298106827518051036"
        "015414262787762879627804165648934234223216948652905993920546904997130825691790753915825536773603473752064"},
    {0x1.290ba9a38c7d2p+714, chars_format::general, 309,
        "10000000000000000979665986870629330198032972686455681148365806988089473848554483477848867530432250375881417919"
        "5711545839946316493393121126499811201907102046476036377876708763639225096339747475108225092810302677843968"},
    {0x1.734e940c6f9c5p+717, chars_format::general, 309,
        "99999999999999986833144350000000628787280970294371165285696588840898045203909441264486958195493227441258825404"
        "0761879473560521568747407734787588406864399290882799171293145332687119715621994096773456255662636329336832"},
    {0x1.734e940c6f9c6p+717, chars_format::general, 309,
        "10000000000000000214215469580419574424931347467449492941767090953422917405833303694048810293471274498629572793"
        "18330932090828950478869943421594604148335480073467842242942440201823873880805647866312652703956229962072064"},
    {0x1.d022390f8b837p+720, chars_format::general, 309,
        "99999999999999996018550557482517698064500472922445423764881181256896722516563598670087645039024937968280966920"
        "73033110439215789148209291468717978517470477604338250142827222541691722147321863584969741246387925089779712"},
    {0x1.d022390f8b838p+720, chars_format::general, 309,
        "10000000000000000826575883412587379043412647642654443507046063781156162560010247521088856083040055200431048894"
        "293585531377363220429189576963174104449239123865018594716021581494785755468791093741283312832736674151661568"},
    {0x1.221563a9b7322p+724, chars_format::general, 309,
        "99999999999999988670225591496504042642724870819986016981533507324097780666440272745607095564199569546663253707"
        "407016578763273303796211201720443029584092898479300433989106071698353021544403254911815982945786756526505984"},
    {0x1.221563a9b7323p+724, chars_format::general, 309,
        "10000000000000000826575883412587379043412647642654443507046063781156162560010247521088856083040055200431048894"
        "293585531377363220429189576963174104449239123865018594716021581494785755468791093741283312832736674151661568"
        "0"},
    {0x1.6a9abc9424febp+727, chars_format::general, 309,
        "99999999999999996508438888548251941759285513062609384217104359519083318639905153731719681670679962529722147801"
        "618552072767416863994485028884962235547412234547654639257549968998154834801806327912222841098418750522549862"
        "4"},
    {0x1.6a9abc9424fecp+727, chars_format::general, 309,
        "10000000000000001218486548265174773999240679754785611868824606390905439458683491570394485388364074849583993599"
        "0041623060775703984391032683214000647474050906684363049794437763597758461316612473913036557403682738514637619"
        "2"},
    {0x1.c5416bb92e3e6p+730, chars_format::general, 309,
        "99999999999999999643724207368951101405909769959658731111332700397077533829291106126164716113272119722945705439"
        "3031662703690742880737945597507699179327399689749963213649275279180755601047675571123855843594715481209674137"
        "6"},
    {0x1.c5416bb92e3e7p+730, chars_format::general, 309,
        "10000000000000001218486548265174773999240679754785611868824606390905439458683491570394485388364074849583993599"
        "00416230607757039843910326832140006474740509066843630497944377635977584613166124739130365574036827385146376192"
        "0"},
    {0x1.1b48e353bce6fp+734, chars_format::general, 309,
        "99999999999999984594354677029595135102113336853821866019036664182705300920238534632828550788829765195472628778"
        "41701812188111865249310881159489304248316684372375624724951524510245607865055365695160441670641811964856316723"
        "2"},
    {0x1.1b48e353bce70p+734, chars_format::general, 309,
        "10000000000000000466018071748206975684050858099493768614209804580186827813230862995727677122141957123210339765"
        "95985489865317261666006898091360622097492643440587430127367316221899487205895055238326459735771560242784354959"
        "36"},
    {0x1.621b1c28ac20bp+737, chars_format::general, 309,
        "99999999999999988607519885120090059449792385682045030043648940506537896362652553697718194875347726402798782554"
        "65332429481124015531462501110312687593638634379075360034695852051995460703834403032781272808056570057453763297"
        "28"},
    {0x1.621b1c28ac20cp+737, chars_format::general, 309,
        "10000000000000000466018071748206975684050858099493768614209804580186827813230862995727677122141957123210339765"
        "95985489865317261666006898091360622097492643440587430127367316221899487205895055238326459735771560242784354959"
        "360"},
    {0x1.baa1e332d728ep+740, chars_format::general, 309,
        "99999999999999991818052051592485998927935624744623561263338761565603972716583768949629910144562095368659705575"
        "64236923315533735757183797070971394269896194384435148282491314085395342974857632902877937717988376531531720556"
        "544"},
    {0x1.baa1e332d728fp+740, chars_format::general, 309,
        "10000000000000000466018071748206975684050858099493768614209804580186827813230862995727677122141957123210339765"
        "95985489865317261666006898091360622097492643440587430127367316221899487205895055238326459735771560242784354959"
        "3600"},
    {0x1.14a52dffc6799p+744, chars_format::general, 309,
        "99999999999999996954903517948319502092964807244749211214842475260109694882873713352688654575305085714037182409"
        "22484113450589288118337870608025324951908290393010809478964053338835154608494800695032601573879266890056452171"
        "3664"},
    {0x1.14a52dffc679ap+744, chars_format::general, 309,
        "10000000000000001750230938337165351475308153724525181102085733003813258354803349096492363229827704709554708974"
        "35547287399081149756295416475624104767995667442731345426485501035259440114304347186365125699744282832415537863"
        "06560"},
    {0x1.59ce797fb817fp+747, chars_format::general, 309,
        "99999999999999992845422344863652699560941461244648691253639504304505117149841757830241659030710693437735200942"
        "35886361342544846229414611778382180406298613586150280521785861936083305301585066461308870489166554603236666879"
        "50848"},
    {0x1.59ce797fb8180p+747, chars_format::general, 309,
        "10000000000000000928334703720231990968903484524505077109845138812692342808196957992002964120908826254294312680"
        "98227736977472261378510764709695475858873732081359239635049862754709070252922400339620379482801740375051580804"
        "694016"},
    {0x1.b04217dfa61dfp+750, chars_format::general, 309,
        "99999999999999996133007283331386141586560138044729107222601881068988779336267322248199255466386207258776786115"
        "85164563028980399740553218842096696042786355031638703687528415058284784747112853848287855356936724432692495112"
        "994816"},
    {0x1.b04217dfa61e0p+750, chars_format::general, 309,
        "10000000000000000928334703720231990968903484524505077109845138812692342808196957992002964120908826254294312680"
        "98227736977472261378510764709695475858873732081359239635049862754709070252922400339620379482801740375051580804"
        "6940160"},
    {0x1.0e294eebc7d2bp+754, chars_format::general, 309,
        "99999999999999988242803431008825880725075313724536108897092176834227990088845967645101024020764974088276981699"
        "46896878981535071313820561889181858515215775562466488089746287565001234077846164119538291674288316841998507352"
        "6276096"},
    {0x1.0e294eebc7d2cp+754, chars_format::general, 309,
        "10000000000000000928334703720231990968903484524505077109845138812692342808196957992002964120908826254294312680"
        "98227736977472261378510764709695475858873732081359239635049862754709070252922400339620379482801740375051580804"
        "69401600"},
    {0x1.51b3a2a6b9c76p+757, chars_format::general, 309,
        "99999999999999992450912152247524686517867220028639041337364019092767077687470690100086747458429631779210210721"
        "53972977140172579808077978930736438529920084612691669741896755561419127768121731974871392305034134223701967491"
        "49011968"},
    {0x1.51b3a2a6b9c77p+757, chars_format::general, 309,
        "10000000000000000928334703720231990968903484524505077109845138812692342808196957992002964120908826254294312680"
        "98227736977472261378510764709695475858873732081359239635049862754709070252922400339620379482801740375051580804"
        "694016000"},
    {0x1.a6208b5068394p+760, chars_format::general, 309,
        "99999999999999999183886106229442775786334270115203733241798966706429617845270246028063904958693084084703377156"
        "85294734193992593398889846197223766553446979093051960385337504355687757672562640543404353314227442034427503713"
        "670135808"},
    {0x1.a6208b5068395p+760, chars_format::general, 309,
        "10000000000000001264983401419327895432326837028833311705066886193375469816086935788401821995921998869568971002"
        "74793824830163262058051358073019842260050076805377254167221900194422501748144445768047027533261405765587857615"
        "8030168064"},
    {0x1.07d457124123cp+764, chars_format::general, 309,
        "99999999999999988411127779858373832956786989976700226194703050524569553592790956543300452958271560395914310860"
        "35179922907880571653590858570844041715803947924475495355832306284857949825457186833751615699518149537266645758"
        "1821100032"},
    {0x1.07d457124123dp+764, chars_format::general, 309,
        "10000000000000000995664443260051171861588155025370724028889488288828968209774953551282735695911460777349244345"
        "33540954548010461514418883382360349139109001026162842541484270242651756551966809425305709092893673453158836166"
        "91581616128"},
    {0x1.49c96cd6d16cbp+767, chars_format::general, 309,
        "99999999999999988411127779858373832956786989976700226194703050524569553592790956543300452958271560395914310860"
        "35179922907880571653590858570844041715803947924475495355832306284857949825457186833751615699518149537266645758"
        "18211000320"},
    {0x1.49c96cd6d16ccp+767, chars_format::general, 309,
        "10000000000000000564754110205208414148406263819830583747005651641554565639675781971892197615894599829797681693"
        "47536362096565980644606923877305160145603279779419783940304062319818564238082591276919599588305301753272401848"
        "696295129088"},
    {0x1.9c3bc80c85c7ep+770, chars_format::general, 309,
        "99999999999999991858410444297115894662242119621021348449773743702764774153584329178424757598406447976326812075"
        "23216662519436418612086534611285553663849717898419964165273969667523488336530932020840491736225123136358120303"
        "938278260736"},
    {0x1.9c3bc80c85c7fp+770, chars_format::general, 309,
        "10000000000000000564754110205208414148406263819830583747005651641554565639675781971892197615894599829797681693"
        "47536362096565980644606923877305160145603279779419783940304062319818564238082591276919599588305301753272401848"
        "6962951290880"},
    {0x1.01a55d07d39cfp+774, chars_format::general, 309,
        "99999999999999997374062707399103193390970327051935144057886852787877127050853725394623645022622268104986814019"
        "04075445897925773745679616275991972780722949856731114260380631079788349954248924320182693394956280894904479577"
        "1481474727936"},
    {0x1.01a55d07d39d0p+774, chars_format::general, 309,
        "10000000000000001943667175980705238830588315677559032649033928912832653863993131025941919471948554861962682179"
        "42751057941188319428005194293481764924821587768997571464080727672884779642512089351755150002988091192908991666"
        "99876243210240"},
    {0x1.420eb449c8842p+777, chars_format::general, 309,
        "99999999999999984136497275954333676442022629217742034598415390983607480097407174475746315204504299796202809353"
        "90014365789551321425056220280696566900227193156784354032124643690352682071725742801761409414001502274393217321"
        "44446136385536"},
    {0x1.420eb449c8843p+777, chars_format::general, 309,
        "10000000000000000178658451788069303237395289299666618054437734005596700936866924236758275496199492420791481557"
        "40876247260071725785255408160775710807422153542338003433646596020960023924842331815965645472194120710174156699"
        "571604284243968"},
    {0x1.9292615c3aa53p+780, chars_format::general, 309,
        "99999999999999991196532172724877418814794734729311692976800170612551291805912001632480891107500549560887611841"
        "97513608514017695996055364811520783369824930063422626153861170298051704942404772944919427537177384205332557191"
        "153093955289088"},
    {0x1.9292615c3aa54p+780, chars_format::general, 309,
        "10000000000000000531660196626596490356033894575245100973356972987043891522292165594595004291349304909025721681"
        "81251209396295044513805365387316921630902040387669917039733422351344975068376283323123546378352914806721123693"
        "0570359138156544"},
    {0x1.f736f9b3494e8p+783, chars_format::general, 309,
        "99999999999999994020546131433094915763903576933939556328154082464128816489313932495174721468699049466761532837"
        "20513305603804245824455022623850469957664024826077935002555780941131314090676385002182634786447736977708293139"
        "0365469918625792"},
    {0x1.f736f9b3494e9p+783, chars_format::general, 309,
        "10000000000000000531660196626596490356033894575245100973356972987043891522292165594595004291349304909025721681"
        "81251209396295044513805365387316921630902040387669917039733422351344975068376283323123546378352914806721123693"
        "05703591381565440"},
    {0x1.3a825c100dd11p+787, chars_format::general, 309,
        "99999999999999994020546131433094915763903576933939556328154082464128816489313932495174721468699049466761532837"
        "20513305603804245824455022623850469957664024826077935002555780941131314090676385002182634786447736977708293139"
        "03654699186257920"},
    {0x1.3a825c100dd12p+787, chars_format::general, 309,
        "10000000000000001209423546716568689623820016704355788177681911831422497446308629001641523578036944886435462720"
        "66771136697843816472621283262276046411983423130707191163420128905684081263961470216866716118177799472091300320"
        "549064642593292288"},
    {0x1.8922f31411455p+790, chars_format::general, 309,
        "99999999999999990405808264286576519669044258912015891238421075294109584894559460990926618606364969587242913963"
        "31073693328877462044103460624068471125229983529879139676226679317989414380888721568885729507381685429067351125"
        "745727105048510464"},
    {0x1.8922f31411456p+790, chars_format::general, 309,
        "10000000000000000486475973287265010404848153099971055159735310397418651127357734700791903005570128910531738945"
        "88883214242858459716550970862319646645496614871467432098154308581055701322003937530207335062364589162363111917"
        "8909006652304785408"},
    {0x1.eb6bafd91596bp+793, chars_format::general, 309,
        "99999999999999999081179145438220670296706622164632687453780292502155740721970192601122065475966761298087599260"
        "65728762788701743116947209423545268323071682640756248459416523213529973684379113808798302177140209145805611957"
        "6436948334022754304"},
    {0x1.eb6bafd91596cp+793, chars_format::general, 309,
        "10000000000000001064834032030707953780025643983478841574092591544621728182518450141471599463543581691254717965"
        "71193552206846745121407220782284766458686061478859239350366964840758405275569963679534839907015157410145662640"
        "01743184712072953856"},
    {0x1.33234de7ad7e2p+797, chars_format::general, 309,
        "99999999999999982887153500621818255791736877426414667851776420380469583177470160262090564652710083437844186705"
        "61039299797029751780972211664521913553767177633785645397462147941854262984530381627628166526924298207894191738"
        "10082174047524749312"},
    {0x1.33234de7ad7e3p+797, chars_format::general, 309,
        "10000000000000000139461138041199244379741658569866383311120941709096804894261305436384085130786057242097951533"
        "99497011464465488473637220910340574757582946907032347746826714825234078949864321840610832155574248213693581484"
        "614981956096327942144"},
    {0x1.7fec216198ddbp+800, chars_format::general, 309,
        "99999999999999990290136652537887930994008760735314333955549619064668969483527317902790679314770279031098318159"
        "34611625736079804963132210640075447162592094208400778225784148066048873590175516339020228538451571779510840981"
        "320420868670460264448"},
    {0x1.7fec216198ddcp+800, chars_format::general, 309,
        "10000000000000000509610295637002728139855252735311366616309601643306774209564163318419090863889067021760658106"
        "68175627761417991132745220859118251438024192735763104388242814831443809480146578576180435256150611892274413946"
        "7759619125060885807104"},
    {0x1.dfe729b9ff152p+803, chars_format::general, 309,
        "99999999999999993251329913304315801074917514058874200397058898538348724005950180959070725179594357268399970740"
        "84040556111699826235996210230296860606122060838246831357112948115726717832433570223577053343062481208157500678"
        "6082605199485453729792"},
    {0x1.dfe729b9ff153p+803, chars_format::general, 309,
        "10000000000000000509610295637002728139855252735311366616309601643306774209564163318419090863889067021760658106"
        "68175627761417991132745220859118251438024192735763104388242814831443809480146578576180435256150611892274413946"
        "77596191250608858071040"},
    {0x1.2bf07a143f6d3p+807, chars_format::general, 309,
        "99999999999999988513420696078031208945463508741178414090644051380461116770073600069022651795875832088717326610"
        "44954267510707792199413810885942599096474114230493146346986868036242167044820684008286133655685026122322845162"
        "94771707790360919932928"},
    {0x1.2bf07a143f6d4p+807, chars_format::general, 309,
        "10000000000000000746505756498316957746327953001196155931630344001201154571357992362921494533074993280744790313"
        "20129942191467592834574340826335964513506590066150788638749118835418037019527222886944981240519484646566146722"
        "558989084608335389392896"},
    {0x1.76ec98994f488p+810, chars_format::general, 309,
        "99999999999999992303748069859058882649026712995335043135775929106771202558774864781061110502850652232463441914"
        "76223298391501419428679730361426008304192471516696094355087732099829807674910992980518869405586990190990569575"
        "476151831539558138249216"},
    {0x1.76ec98994f489p+810, chars_format::general, 309,
        "10000000000000000746505756498316957746327953001196155931630344001201154571357992362921494533074993280744790313"
        "20129942191467592834574340826335964513506590066150788638749118835418037019527222886944981240519484646566146722"
        "5589890846083353893928960"},
    {0x1.d4a7bebfa31aap+813, chars_format::general, 309,
        "99999999999999992303748069859058882649026712995335043135775929106771202558774864781061110502850652232463441914"
        "76223298391501419428679730361426008304192471516696094355087732099829807674910992980518869405586990190990569575"
        "4761518315395581382492160"},
    {0x1.d4a7bebfa31abp+813, chars_format::general, 309,
        "10000000000000000443279566595834743850042896660863625608019793783096347708261891185958417836517007669245101088"
        "85628419721004102656233067268297291776889121483254552798101049710331025769119998169166362380527327521072728769"
        "55671430431745947427930112"},
    {0x1.24e8d737c5f0ap+817, chars_format::general, 309,
        "99999999999999987452129031419343460308465811550014557958007125617094292749237245949651883357922882448468414325"
        "24198938864085576575219353432807244518312974190356320904718626098437627668395397496060967645712476183095882327"
        "43975534688554349643169792"},
    {0x1.24e8d737c5f0bp+817, chars_format::general, 309,
        "10000000000000000685860518517820514967070941733129649866908233957580193198738772127528879193763396158444852468"
        "33229637697374894798906086114728229966183096349571541470619505010400634769445777943389257468521053221467463131"
        "958534128550160206370177024"},
    {0x1.6e230d05b76cdp+820, chars_format::general, 309,
        "99999999999999995214719492922888136053363253862527334242437211200577348444497436079906646789807314102860458468"
        "47437914107950925140755956518597266575720169912499958425309195700665115678820350271193610461511698595727381924"
        "297989722331966923339726848"},
    {0x1.6e230d05b76cep+820, chars_format::general, 309,
        "10000000000000001073990041592997748754315813848755288681129738236754345983501781634041617365357617741164454675"
        "49391586459568162227182916269017731069053456135678723346649033490512009169967025582145889609311014342099038111"
        "8014458473224813777155784704"},
    {0x1.c9abd04725480p+823, chars_format::general, 309,
        "99999999999999992109683308321470265755404276937522223728665176967184126166393360027804741417053541441103640811"
        "18142324010404785714541315284281257752757291623642503417072967859774120474650369161140553335192009630674782085"
        "5546959721533975525765152768"},
    {0x1.c9abd04725481p+823, chars_format::general, 309,
        "10000000000000000452982804672714174694724018463754266578375331390075701527880966423621236290806863208813091144"
        "03532468440058934341939988022154529304460880477907232345001787922333810129133029360135278184047076549088518144"
        "05278709728676750356293615616"},
    {0x1.1e0b622c774d0p+827, chars_format::general, 309,
        "99999999999999992109683308321470265755404276937522223728665176967184126166393360027804741417053541441103640811"
        "18142324010404785714541315284281257752757291623642503417072967859774120474650369161140553335192009630674782085"
        "55469597215339755257651527680"},
    {0x1.1e0b622c774d1p+827, chars_format::general, 309,
        "10000000000000001198191488977054463566234172925755493101680619606090074874625944676125693580267768647634727381"
        "78563410063470007804231501918390371421971971267233021546978482604147648978133824826548011894363801900701142105"
        "351177597329624152546106933248"},
    {0x1.658e3ab795204p+830, chars_format::general, 309,
        "99999999999999992109683308321470265755404276937522223728665176967184126166393360027804741417053541441103640811"
        "18142324010404785714541315284281257752757291623642503417072967859774120474650369161140553335192009630674782085"
        "554695972153397552576515276800"},
    {0x1.658e3ab795205p+830, chars_format::general, 309,
        "10000000000000000800746857348072976168095423879354838955917799224215742423028622941456649692555285746929854721"
        "65213574530984101957676027840397922292632722846259267305924245440513601592000067244461220582194881713174409325"
        "9920359973067672730884158521344"},
    {0x1.bef1c9657a685p+833, chars_format::general, 309,
        "99999999999999992109683308321470265755404276937522223728665176967184126166393360027804741417053541441103640811"
        "18142324010404785714541315284281257752757291623642503417072967859774120474650369161140553335192009630674782085"
        "5546959721533975525765152768000"},
    {0x1.bef1c9657a686p+833, chars_format::general, 309,
        "10000000000000000482791152044887786249584424642234315639307542918716276461750765553721414582385299426365956593"
        "54533706104995377280431648578003962989161324109480263913080855709606363683093061178791787532459745563153023102"
        "50472271728848176952226298724352"},
    {0x1.17571ddf6c813p+837, chars_format::general, 309,
        "99999999999999989566037665895988746407316283040558037195783126523188398476170500925922860535693650876592455786"
        "32703376602494988296586281185129583324986101729410476274325850012516217203394320635785088937310920430503692297"
        "65618973200711352404729235767296"},
    {0x1.17571ddf6c814p+837, chars_format::general, 309,
        "10000000000000000991520280529984090119202023421627152945883953007515421999795337374097790758657277539268193598"
        "51621495586577336764022655397834297874715562088326669341630279279057944337344270883862880412035963403187241060"
        "084423965317738575228107571068928"},
    {0x1.5d2ce55747a18p+840, chars_format::general, 309,
        "99999999999999993635870693776759177364257073275700735648394407233581562780527075488933869945869475779810351826"
        "09405692455150664165314335743772262409420005560181719702721238568128862437403998276353831973920663150777435958"
        "293799716241167969694049028276224"},
    {0x1.5d2ce55747a19p+840, chars_format::general, 309,
        "10000000000000000991520280529984090119202023421627152945883953007515421999795337374097790758657277539268193598"
        "51621495586577336764022655397834297874715562088326669341630279279057944337344270883862880412035963403187241060"
        "0844239653177385752281075710689280"},
    {0x1.b4781ead1989ep+843, chars_format::general, 309,
        "99999999999999993635870693776759177364257073275700735648394407233581562780527075488933869945869475779810351826"
        "09405692455150664165314335743772262409420005560181719702721238568128862437403998276353831973920663150777435958"
        "2937997162411679696940490282762240"},
    {0x1.b4781ead1989fp+843, chars_format::general, 309,
        "10000000000000000665933638299522455642646760202815737069675050550683968855446811409056910005843211547010761915"
        "33485310318364882694524411033142883547960849781864969867358648194608932718623496672617380969107183985565341567"
        "23341516657901421957636703742066688"},
    {0x1.10cb132c2ff63p+847, chars_format::general, 309,
        "99999999999999998845256969464145328989141284776683389667736846542884813090103490929587961990894531655929258756"
        "99584656746549929277286245578834891637495402463568911291067335919313048336936385656281823060781133832727827843"
        "90994049606075766012189756664840192"},
    {0x1.10cb132c2ff64p+847, chars_format::general, 309,
        "10000000000000001968280207221368993548867813078061400574510660378009781432840915269220433017099475516040488648"
        "06030051391214698972517388491908540854979699007711767764445172532404979193506593517599378740822301656052939538"
        "637450361533911642183329172013711360"},
    {0x1.54fdd7f73bf3bp+850, chars_format::general, 309,
        "99999999999999986342729907814418565089419177174325020021314992200557012347120093872018141082834397553243882122"
        "83155142447191693008553661974684581490114449895439651479036702276471002178058655944454644452316004196046887318"
        "431202624493742403095061074555174912"},
    {0x1.54fdd7f73bf3cp+850, chars_format::general, 309,
        "10000000000000000301276599001405425028904865397746951288321079799032741333776462328211123562691457635682438430"
        "17172782817966934136686377344688499501995571998627866456174421380026039705656229556022421593026951037828814135"
        "2402853119916429412464176397346144256"},
    {0x1.aa3d4df50af0ap+853, chars_format::general, 309,
        "99999999999999989676737124254345702129345072534953918593694153358511092545248999754036759991650433313959982558"
        "60869679593687222680215684269124664196082703913607454095578204581228881153759383867608558747906705432495138125"
        "2255327235782798049688841391133687808"},
    {0x1.aa3d4df50af0bp+853, chars_format::general, 309,
        "10000000000000000301276599001405425028904865397746951288321079799032741333776462328211123562691457635682438430"
        "17172782817966934136686377344688499501995571998627866456174421380026039705656229556022421593026951037828814135"
        "24028531199164294124641763973461442560"},
    {0x1.0a6650b926d66p+857, chars_format::general, 309,
        "99999999999999984342325577950462282865463639957947680877887495505784564228242750342806969737544776096814221861"
        "36526420159294375205556448598020531866533497484538969909111800893616274792638219190562295874961583454177936834"
        "35460456504301996197076723582025859072"},
    {0x1.0a6650b926d67p+857, chars_format::general, 309,
        "10000000000000000567997176316599595992098937026597263174111412691669067749626774798772613075396740496539726465"
        "03389945789686576510419339128243706118473032320081290665497741564406670023712287789874734736674207136744674199"
        "783831719918405933396323484899269935104"},
    {0x1.4cffe4e7708c0p+860, chars_format::general, 309,
        "99999999999999992877384052036675753687673932081157661223178148070147009535452749400774634144113827644247438976"
        "95475635254322931165011225671787143593812227771048544607458046793796444970432082673836316471673778619485458899"
        "748089618699435710767754281089234894848"},
    {0x1.4cffe4e7708c1p+860, chars_format::general, 309,
        "10000000000000000994750100020910269533209451632757762191375945319887190014987274751670996295725193073911387320"
        "81337406544438004308392077981932036704836968834406769400415053859415678532601980964038435766509816895010050303"
        "0535059726012267208361728371627187503104"},
    {0x1.a03fde214caf0p+863, chars_format::general, 309,
        "99999999999999992877384052036675753687673932081157661223178148070147009535452749400774634144113827644247438976"
        "95475635254322931165011225671787143593812227771048544607458046793796444970432082673836316471673778619485458899"
        "7480896186994357107677542810892348948480"},
    {0x1.a03fde214caf1p+863, chars_format::general, 309,
        "10000000000000000653347761057461730700321039947829362977564319217312692202698874789352289719462431012014058636"
        "18979437940636862070013886898981372235745819622946386412481204023408471725490226424707474942641329088397749420"
        "43776657045497009088429335535195969814528"},
    {0x1.0427ead4cfed6p+867, chars_format::general, 309,
        "99999999999999992877384052036675753687673932081157661223178148070147009535452749400774634144113827644247438976"
        "95475635254322931165011225671787143593812227771048544607458046793796444970432082673836316471673778619485458899"
        "74808961869943571076775428108923489484800"},
    {0x1.0427ead4cfed7p+867, chars_format::general, 309,
        "10000000000000001472713374569738223899253227991657521090712221863491486952191034698917185502493059960567647479"
        "28638562589759603442121545498062966961564577730451305583522443629825768062558437319101780919925699824267271538"
        "715541135605986002768804111697781423341568"},
    {0x1.4531e58a03e8bp+870, chars_format::general, 309,
        "99999999999999984137484174572393159565730592946990641349600519844239865540869710365415745791787118859675824650"
        "59111638997013689862529533948250133185078807957662740116351490992011950708371166466963719380640490770210556304"
        "785160923755265983999639546733803159420928"},
    {0x1.4531e58a03e8cp+870, chars_format::general, 309,
        "10000000000000000161728392950095834780961727121532468109675577629605415353003578843613352249644053642881905330"
        "33183963151163217246749291739532415400254564758443434909856460259558093923249299888070891356270706646876036149"
        "4711018313643605437535869015444666630275072"},
    {0x1.967e5eec84e2ep+873, chars_format::general, 309,
        "99999999999999987633444125558106197214507928600657449299031571134602723138702925979559301132717802373504470381"
        "13657237499937386383522210637664937348572175883017061912794113312725748413195532949712758217053805909920517342"
        "7703324017329338747068854404759758535917568"},
    {0x1.967e5eec84e2fp+873, chars_format::general, 309,
        "10000000000000000161728392950095834780961727121532468109675577629605415353003578843613352249644053642881905330"
        "33183963151163217246749291739532415400254564758443434909856460259558093923249299888070891356270706646876036149"
        "47110183136436054375358690154446666302750720"},
    {0x1.fc1df6a7a61bap+876, chars_format::general, 309,
        "99999999999999993226980047135247057452551665646524342018121253199183295295236070962188989678206895995630303550"
        "00930195104615300817110493340728624010161564563583976787102309025867824740914519322111220355315110133456455003"
        "54660676649720249983847887046345216426508288"},
    {0x1.fc1df6a7a61bbp+876, chars_format::general, 309,
        "10000000000000000441405189028952877792863913973825812745630061732834443960830236092744836676918508323988196988"
        "77547611031397112968428705874685599733334034192471780653571870045215197739635249206690814463183771858052833032"
        "509915549602573975010166573043840478561173504"},
    {0x1.3d92ba28c7d14p+880, chars_format::general, 309,
        "99999999999999988752151309873534369262116676009830827842849507547518837570009554976085238841815621097929637014"
        "91111829020872969270239867178277674680890053619130444887655752455354163678739330224192450644706066754627704874"
        "925587274685787599733204126473471115726422016"},
    {0x1.3d92ba28c7d15p+880, chars_format::general, 309,
        "10000000000000000665146625892038512202385663455660488454393649015417666847091561892050024218738072068873230315"
        "53038529335584229545772237182808147199797609739694457248544197873740880792744008661586752948714224026994270538"
        "9409665241931447200154303102433395309881065472"},
    {0x1.8cf768b2f9c59p+883, chars_format::general, 309,
        "99999999999999988752151309873534369262116676009830827842849507547518837570009554976085238841815621097929637014"
        "91111829020872969270239867178277674680890053619130444887655752455354163678739330224192450644706066754627704874"
        "9255872746857875997332041264734711157264220160"},
    {0x1.8cf768b2f9c5ap+883, chars_format::general, 309,
        "10000000000000000307160326911101497147150864284725007320371909363284510229073440613161724151826770077057176992"
        "72253060048884843022022587089812071253455888864138174696588473348099787907769993533753251371865500556687970528"
        "65128496484823152800700833072414104710501367808"},
    {0x1.f03542dfb8370p+886, chars_format::general, 309,
        "99999999999999997343822485416022730587751856112282375059371259198714596402444465669404440447686868901514916762"
        "29963091901658245840231469410183497393091354632481226134593141070740392918115693292196488489075430041978905121"
        "87794469896370420793533163493423472892065087488"},
    {0x1.f03542dfb8371p+886, chars_format::general, 309,
        "10000000000000000879938405280600721235526542958221777134806692806697560817902434659383004258884853263962862309"
        "21509810907603861460022027238605792767602642265028226779717632589125536523728417738286853894823458109178050545"
        "114775459800092635220483497954858621317962268672"},
    {0x1.362149cbd3226p+890, chars_format::general, 309,
        "99999999999999997343822485416022730587751856112282375059371259198714596402444465669404440447686868901514916762"
        "29963091901658245840231469410183497393091354632481226134593141070740392918115693292196488489075430041978905121"
        "877944698963704207935331634934234728920650874880"},
    {0x1.362149cbd3227p+890, chars_format::general, 309,
        "10000000000000001567272099323999790141577357366417900912128432938793221524497227514848540387354553088249684689"
        "00617911938066683585621355417158258584578746346096289279472623678356434862878526783727176922373007172166146564"
        "8709640537423259638766536986317197103735005773824"},
    {0x1.83a99c3ec7eafp+893, chars_format::general, 309,
        "99999999999999990012263082286432662256543169091523721434606031123027548865433341877772055077343404109122144711"
        "19476680910054809833838635505623862012012911101088559470539902785610810633847863474166376195213573370105880911"
        "1452663635798820356028494943810497789949089153024"},
    {0x1.83a99c3ec7eb0p+893, chars_format::general, 309,
        "10000000000000000467538188854561279891896054313304102868413648727440164393945558946103682581803033369390768881"
        "34044950289326168184662430331474313277416979816387389279864637935586997520238352311022660078293728671385192933"
        "26106230343475263802678137754874196788463928344576"},
    {0x1.e494034e79e5bp+896, chars_format::general, 309,
        "99999999999999992944886843538268689589026643899827182884512122353302367880237791394425009225480790026079253531"
        "63671245306696184236395769067447716164444288513645626136161198099662643547554995401378421112758316038855090595"
        "43833769773341090453584235060232375896520569913344"},
    {0x1.e494034e79e5cp+896, chars_format::general, 309,
        "10000000000000000467538188854561279891896054313304102868413648727440164393945558946103682581803033369390768881"
        "34044950289326168184662430331474313277416979816387389279864637935586997520238352311022660078293728671385192933"
        "261062303434752638026781377548741967884639283445760"},
    {0x1.2edc82110c2f9p+900, chars_format::general, 309,
        "99999999999999995290985852539737511455013423746469952044436995337522223092081351007747372543990698759644940587"
        "99026896824009283758441475916906799486389390443691279468658234350904109878520700943148057046794110173854458342"
        "872794765056233999682236635579342942941443126198272"},
    {0x1.2edc82110c2fap+900, chars_format::general, 309,
        "10000000000000001405977792455148808638290766251961210532383597921128106478682982791432627909206996862817043703"
        "88187210896251407993480713071257946606195020588405650612863452436083584052624634527730514451908046325384940032"
        "2348451303638818760853390915395496414751342542716928"},
    {0x1.7a93a2954f3b7p+903, chars_format::general, 309,
        "99999999999999991537227438137387396469434575991841521388557198562770454753131655626431591234374844785939841297"
        "82457854396308324523168344957772266171277227355618234136662976348917763748975572076316639552336839557855469946"
        "9776634573397170474480057796161122485794632428945408"},
    {0x1.7a93a2954f3b8p+903, chars_format::general, 309,
        "10000000000000000655226109574678785641174996701035524401207638566177752810893043715169471647283826068076023845"
        "84873402410711216146426086879431039943172587970791041546464400835686314826715608754364230953016592202185142353"
        "05581886882057848563849292034690350260273827761094656"},
    {0x1.d9388b3aa30a5p+906, chars_format::general, 309,
        "99999999999999994540234169659267488457897654195544265913261035982571869424291411931484216282067527964903920729"
        "95713088338469091911386849725079892823366957826076670402259182750506840652611675169781773547902656050654660663"
        "69376850351293060923539046438669680406904714953752576"},
    {0x1.d9388b3aa30a6p+906, chars_format::general, 309,
        "10000000000000000655226109574678785641174996701035524401207638566177752810893043715169471647283826068076023845"
        "84873402410711216146426086879431039943172587970791041546464400835686314826715608754364230953016592202185142353"
        "055818868820578485638492920346903502602738277610946560"},
    {0x1.27c35704a5e67p+910, chars_format::general, 309,
        "99999999999999992137828784441763414867127191632582070293497966046730737687363606887442116243913381421732657184"
        "25108901184740478000812045911233791501695173449709921389782217629235579129702792695009666351450002856415308090"
        "320884466574359759805482716570229159677380024223137792"},
    {0x1.27c35704a5e68p+910, chars_format::general, 309,
        "10000000000000001135707186618179600359329089213627963525160252553345979158278604723977891654914655376710276554"
        "98994239841456938928541047642200260207506944846064391348959793859940567131297385249318652392307122841033012867"
        "7303956762082926555244744699101970314810717026738241536"},
    {0x1.71b42cc5cf601p+913, chars_format::general, 309,
        "99999999999999995981677400789769932612359931733321583285118877944076548466448094957909476304960015890806678857"
        "38075600630706260257731732013387553616370028451896719809745361823269597566357004654645037865774247967198272207"
        "7174989256760731188933351130765773907040474247261585408"},
    {0x1.71b42cc5cf602p+913, chars_format::general, 309,
        "10000000000000001135707186618179600359329089213627963525160252553345979158278604723977891654914655376710276554"
        "98994239841456938928541047642200260207506944846064391348959793859940567131297385249318652392307122841033012867"
        "73039567620829265552447446991019703148107170267382415360"},
    {0x1.ce2137f743381p+916, chars_format::general, 309,
        "99999999999999992906598507711364718416173739652729972891822148426199899843180504501535588256122708315547461518"
        "87702241073933634452195983131664543924630144450147281073774846468042382817033635086936740654314851878571900913"
        "80020735839470243162305319587149880588271350432374194176"},
    {0x1.ce2137f743382p+916, chars_format::general, 309,
        "10000000000000000520691408002498557520091850797509641446500906649770649433625086632703114045147193861658433087"
        "28919567930102413767433897865855658269158968045714503601765690788895124181432711335776992950015243623307738608"
        "946937362752018518070418086469181314516804918593340833792"},
    {0x1.20d4c2fa8a030p+920, chars_format::general, 309,
        "99999999999999980606282935397743861631428971330363531318635230354693305350110142676040036060773478014510592164"
        "86208802846843131230052987604772505157670608443149526129892785047133523819740156816103551808477267524066415738"
        "131041089269219682541925527051184466597377822714075545600"},
    {0x1.20d4c2fa8a031p+920, chars_format::general, 309,
        "10000000000000000028678785109953723248702060064614983783573429926910385653902272159683291957333224649616958313"
        "12859830401018793638548178044779976718480586605434593404010408332058769821540972204943665396181740249127519201"
        "9201707119869992081071729797163687409453914913289541779456"},
    {0x1.6909f3b92c83dp+923, chars_format::general, 309,
        "99999999999999996350686867959178558315902274782992576532314485486221746301240205812674342870820492799837784938"
        "00120403777518975354396021879194314779378814532106652458061823665896863336275809002770033531149375497833436762"
        "9875739137498376013657689431411868208826074951744485326848"},
    {0x1.6909f3b92c83ep+923, chars_format::general, 309,
        "10000000000000001209509080052061325500037557823562162174599374061775018725237026894930864968086750758516497771"
        "11403200470819481947873905615361612440108702062106377878623086228466020285281146118943651525382148347160045778"
        "78441067382304555201896123592311891751678371676348215197696"},
    {0x1.c34c70a777a4cp+926, chars_format::general, 309,
        "99999999999999993201806081446891618979007614092466767489578634459916058111014193185347481508811089842772346383"
        "37338083591383806529527415024309952855037173314315227192428015942144195432968678565436737186614953903080032558"
        "01626734885371401760100025992318635002556156068237393526784"},
    {0x1.c34c70a777a4dp+926, chars_format::general, 309,
        "10000000000000000579732922749603937632658625685457000366052203856513881087191824369465492695684870167103410060"
        "18846736433592448182900184244384740055240373818548092825496324683715486704619720031476992256475264028209364937"
        "790149360843820835266007499279518823345374529865067232493568"},
    {0x1.1a0fc668aac6fp+930, chars_format::general, 309,
        "99999999999999983125387564607573413100944699882784178552823911175737855902290952777901525150381000380162943008"
        "56434658995751266289947873088679994697143921417382666342399831226135658142385861165970188884104804799869139102"
        "108086341186118549553740473625584843283014570307735223533568"},
    {0x1.1a0fc668aac70p+930, chars_format::general, 309,
        "10000000000000000327822459828620982485707052830214935642633335774409426031973743359279343786724117930538174975"
        "81824150818701634676910695695993991101293042521124778804245620065815273272355149596490328548912510300629092601"
        "3924448356521309485648260046220787856768108551057012647002112"},
    {0x1.6093b802d578bp+933, chars_format::general, 309,
        "99999999999999987155954971343300695452169865566657214127525800489409136785780248940879907693753036165206704358"
        "48796028834004282385779689862931977960301222176155690682411105112539073058618988125756808205108864441153496484"
        "4713587442531567367726443881446254459800333664575907082272768"},
    {0x1.6093b802d578cp+933, chars_format::general, 309,
        "10000000000000000327822459828620982485707052830214935642633335774409426031973743359279343786724117930538174975"
        "81824150818701634676910695695993991101293042521124778804245620065815273272355149596490328548912510300629092601"
        "39244483565213094856482600462207878567681085510570126470021120"},
    {0x1.b8b8a6038ad6ep+936, chars_format::general, 309,
        "99999999999999990380408896731882521333149998113755642587287311940346161492571685871262613728450664793241713438"
        "42685124704606695262445143282333564570827062783174110154420124221661804991605489693586103661912112154180982390"
        "36197666670678728654776751975985792813764840337747509598224384"},
    {0x1.b8b8a6038ad6fp+936, chars_format::general, 309,
        "10000000000000000327822459828620982485707052830214935642633335774409426031973743359279343786724117930538174975"
        "81824150818701634676910695695993991101293042521124778804245620065815273272355149596490328548912510300629092601"
        "392444835652130948564826004622078785676810855105701264700211200"},
    {0x1.137367c236c65p+940, chars_format::general, 309,
        "99999999999999995539535177353613442742718210189113128122905730261845401023437984959874943383966870598097727966"
        "32907678097570555865109868753376103147668407754403581309634554796258176084383892202112976392797308495024959839"
        "786965342632596166187964530344229899589832462449290116390191104"},
    {0x1.137367c236c66p+940, chars_format::general, 309,
        "10000000000000001617604029984053712838099105849054307026537940354784235914690318131432426200603169381752178607"
        "79379789166942599827576877063754625745503378763932146593049227709464366045549750223622046731633809385840086963"
        "7486920046335831684748752572681717785398568698736550198021980160"},
    {0x1.585041b2c477ep+943, chars_format::general, 309,
        "99999999999999991412234152856228705615063640528827139694410995604646009398744945688985079659553905954212916344"
        "00729635383199467382978088376542072286195331777420004385463010336581079210161170195291478208089151422349777880"
        "2469744018919490624758069218767323224280852151918381000638332928"},
    {0x1.585041b2c477fp+943, chars_format::general, 309,
        "10000000000000000792143825084576765412568191916997109340838993423344357589751710277254453455720576452975216283"
        "32944180624068382131150520988387819573208763568535431208214918817528946670705205822257747094692177971305050571"
        "84069381648545374773244373557467226310750742042216461653692645376"},
    {0x1.ae64521f7595ep+946, chars_format::general, 309,
        "99999999999999998015915792052044285019310951985284721180002571056165035998253808522408861618614649384428614939"
        "72214503726193208954388936979476521664552253340593727464137481472064434208917525406205875303622202738630069015"
        "51095990707698442841525909542472844588688081080376132618600579072"},
    {0x1.ae64521f7595fp+946, chars_format::general, 309,
        "10000000000000001122327907044367544382780557489819988415118572195920308919727153418925642553673613624486001213"
        "11518424041218069209721063418534542042126609646694117362148642374303114420643023582803466949468830537119065128"
        "603893091744705516029416344252072069280447200202760777843035078656"},
    {0x1.0cfeb353a97dap+950, chars_format::general, 309,
        "99999999999999982167079857982086894449117404489786525614582789972519372159432537722191784916868865151910938310"
        "00650819703008229183002900332433843156495641588976792075318750746904382211902272900011322274342879579557370290"
        "877394694632899550160573878909537749585771381335145583492791795712"},
    {0x1.0cfeb353a97dbp+950, chars_format::general, 309,
        "10000000000000000329886110340869674854270880115045078636847583141738025727786089878914788718586324412860117381"
        "62940239840058820221151761586182408116723779059113270592707705838045111820792260957493739298004864379165430192"
        "3722148311225012721166820834263125344653917287293299907083743789056"},
    {0x1.503e602893dd1p+953, chars_format::general, 309,
        "99999999999999990619792356152730836086553963154052229916140006550463726206803882148974225824466616742587032512"
        "52151451182040218394408786544189938360792501189839157616022073800323076610310407569981750556625185264396142944"
        "0152961412697448185630726610509727876130297437184073129291725930496"},
    {0x1.503e602893dd2p+953, chars_format::general, 309,
        "10000000000000000752521735249401871936142708048258363851925443970635243430154657100253910763966211992393922091"
        "75515271414010419681722055896770212876938622039156388869742871990716046540712667690992260712118979663407368825"
        "02910990345434353553680702253338428636675464684849307718019341877248"},
    {0x1.a44df832b8d45p+956, chars_format::general, 309,
        "99999999999999987238707356884473259431579339688345948195517119919285984587855344378261249461427516106316594831"
        "51551198590427422709846432059487500279073757349494211399740744578955598850947153701993579243712262990460633882"
        "76013556261500671120207314819439877240212639876510262115462027411456"},
    {0x1.a44df832b8d46p+956, chars_format::general, 309,
        "10000000000000000076304735395750356605147783355117107507800866644399695106364949546111315491358391865139834555"
        "55395220895687860544809584999829725260594873271087399626486606146442550988840016917394626449536395208620267012"
        "778077787723395914064607119962069483324573977857832138825282954985472"},
    {0x1.06b0bb1fb384bp+960, chars_format::general, 309,
        "99999999999999984533839357469867198107599640915780922819018810614343791292696514161690868370996235597300244686"
        "71070996517137186162196548471725549813698762277218254426715681201861616643456550607603042193381925171312226633"
        "756007099691216225313273537909139560233403722802458867734978418966528"},
    {0x1.06b0bb1fb384cp+960, chars_format::general, 309,
        "10000000000000000617278335278671568869943723109630112583100528505388133765396715589425391709444647966943104584"
        "51491261310345907854339561717382115353669872285542591021091618821861347430338137536272733859602462772449948462"
        "5789034803081540112423670420191213257583185130503608895092113260150784"},
    {0x1.485ce9e7a065ep+963, chars_format::general, 309,
        "99999999999999988861628156533236896225967158951884963421416105502251300564950642508203478115686284411726404918"
        "39839319834401564638436362212144670558298754392859785583555782605211988175441515558627901473910465681949678232"
        "1626126403692810027353529143655542997033600043426888732064053872033792"},
    {0x1.485ce9e7a065fp+963, chars_format::general, 309,
        "10000000000000000617278335278671568869943723109630112583100528505388133765396715589425391709444647966943104584"
        "51491261310345907854339561717382115353669872285542591021091618821861347430338137536272733859602462772449948462"
        "57890348030815401124236704201912132575831851305036088950921132601507840"},
    {0x1.9a742461887f6p+966, chars_format::general, 309,
        "99999999999999995786090235034628413215355187809651428385251777322903315400557247862623653707190362514808261289"
        "09868637142024570200420064196815263749658741777886235434499944850572582626617459480267676322756130498969600789"
        "61318150545418464661067991669581788285529005480705688196068853638234112"},
    {0x1.9a742461887f7p+966, chars_format::general, 309,
        "10000000000000000963501439203741144719413124552518435831292312096420734507177045857146400489019851872097197403"
        "04992727175727058132438746816615645013237871654793913513638826934129377152896934732354722602044746013300944590"
        "451431923562399193436133392135634504915915015573579289946925483474026496"},
    {0x1.008896bcf54f9p+970, chars_format::general, 309,
        "99999999999999979167381246631288772440823918551011912472046164953338479795101395012015232287580575067411805999"
        "41798275603729356851659179433605840090394772053822755792233955461707155943795194068332216685526534938121786651"
        "731816229250415901309895111103185283290657933692573660950408978352832512"},
    {0x1.008896bcf54fap+970, chars_format::general, 309,
        "10000000000000000132565989783574162680686561089586460035632031477942492726904253214615979418039362499727374638"
        "56589209098812297465000702578455173830274673168590739531525527464686105818755821461757949620183266235258553883"
        "5573636597522107561710941518560028749376834095178551288964115055725510656"},
    {0x1.40aabc6c32a38p+973, chars_format::general, 309,
        "99999999999999992462348437353960485060448933957923525202610654848990348279466077292501969423268405025328970231"
        "16254564834365527530667887244173379017805947833073539506046746972799497290053006397880584395310211386800037962"
        "0369084502134308975505229555772913629423636305841602377586326247764393984"},
    {0x1.40aabc6c32a39p+973, chars_format::general, 309,
        "10000000000000001018897135831752276855328228783380567551002997470985950625861898699981761893751884496921852254"
        "01552961714188042176934616432493009758768751553874125112446380232092261908506342283727840800835511331837103970"
        "91103647448307842258713600815427661358113045597729423401695974866745819136"},
    {0x1.90d56b873f4c6p+976, chars_format::general, 309,
        "99999999999999992462348437353960485060448933957923525202610654848990348279466077292501969423268405025328970231"
        "16254564834365527530667887244173379017805947833073539506046746972799497290053006397880584395310211386800037962"
        "03690845021343089755052295557729136294236363058416023775863262477643939840"},
    {0x1.90d56b873f4c7p+976, chars_format::general, 309,
        "10000000000000000664364677412481031185471561705862924544854611073768567466278840505835448903466875698044061207"
        "83567460668037744292161050890877875387371120199760770880078039125129799472606133954939884328574613293205683935"
        "969567348590731356020719265634967118123751637393518591968740451429495341056"},
    {0x1.f50ac6690f1f8p+979, chars_format::general, 309,
        "99999999999999998134867772062300415778155607198205813300984837204468478832795008398842977267828545807373626970"
        "04022581572770293687044935910015528960168049498887207223940204684198896264456339658487887951484580004902758521"
        "100414464490983962613190835886243290260424727924570510530141380583845003264"},
    {0x1.f50ac6690f1f9p+979, chars_format::general, 309,
        "10000000000000000947990644147898027721356895367877038949773320191542473993945287061152499295694882737146294044"
        "77955861504957982599979903324169982884489225283051454265972712010699769421326300617970249506383331724110819963"
        "9227426493046090092738526596504147144896546922605391056073158892198656212992"},
    {0x1.3926bc01a973bp+983, chars_format::general, 309,
        "99999999999999998134867772062300415778155607198205813300984837204468478832795008398842977267828545807373626970"
        "04022581572770293687044935910015528960168049498887207223940204684198896264456339658487887951484580004902758521"
        "1004144644909839626131908358862432902604247279245705105301413805838450032640"},
    {0x1.3926bc01a973cp+983, chars_format::general, 309,
        "10000000000000001628692964312898819407481696156710913521578222074199849660344758793913420237042099630991652853"
        "44488023513566554538745149164071040877572677482949094392119926936067697298254700609243125933124255958283146431"
        "01036337101791537708137280528748894576782202394138833833989693991675429388288"},
    {0x1.87706b0213d09p+986, chars_format::general, 309,
        "99999999999999987243630649422287748800158794576863820152106407081950468170403460674668242206273075505847886031"
        "39507989435033142666801002471598601070832814300524965205584765878312050233601939798121865123629792258145535047"
        "69848291707808207769286850569305558980974742103098278680884456943362624192512"},
    {0x1.87706b0213d0ap+986, chars_format::general, 309,
        "10000000000000000176528014627563797143748787807198647768394431391197448238692552430690122228834703590788220728"
        "29219411228534934402712624705615450492327979456500795456339201761949451160807447294527656222743617592048849967"
        "890105831362861792425329827928397252374398383022243308510390698430058459037696"},
    {0x1.e94c85c298c4cp+989, chars_format::general, 309,
        "99999999999999995956620347534297882382556244673937414671209151179964876700316698854008030255517451747068478782"
        "31119663145222863482996149222332143382301002459214758820269116923021527058285459686414683385913622455551313826"
        "420028155008403585629126369847605750170289266545852965785882018353801250996224"},
    {0x1.e94c85c298c4dp+989, chars_format::general, 309,
        "10000000000000000757393994501697806049241951147003554069667947664398408807353434975979441432117662006869593578"
        "35326856142547582457125634488997686646425858667080115030651491831596749615786348620413844106895872938542568553"
        "1382088472248832262877470188720339297317678393899013204421931950247367929757696"},
    {0x1.31cfd3999f7afp+993, chars_format::general, 309,
        "99999999999999986662764669548153739894665631237058913850832890808749507601742578129378923002990117089766513181"
        "33400544521020494612387992688216364916734935089945645631272475808664751778623038472235677239477536911651816462"
        "4503799012160606438304513147494189124523779646633247748770420728389479079870464"},
    {0x1.31cfd3999f7b0p+993, chars_format::general, 309,
        "10000000000000000525047602552044202487044685811081591549158541155118024579889081957863713750804478640437044438"
        "32883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119"
        "03896764088007465274278014249457925878882005684283811566947219638686545940054016"},
    {0x1.7e43c8800759bp+996, chars_format::general, 309,
        "99999999999999990380306940742611396889821876611810314178983394957235655241172226419230565904001050952687299421"
        "72488191970701442160631255301862676302961362037653290906871132254407461890488006957907279698051971129211615408"
        "03823920273299782054992133678869364753954248541633605124057805104488924519071744"},
    {0x1.7e43c8800759cp+996, chars_format::general, 309,
        "10000000000000000525047602552044202487044685811081591549158541155118024579889081957863713750804478640437044438"
        "32883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119"
        "038967640880074652742780142494579258788820056842838115669472196386865459400540160"},
    {0x1.ddd4baa009302p+999, chars_format::general, 309,
        "99999999999999993354340757698177522485946872911611434441503798276024573352715945051111880224809798043023928414"
        "03758309930446200199225865392779725411942503595819407127350057411001629979979981746444561664911518503259454564"
        "508526643946547561925497354420113435609274102018745072331406833609642314953654272"},
    {0x1.ddd4baa009303p+999, chars_format::general, 309,
        "10000000000000000525047602552044202487044685811081591549158541155118024579889081957863713750804478640437044438"
        "32883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119"
        "0389676408800746527427801424945792587888200568428381156694721963868654594005401600"},
    {0x1.2aa4f4a405be1p+1003, chars_format::general, 309,
        "99999999999999988595886650569271721532146878831929642021471152965962304374245995240101777311515802698485322026"
        "33726121194854587337474489247312446837572677102753621174583777160450961036792822084784910517936242704782911914"
        "1560667380048679757245757262098417746977035154548906385860807815060374033329553408"},
    {0x1.2aa4f4a405be2p+1003, chars_format::general, 309,
        "10000000000000000762970307908489492534734685515065681170160173420621138028812579448414218896469178407663974757"
        "71385487613722103878447999382918156113505198307501676498564889816265363680954146073142351510583734589868908251"
        "55659063617715863205282622390509284183439858617103083735673849899204570498157510656"},
    {0x1.754e31cd072d9p+1006, chars_format::general, 309,
        "99999999999999984789123364866147080769106883568184208085445036717912489191470035391293694980880606422854436916"
        "17700370206381297048073388330938623978076815908300992412370752960010425882243094355457189600356022066001677793"
        "87409881325152430676383842364162444596844704620380709158981993982315347403639619584"},
    {0x1.754e31cd072dap+1006, chars_format::general, 309,
        "10000000000000000001617650767864564382126686462316594382954950171011174992257387478652602430342139152537797735"
        "68180337416027445820567779199643391541606026068611150746122284976177256650044200527276807327067690462112661427"
        "500197051226489898260678763391449376088547292320814127957486330655468919122263277568"},
    {0x1.d2a1be4048f90p+1009, chars_format::general, 309,
        "99999999999999993925355250553646218600402872201173249531907715713232045630132339028433092574405077484368561180"
        "56162172578717193742636030530235798840866882774987301441682011041067710253162440905843719802548551599076639682"
        "550821832659549112269607949805346034918662572406407604380845959862074904348138143744"},
    {0x1.d2a1be4048f91p+1009, chars_format::general, 309,
        "10000000000000000610699776480364506904213085704515863812719128770699145421501541054461895603243770556638739353"
        "30744457574183172266871955346263203199125363859723571348076368848247742274772156963969242673880525764317658886"
        "7453119191870248852943967318023641486852283274009874954768880653247303478097127407616"},
    {0x1.23a516e82d9bap+1013, chars_format::general, 309,
        "99999999999999993925355250553646218600402872201173249531907715713232045630132339028433092574405077484368561180"
        "56162172578717193742636030530235798840866882774987301441682011041067710253162440905843719802548551599076639682"
        "5508218326595491122696079498053460349186625724064076043808459598620749043481381437440"},
    {0x1.23a516e82d9bbp+1013, chars_format::general, 309,
        "10000000000000001341598327335364437930716764795154987128436143090324709936594525345433047410725728241559869294"
        "45821401763970044002436966722206977188148569209058476070421269494732325024445704688000165090055928126963655837"
        "83944976073966686973485829389546187580124556949719553650017014692784406223465209659392"},
    {0x1.6c8e5ca239028p+1016, chars_format::general, 309,
        "99999999999999986129104041433646954317696961901022600830926229637226024135807173258074139961264195511876508474"
        "95341434554323895229942575853502209624619359048748317736669737478565494256644598516180547363344259730852672204"
        "21335152276470127823801795414563694568114532338018850013250375609552861714878501486592"},
    {0x1.6c8e5ca239029p+1016, chars_format::general, 309,
        "10000000000000000172160645967364548288310878250132389823288920178923806712445750479879204518754595945686061388"
        "61698291060311049225532948520696938805711440650122628514669428460356992624968028329550689224175284346730060716"
        "088829214255439694630119794546505512415617982143262670862918816362862119154749127262208"},
    {0x1.c7b1f3cac7433p+1019, chars_format::general, 309,
        "99999999999999998603105976025645777170026418381263638752496607358835658526727438490648464142289606667863792803"
        "92654615393353172850252103336275952370615397010730691664689375178569039851073146339641623266071126720011020169"
        "553304018596457812688561947201171488461172921822139066929851282122002676667750021070848"},
    {0x1.c7b1f3cac7434p+1019, chars_format::general, 309,
        "10000000000000001107710791061764460002235587486150467667406698508044529291764770372322278832331501782385107713"
        "28996779623238245047056163081904969511661143497271306559270901287857258544550169416310269916879799370916936813"
        "4893256514428214347139105940256706031241200520264089633727198808148476736186715027275776"},
    {0x1.1ccf385ebc89fp+1023, chars_format::general, 309,
        "99999999999999981139503267596847425176765179308926185662298078548582170379439067165044410288854031049481594743"
        "36416162218712184181818764860392712526220943863955368165461882398564076018873179386796117002253512935189333018"
        "0773705244319986644578003569234231285691342840034082734135647456849389933411990123839488"},
    {0x1.1ccf385ebc8a0p+1023, chars_format::general, 309,
        "10000000000000000109790636294404554174049230967731184633681068290315758540491149153716332897849468889906124966"
        "97211725156115902837431400883283070091981460460312716645029330271856974896995885590433383844661650011784268976"
        "26212945177628091195786707458122783970171784415105291802893207873272974885715430223118336"},
    {0x1.fffffffffffffp+1023, chars_format::general, 309,
        "17976931348623157081452742373170435679807056752584499659891747680315726078002853876058955863276687817154045895"
        "35143824642343213268894641827684675467035375169860499105765512820762454900903893289440758685084551339423045832"
        "36903222948165808559332123348274797826204144723168738177180919299881250404026184124858368"},

    // The UCRT had trouble with rounding this value. charconv was never affected, but let's test it anyways.
    {0x1.88e2d605edc3dp+345, chars_format::general, 105,
        "109995565999999994887854821710219658911365648587951921896774663603198787416706536331386569598149846892544"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 19, "1.099955659999999949e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 18, "1.09995565999999995e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 17, "1.0999556599999999e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 16, "1.09995566e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 15, "1.09995566e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 14, "1.09995566e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 13, "1.09995566e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 12, "1.09995566e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 11, "1.09995566e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 10, "1.09995566e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 9, "1.09995566e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 8, "1.0999557e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 7, "1.099956e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 6, "1.09996e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 5, "1.1e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 4, "1.1e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 3, "1.1e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 2, "1.1e+104"},
    {0x1.88e2d605edc3dp+345, chars_format::general, 1, "1e+104"},

    // More cases that the UCRT had trouble with (e.g. DevCom-1093399).
    {0x1.8p+62, chars_format::general, 17, "6.9175290276410819e+18"},
    {0x1.0a2742p+17, chars_format::general, 6, "136271"},
    {0x1.f8b0f962cdffbp+205, chars_format::general, 14, "1.0137595739223e+62"},
    {0x1.f8b0f962cdffbp+205, chars_format::general, 17, "1.0137595739222531e+62"},
    {0x1.f8b0f962cdffbp+205, chars_format::general, 51, "1.01375957392225305727423222620636224221808910954041e+62"},
    {0x1.f8b0f962cdffbp+205, chars_format::general, 55, "1.013759573922253057274232226206362242218089109540405973e+62"},
};

#endif // DOUBLE_GENERAL_PRECISION_TO_CHARS_TEST_CASES_HPP
