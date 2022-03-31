// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test.hpp"

#include <algorithm>
#include <array>
#include <assert.h>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <locale>
#include <optional>
#include <random>
#include <set>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "double_fixed_precision_to_chars_test_cases_1.hpp"
#include "double_fixed_precision_to_chars_test_cases_2.hpp"
#include "double_fixed_precision_to_chars_test_cases_3.hpp"
#include "double_fixed_precision_to_chars_test_cases_4.hpp"
#include "double_from_chars_test_cases.hpp"
#include "double_general_precision_to_chars_test_cases.hpp"
#include "double_hex_precision_to_chars_test_cases.hpp"
#include "double_scientific_precision_to_chars_test_cases_1.hpp"
#include "double_scientific_precision_to_chars_test_cases_2.hpp"
#include "double_scientific_precision_to_chars_test_cases_3.hpp"
#include "double_scientific_precision_to_chars_test_cases_4.hpp"
#include "double_to_chars_test_cases.hpp"
#include "float_fixed_precision_to_chars_test_cases.hpp"
#include "float_from_chars_test_cases.hpp"
#include "float_general_precision_to_chars_test_cases.hpp"
#include "float_hex_precision_to_chars_test_cases.hpp"
#include "float_scientific_precision_to_chars_test_cases.hpp"
#include "float_to_chars_test_cases.hpp"

using namespace std;

void initialize_randomness(mt19937_64& mt64, const int argc, char** const argv) {
    constexpr size_t n = mt19937_64::state_size;
    constexpr size_t w = mt19937_64::word_size;
    static_assert(w % 32 == 0);
    constexpr size_t k = w / 32;

    vector<uint32_t> vec(n * k);

    puts("USAGE:");
    puts("test.exe              : generates seed data from random_device.");
    puts("test.exe filename.txt : loads seed data from a given text file.");

    if (argc == 1) {
        random_device rd;
        generate(vec.begin(), vec.end(), ref(rd));
        puts("Generated seed data.");
    } else if (argc == 2) {
        const char* const filename = argv[1];

        ifstream file(filename);

        if (!file) {
            printf("ERROR: Can't open %s.\n", filename);
            abort();
        }

        for (auto& elem : vec) {
            file >> elem;

            if (!file) {
                printf("ERROR: Can't read seed data from %s.\n", filename);
                abort();
            }
        }

        printf("Loaded seed data from %s.\n", filename);
    } else {
        puts("ERROR: Too many command-line arguments.");
        abort();
    }

    puts("SEED DATA:");
    for (const auto& elem : vec) {
        printf("%zu ", static_cast<size_t>(elem));
    }
    printf("\n");

    seed_seq seq(vec.cbegin(), vec.cend());

    mt64.seed(seq);

    puts("Successfully seeded mt64. First three values:");
    for (int i = 0; i < 3; ++i) {
        // libc++ uses long for 64-bit values.
        printf("0x%016llX\n", static_cast<unsigned long long>(mt64()));
    }
}

static_assert((chars_format::scientific & chars_format::fixed) == chars_format{});
static_assert((chars_format::scientific & chars_format::hex) == chars_format{});
static_assert((chars_format::fixed & chars_format::hex) == chars_format{});
static_assert(chars_format::general == (chars_format::fixed | chars_format::scientific));

template <typename T, typename Optional>
void test_common_to_chars(
    const T value, const Optional opt_arg, const optional<int> opt_precision, const string_view correct) {

    // Important: Test every effective buffer size from 0 through correct.size() and slightly beyond. For the sizes
    // less than correct.size(), this verifies that the too-small buffer is correctly detected, and that we don't
    // attempt to write outside of it, even by a single char. (This exhaustive validation is necessary because the
    // implementation must check whenever it attempts to write. Sometimes we can calculate the total size and perform
    // a single check, but sometimes we need to check when writing each part of the result.) Testing correct.size()
    // verifies that we can succeed without overrunning, and testing slightly larger sizes verifies that we can succeed
    // without attempting to write to extra chars even when they're available. Finally, we also verify that we aren't
    // underrunning the buffer. This is a concern because sometimes we walk backwards when rounding.

    constexpr size_t BufferPrefix = 20; // detect buffer underruns (specific value isn't important)

    constexpr size_t Space = is_integral_v<T> ? 1 + 64 // worst case: -2^63 in binary
                           : is_same_v<T, float>
                               ? 1 + 151 // worst case: negative min subnormal float, fixed notation
                               : 1 + 1076; // worst case: negative min subnormal double, fixed notation

    constexpr size_t BufferSuffix = 30; // detect buffer overruns (specific value isn't important)

    array<char, BufferPrefix + Space + BufferSuffix> buff;

    char* const buff_begin = buff.data();
    char* const first      = buff_begin + BufferPrefix;
    char* const buff_end   = buff_begin + buff.size();

    constexpr size_t ExtraChars = 3;
    static_assert(ExtraChars + 10 < BufferSuffix,
        "The specific values aren't important, but there should be plenty of room to detect buffer overruns.");

    for (size_t n = 0; n <= correct.size() + ExtraChars; ++n) {
        assert(n <= static_cast<size_t>(buff_end - first));
        char* const last = first + n;

        buff.fill('@');
        const auto is_fill_char = [](const char c) { return c == '@'; };

        to_chars_result result{};
        if (opt_precision.has_value()) {
            assert(opt_arg.has_value());

            if constexpr (is_floating_point_v<T>) {
                result = to_chars(first, last, value, opt_arg.value(), opt_precision.value());
            } else {
                abort();
            }
        } else if (opt_arg.has_value()) {
            result = to_chars(first, last, value, opt_arg.value());
        } else {
            result = to_chars(first, last, value);
        }

        if (n < correct.size()) {
            assert(result.ptr == last);
            assert(result.ec == errc::value_too_large);
            assert(all_of(buff_begin, first, is_fill_char));
            // [first, last) is unspecified
            assert(all_of(last, buff_end, is_fill_char));
        } else {
            assert(result.ptr == first + correct.size());
            assert(result.ec == errc{});
            assert(all_of(buff_begin, first, is_fill_char));
            assert(equal(first, result.ptr, correct.begin(), correct.end()));
            assert(all_of(result.ptr, buff_end, is_fill_char));
        }
    }
}

template <typename T>
void test_integer_to_chars(const T value, const optional<int> opt_base, const string_view correct) {

    test_common_to_chars(value, opt_base, nullopt, correct);

    { // Also test successful from_chars() scenarios.
        const char* const correct_first = correct.data();
        const char* const correct_last  = correct_first + correct.size();

        T dest = 0;

        const from_chars_result from_res =
            (opt_base.has_value() ? from_chars(correct_first, correct_last, dest, opt_base.value())
                                  : from_chars(correct_first, correct_last, dest));

        assert(from_res.ptr == correct_last);
        assert(from_res.ec == errc{});
        assert(dest == value);
    }
}

// https://www.wolframalpha.com : Table[BaseForm[n * 2 - 1, n], {n, 2, 36}]
constexpr const char* output_max_digit[] = {"skip0", "skip1", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h", "1i", "1j", "1k", "1l", "1m", "1n", "1o", "1p", "1q", "1r", "1s",
    "1t", "1u", "1v", "1w", "1x", "1y", "1z"};

// https://www.wolframalpha.com : Table[BaseForm[k, n], {k, {MEOW, MEOW, MEOW}}, {n, 2, 36}]
constexpr uint64_t stress_chunks_positive                          = 12000000345000678900ULL;
constexpr pair<uint64_t, array<const char*, 37>> output_positive[] = {
    {123U, {{"skip0", "skip1", "1111011", "11120", "1323", "443", "323", "234", "173", "146", "123", "102", "a3", "96",
               "8b", "83", "7b", "74", "6f", "69", "63", "5i", "5d", "58", "53", "4n", "4j", "4f", "4b", "47", "43",
               "3u", "3r", "3o", "3l", "3i", "3f"}}},
    {uint64_t{INT8_MAX}, {{"skip0", "skip1", "1111111", "11201", "1333", "1002", "331", "241", "177", "151", "127",
                             "106", "a7", "9a", "91", "87", "7f", "78", "71", "6d", "67", "61", "5h", "5c", "57", "52",
                             "4n", "4j", "4f", "4b", "47", "43", "3v", "3s", "3p", "3m", "3j"}}},
    {161U, {{"skip0", "skip1", "10100001", "12222", "2201", "1121", "425", "320", "241", "188", "161", "137", "115",
               "c5", "b7", "ab", "a1", "98", "8h", "89", "81", "7e", "77", "70", "6h", "6b", "65", "5q", "5l", "5g",
               "5b", "56", "51", "4t", "4p", "4l", "4h"}}},
    {UINT8_MAX, {{"skip0", "skip1", "11111111", "100110", "3333", "2010", "1103", "513", "377", "313", "255", "212",
                    "193", "168", "143", "120", "ff", "f0", "e3", "d8", "cf", "c3", "bd", "b2", "af", "a5", "9l", "9c",
                    "93", "8n", "8f", "87", "7v", "7o", "7h", "7a", "73"}}},
    {1729U, {{"skip0", "skip1", "11011000001", "2101001", "123001", "23404", "12001", "5020", "3301", "2331", "1729",
                "1332", "1001", "a30", "8b7", "7a4", "6c1", "5gc", "561", "4f0", "469", "3j7", "3cd", "364", "301",
                "2j4", "2ed", "2a1", "25l", "21i", "1rj", "1oo", "1m1", "1jd", "1gt", "1ee", "1c1"}}},
    {uint64_t{INT16_MAX}, {{"skip0", "skip1", "111111111111111", "1122221121", "13333333", "2022032", "411411",
                              "164350", "77777", "48847", "32767", "22689", "16b67", "11bb7", "bd27", "9a97", "7fff",
                              "6b68", "5b27", "4eeb", "41i7", "3b67", "31f9", "2flf", "28l7", "22ah", "1mc7", "1hpg",
                              "1dm7", "19rq", "16c7", "1330", "vvv", "u2v", "sbp", "qq7", "pa7"}}},
    {57494U, {{"skip0", "skip1", "1110000010010110", "2220212102", "32002112", "3314434", "1122102", "326423", "160226",
                 "86772", "57494", "3a218", "29332", "20228", "16d4a", "1207e", "e096", "bbg0", "9f82", "8750", "73ee",
                 "647h", "58h8", "4gfh", "43je", "3goj", "3718", "2onb", "2h9a", "2aag", "23qe", "1spk", "1o4m", "1jq8",
                 "1fp0", "1bwo", "18d2"}}},
    {UINT16_MAX, {{"skip0", "skip1", "1111111111111111", "10022220020", "33333333", "4044120", "1223223", "362031",
                     "177777", "108806", "65535", "45268", "31b13", "23aa2", "19c51", "14640", "ffff", "d5d0", "b44f",
                     "9aa4", "83gf", "71cf", "638j", "58k8", "4hif", "44la", "3iof", "38o6", "2rgf", "2jqo", "2cof",
                     "2661", "1vvv", "1r5u", "1mnh", "1ihf", "1ekf"}}},
    {71125478U, {{"skip0", "skip1", "100001111010100100111100110", "11221211112210222", "10033110213212",
                    "121202003403", "11020244342", "1522361624", "417244746", "157745728", "71125478", "3716a696",
                    "1b9a06b2", "11973ba8", "9636514", "639e338", "43d49e6", "2g19gfb", "21b9d18", "19dec94", "124addi",
                    "h8f25b", "dhdfa6", "b13hg2", "8m91he", "7720j3", "5pgj58", "4pmelq", "43k17i", "3dg8ek", "2ro898",
                    "2f0et8", "23qif6", "1qw5lh", "1j7l7s", "1cdvli", "16cgrq"}}},
    {uint64_t{INT32_MAX},
        {{"skip0", "skip1", "1111111111111111111111111111111", "12112122212110202101", "1333333333333333",
            "13344223434042", "553032005531", "104134211161", "17777777777", "5478773671", "2147483647", "a02220281",
            "4bb2308a7", "282ba4aaa", "1652ca931", "c87e66b7", "7fffffff", "53g7f548", "3928g3h1", "27c57h32",
            "1db1f927", "140h2d91", "ikf5bf1", "ebelf95", "b5gge57", "8jmdnkm", "6oj8ion", "5ehncka", "4clm98f",
            "3hk7987", "2sb6cs7", "2d09uc1", "1vvvvvv", "1lsqtl1", "1d8xqrp", "15v22um", "zik0zj"}}},
    {3522553278ULL,
        {{"skip0", "skip1", "11010001111101011110010110111110", "100002111022020200020", "3101331132112332",
            "24203233201103", "1341312313010", "153202131426", "32175362676", "10074266606", "3522553278", "1548431462",
            "823842766", "441a34c6a", "255b8d486", "1593b4753", "d1f5e5be", "89ffb3b6", "5da3e606", "3hgbfb5i",
            "2f0fj33i", "1k1ac536", "191b46e2", "10i6fmk8", "ia967l6", "eahia63", "baca9ga", "92d86i6", "78iq4i6",
            "5qlc1dc", "4osos2i", "3u1862s", "38vbpdu", "2o0a7ro", "29hx9e6", "1w2dnod", "1m98ji6"}}},
    {UINT32_MAX,
        {{"skip0", "skip1", "11111111111111111111111111111111", "102002022201221111210", "3333333333333333",
            "32244002423140", "1550104015503", "211301422353", "37777777777", "12068657453", "4294967295", "1904440553",
            "9ba461593", "535a79888", "2ca5b7463", "1a20dcd80", "ffffffff", "a7ffda90", "704he7g3", "4f5aff65",
            "3723ai4f", "281d55i3", "1fj8b183", "1606k7ib", "mb994af", "hek2mgk", "dnchbnl", "b28jpdl", "8pfgih3",
            "76beigf", "5qmcpqf", "4q0jto3", "3vvvvvv", "3aokq93", "2qhxjlh", "2br45qa", "1z141z3"}}},
    {545890816626160ULL,
        {{"skip0", "skip1", "1111100000111110000011100001101100000110111110000", "2122120211122121121021010202111",
            "1330013300130031200313300", "1033022333343024014120", "5213002440142255104", "222661211220253465",
            "17407603415406760", "2576748547233674", "545890816626160", "148a34aa4706535", "51285369b87494",
            "1a57a38b045a95", "98b3383b9766c", "4319d1601875a", "1f07c1c360df0", "ffd471f34f13", "88g09ff9dh84",
            "4d0d5e232c53", "2d63h403i580", "1bf5h8185hdj", "kc3g550fkcg", "d41id5k9984", "8ef5n0him4g", "5i2dijfe1la",
            "3me22fm5fhi", "2hfmhgg73kd", "1ngpfabr53c", "18i7220bh11", "rm0lcjngpa", "kk1elesni1", "fgfge3c3fg",
            "bp4q5l6bjg", "8xna46jp0k", "6wejomvji5", "5di2s1qhv4"}}},
    {uint64_t{INT64_MAX},
        {{"skip0", "skip1", "111111111111111111111111111111111111111111111111111111111111111",
            "2021110011022210012102010021220101220221", "13333333333333333333333333333333",
            "1104332401304422434310311212", "1540241003031030222122211", "22341010611245052052300",
            "777777777777777777777", "67404283172107811827", "9223372036854775807", "1728002635214590697",
            "41a792678515120367", "10b269549075433c37", "4340724c6c71dc7a7", "160e2ad3246366807", "7fffffffffffffff",
            "33d3d8307b214008", "16agh595df825fa7", "ba643dci0ffeehh", "5cbfjia3fh26ja7", "2heiciiie82dh97",
            "1adaibb21dckfa7", "i6k448cf4192c2", "acd772jnc9l0l7", "64ie1focnn5g77", "3igoecjbmca687", "27c48l5b37oaop",
            "1bk39f3ah3dmq7", "q1se8f0m04isb", "hajppbc1fc207", "bm03i95hia437", "7vvvvvvvvvvvv", "5hg4ck9jd4u37",
            "3tdtk1v8j6tpp", "2pijmikexrxp7", "1y2p0ij32e8e7"}}},
    {stress_chunks_positive,
        {{"skip0", "skip1", "1010011010001000100100001011110000101100010101001001010111110100",
            "2221221122020020011022001202200200202200", "22122020210023300230111021113310",
            "1301130403021123030133211100", "2311004450342244200504500", "30325064311430214266301",
            "1232104413605425112764", "87848206138052620680", "12000000345000678900", "2181782a1686924456a",
            "54aa47a9058877b130", "150593a5b002c87b16", "571cad2b93c7760a8", "1c60d2676d4e53e00", "a68890bc2c5495f4",
            "43499224707a4f4g", "1e052gdga1d26f40", "f06dh4g564c8a91", "769df0d9ace4h50", "3ee7bcj1ajghi4f",
            "1k9agc4gfl0l43a", "10id7dakdlcjd22", "dge08fe0l5hl7c", "8184326d31ib60", "4ljbglf3cpim76",
            "2pph66481kiiki", "1niph2ao132e58", "14qgbgk3c3iffg", "mhc35an1bhb00", "f78o8ur705ln5", "ad24gngm595fk",
            "76e1n5i5v0ivl", "50wu8jsnks82g", "3ja41smfvqb1f", "2j64t3qgq0ut0"}}},
    {14454900944617508688ULL,
        {{"skip0", "skip1", "1100100010011010000111111101001011100011011000101000111101010000",
            "10120022020112011211121221212101012220210", "30202122013331023203120220331100",
            "1432224030234034034040234223", "3014532424232535441404120", "34610451042001242144165",
            "1442320775134330507520", "116266464747855335823", "14454900944617508688", "266642a9a9471339935",
            "662251403263939640", "1895280092bc310481", "68cb9c8292557406c", "23023deab20002893", "c89a1fd2e3628f50",
            "50e7147a7db8ef84", "22a34a05086f78ec", "i1dgef04357g7i1", "8g90b882jcj8be8", "49c1kk35i0k24ic",
            "272a16i54ebkacg", "15fdih7l3m7k8md", "gbj7303eg9nge0", "9hckfdkj3kkdmd", "5lc7hifdkl4nne",
            "3f86e4mgpna5ol", "266pj428na273c", "1bomgjbnlg4m3f", "r5tf1f7f009ji", "iarsig29iqhhm", "ch6gvqbhm53qg",
            "8lwtvcdj6rlqr", "61w23lajggp44", "49p1f3dsqqcdx", "31tkqqkxypopc"}}},
    {UINT64_MAX,
        {{"skip0", "skip1", "1111111111111111111111111111111111111111111111111111111111111111",
            "11112220022122120101211020120210210211220", "33333333333333333333333333333333",
            "2214220303114400424121122430", "3520522010102100444244423", "45012021522523134134601",
            "1777777777777777777777", "145808576354216723756", "18446744073709551615", "335500516a429071284",
            "839365134a2a240713", "219505a9511a867b72", "8681049adb03db171", "2c1d56b648c6cd110", "ffffffffffffffff",
            "67979g60f5428010", "2d3fgb0b9cg4bd2f", "141c8786h1ccaagg", "b53bjh07be4dj0f", "5e8g4ggg7g56dif",
            "2l4lf104353j8kf", "1ddh88h2782i515", "l12ee5fn0ji1if", "c9c336o0mlb7ef", "7b7n2pcniokcgf",
            "4eo8hfam6fllmo", "2nc6j26l66rhof", "1n3rsh11f098rn", "14l9lkmo30o40f", "nd075ib45k86f", "fvvvvvvvvvvvv",
            "b1w8p7j5q9r6f", "7orp63sh4dphh", "5g24a25twkwff", "3w5e11264sgsf"}}},
};

// https://www.wolframalpha.com : Table[BaseForm[k, n], {k, {MEOW, MEOW, MEOW}}, {n, 2, 36}]
constexpr int64_t stress_chunks_negative                          = -9000876000000054321LL;
constexpr pair<int64_t, array<const char*, 37>> output_negative[] = {
    {-85, {{"skip0", "skip1", "-1010101", "-10011", "-1111", "-320", "-221", "-151", "-125", "-104", "-85", "-78",
              "-71", "-67", "-61", "-5a", "-55", "-50", "-4d", "-49", "-45", "-41", "-3j", "-3g", "-3d", "-3a", "-37",
              "-34", "-31", "-2r", "-2p", "-2n", "-2l", "-2j", "-2h", "-2f", "-2d"}}},
    {INT8_MIN, {{"skip0", "skip1", "-10000000", "-11202", "-2000", "-1003", "-332", "-242", "-200", "-152", "-128",
                   "-107", "-a8", "-9b", "-92", "-88", "-80", "-79", "-72", "-6e", "-68", "-62", "-5i", "-5d", "-58",
                   "-53", "-4o", "-4k", "-4g", "-4c", "-48", "-44", "-40", "-3t", "-3q", "-3n", "-3k"}}},
    {-1591, {{"skip0", "skip1", "-11000110111", "-2011221", "-120313", "-22331", "-11211", "-4432", "-3067", "-2157",
                "-1591", "-1217", "-b07", "-955", "-819", "-711", "-637", "-58a", "-4g7", "-47e", "-3jb", "-3cg",
                "-367", "-304", "-2i7", "-2dg", "-295", "-24p", "-20n", "-1pp", "-1n1", "-1ka", "-1hn", "-1f7", "-1cr",
                "-1ag", "-187"}}},
    {INT16_MIN, {{"skip0", "skip1", "-1000000000000000", "-1122221122", "-20000000", "-2022033", "-411412", "-164351",
                    "-100000", "-48848", "-32768", "-2268a", "-16b68", "-11bb8", "-bd28", "-9a98", "-8000", "-6b69",
                    "-5b28", "-4eec", "-41i8", "-3b68", "-31fa", "-2flg", "-28l8", "-22ai", "-1mc8", "-1hph", "-1dm8",
                    "-19rr", "-16c8", "-1331", "-1000", "-u2w", "-sbq", "-qq8", "-pa8"}}},
    {-66748412,
        {{"skip0", "skip1", "-11111110100111111111111100", "-11122121011121102", "-3332213333330", "-114041422122",
            "-10342352232", "-1440231533", "-376477774", "-148534542", "-66748412", "-34750085", "-1a42b678",
            "-10aa0803", "-8c1731a", "-5cd7492", "-3fa7ffc", "-2d03163", "-1h5f3b2", "-17i39c6", "-10h3b0c", "-g749jh",
            "-ckkdkg", "-a8c0ak", "-894afk", "-6klmbc", "-5g1i6g", "-4hg4gb", "-3ogi7o", "-37anqb", "-2mc4r2",
            "-2a8h7i", "-1vkvvs", "-1n9ca5", "-1fw8sk", "-19gshh", "-13qnek"}}},
    {INT32_MIN, {{"skip0", "skip1", "-10000000000000000000000000000000", "-12112122212110202102", "-2000000000000000",
                    "-13344223434043", "-553032005532", "-104134211162", "-20000000000", "-5478773672", "-2147483648",
                    "-a02220282", "-4bb2308a8", "-282ba4aab", "-1652ca932", "-c87e66b8", "-80000000", "-53g7f549",
                    "-3928g3h2", "-27c57h33", "-1db1f928", "-140h2d92", "-ikf5bf2", "-ebelf96", "-b5gge58", "-8jmdnkn",
                    "-6oj8ioo", "-5ehnckb", "-4clm98g", "-3hk7988", "-2sb6cs8", "-2d09uc2", "-2000000", "-1lsqtl2",
                    "-1d8xqrq", "-15v22un", "-zik0zk"}}},
    {-297139747082649553LL,
        {{"skip0", "skip1", "-10000011111101001110000011010010001100000101011111111010001",
            "-1222110012002112101210012211022102101", "-100133221300122101200223333101", "-4443033200104011124241203",
            "-21313431255203203120401", "-350320603201030412545", "-20375160322140537721", "-1873162471705738371",
            "-297139747082649553", "-65150976074a24025", "-173522497b5373101", "-5a60a99bc3b71654", "-1ca51a06cc38ba25",
            "-a2a25babe62241d", "-41fa7069182bfd1", "-1d00134fba1769g", "-e4f799fc5f7e81", "-714ebbh8388188",
            "-3cahb17836b3hd", "-1j8659jf5hbg3j", "-112bbb2jege5c5", "-dcjfmk2kjb4cc", "-836bm4klbgl61",
            "-4ofia1416ee73", "-32ommgjef1l2h", "-1qc52eal5m8ba", "-17n53r05a4r15", "-oa88m2qiqjik", "-gn67qoat5r8d",
            "-blgd6n5s90al", "-87t70q8o5fuh", "-5t09hwaqu9qg", "-47vssihaoa4x", "-32p24fbjye7x", "-299r8zck3841"}}},
    {stress_chunks_negative,
        {{"skip0", "skip1", "-111110011101001100010010000100010000111010101111001010000110001",
            "-2012222010200021010000112111002001111200", "-13303221202100202013111321100301",
            "-1101001100304341000003214241", "-1522150121302454031001413", "-22054250360123016161454",
            "-763514220420725712061", "-65863607100474061450", "-9000876000000054321", "-1689813530958833498",
            "-408258185a67069269", "-106b01597a47ba2948", "-41c02922bc776d49b", "-1584cd10979dc84b6",
            "-7ce9890887579431", "-327cf6cbc67023c3", "-1604b5f6a0de8129", "-b50d3ef02f124a4", "-59h9bfif0006fg1",
            "-2g5d8ekh05d2dfi", "-19i418c38g1chfj", "-hjgf7d0k0gla9a", "-a6b21ncehfa3f9", "-61060fnl003bml",
            "-3g88bakondgf8l", "-25q3i730ed21di", "-1al84glo518iip", "-pcli8ig7pjhbo", "-gs31q8id2jnkl",
            "-bd7kaglgdrbgk", "-7pqc9123lf51h", "-5d2sd1r5ms7su", "-3q833s8kdrun3", "-2n7vmqigfueqb",
            "-1wdu892toj0a9"}}},
    {INT64_MIN, {{"skip0", "skip1", "-1000000000000000000000000000000000000000000000000000000000000000",
                    "-2021110011022210012102010021220101220222", "-20000000000000000000000000000000",
                    "-1104332401304422434310311213", "-1540241003031030222122212", "-22341010611245052052301",
                    "-1000000000000000000000", "-67404283172107811828", "-9223372036854775808", "-1728002635214590698",
                    "-41a792678515120368", "-10b269549075433c38", "-4340724c6c71dc7a8", "-160e2ad3246366808",
                    "-8000000000000000", "-33d3d8307b214009", "-16agh595df825fa8", "-ba643dci0ffeehi",
                    "-5cbfjia3fh26ja8", "-2heiciiie82dh98", "-1adaibb21dckfa8", "-i6k448cf4192c3", "-acd772jnc9l0l8",
                    "-64ie1focnn5g78", "-3igoecjbmca688", "-27c48l5b37oaoq", "-1bk39f3ah3dmq8", "-q1se8f0m04isc",
                    "-hajppbc1fc208", "-bm03i95hia438", "-8000000000000", "-5hg4ck9jd4u38", "-3tdtk1v8j6tpq",
                    "-2pijmikexrxp8", "-1y2p0ij32e8e8"}}},
};

template <typename T>
void test_integer_to_chars() {
    for (int base = 2; base <= 36; ++base) {
        test_integer_to_chars(static_cast<T>(0), base, "0");
        test_integer_to_chars(static_cast<T>(1), base, "1");

        // tests [3, 71]
        test_integer_to_chars(static_cast<T>(base * 2 - 1), base, output_max_digit[base]);

        for (const auto& p : output_positive) {
            if (p.first <= static_cast<uint64_t>(numeric_limits<T>::max())) {
                test_integer_to_chars(static_cast<T>(p.first), base, p.second[static_cast<size_t>(base)]);
            }
        }

        if constexpr (is_signed_v<T>) {
            test_integer_to_chars(static_cast<T>(-1), base, "-1");

            for (const auto& p : output_negative) {
                if (p.first >= static_cast<int64_t>(numeric_limits<T>::min())) {
                    test_integer_to_chars(static_cast<T>(p.first), base, p.second[static_cast<size_t>(base)]);
                }
            }
        }
    }

    test_integer_to_chars(static_cast<T>(42), nullopt, "42");
}

enum class TestFromCharsMode { Normal, SignalingNaN };

template <typename T, typename BaseOrFmt>
void test_from_chars(const string_view input, const BaseOrFmt base_or_fmt, const size_t correct_idx,
    const errc correct_ec, const optional<T> opt_correct = nullopt,
    const TestFromCharsMode mode = TestFromCharsMode::Normal) {

    if constexpr (is_integral_v<T>) {
        assert(mode == TestFromCharsMode::Normal);
    }

    constexpr T unmodified = 111;

    T dest = unmodified;

    const from_chars_result result = from_chars(input.data(), input.data() + input.size(), dest, base_or_fmt);

    assert(result.ptr == input.data() + correct_idx);
    assert(result.ec == correct_ec);

    if (correct_ec == errc{} || (is_floating_point_v<T> && correct_ec == errc::result_out_of_range)) {
        if constexpr (is_floating_point_v<T>) {
            if (mode == TestFromCharsMode::Normal) {
                using Uint = conditional_t<is_same_v<T, float>, uint32_t, uint64_t>;
                assert(opt_correct.has_value());
                assert(_Bit_cast<Uint>(dest) == _Bit_cast<Uint>(opt_correct.value()));
            } else {
                assert(mode == TestFromCharsMode::SignalingNaN);
                assert(!opt_correct.has_value());
                assert(isnan(dest));
            }
        } else {
            assert(opt_correct.has_value());
            assert(dest == opt_correct.value());
        }
    } else {
        assert(!opt_correct.has_value());
        assert(dest == unmodified);
    }
}

constexpr errc inv_arg = errc::invalid_argument;
constexpr errc out_ran = errc::result_out_of_range;

template <typename T>
void test_integer_from_chars() {
    for (int base = 2; base <= 36; ++base) {
        test_from_chars<T>("", base, 0, inv_arg); // no characters
        test_from_chars<T>("@1", base, 0, inv_arg); // '@' is bogus
        test_from_chars<T>(".1", base, 0, inv_arg); // '.' is bogus, for integers
        test_from_chars<T>("+1", base, 0, inv_arg); // '+' is bogus, N4713 23.20.3 [charconv.from.chars]/3
                                                    // "a minus sign is the only sign that may appear"
        test_from_chars<T>(" 1", base, 0, inv_arg); // ' ' is bogus, no whitespace in subject sequence

        if constexpr (is_unsigned_v<T>) { // N4713 23.20.3 [charconv.from.chars]/3
            test_from_chars<T>("-1", base, 0, inv_arg); // "and only if value has a signed type"
        }

        // N4713 23.20.3 [charconv.from.chars]/1 "[ Note: If the pattern allows for an optional sign,
        // but the string has no digit characters following the sign, no characters match the pattern. -end note ]"
        test_from_chars<T>("-", base, 0, inv_arg); // '-' followed by no characters
        test_from_chars<T>("-@1", base, 0, inv_arg); // '-' followed by bogus '@'
        test_from_chars<T>("-.1", base, 0, inv_arg); // '-' followed by bogus '.'
        test_from_chars<T>("-+1", base, 0, inv_arg); // '-' followed by bogus '+'
        test_from_chars<T>("- 1", base, 0, inv_arg); // '-' followed by bogus ' '
        test_from_chars<T>("--1", base, 0, inv_arg); // '-' can't be repeated

        vector<char> bogus_digits;

        if (base < 10) {
            bogus_digits = {static_cast<char>('0' + base), 'A', 'a'};
        } else {
            // '[' and '{' are bogus for base 36
            bogus_digits = {static_cast<char>('A' + (base - 10)), static_cast<char>('a' + (base - 10))};
        }

        for (const auto& bogus : bogus_digits) {
            test_from_chars<T>(bogus + "1"s, base, 0, inv_arg); // bogus digit (for this base)
            test_from_chars<T>("-"s + bogus + "1"s, base, 0, inv_arg); // '-' followed by bogus digit
        }

        // Test leading zeroes.
        test_from_chars<T>(string(100, '0'), base, 100, errc{}, static_cast<T>(0));
        test_from_chars<T>(string(100, '0') + "11"s, base, 102, errc{}, static_cast<T>(base + 1));

        // Test negative zero and negative leading zeroes.
        if constexpr (is_signed_v<T>) {
            test_from_chars<T>("-0", base, 2, errc{}, static_cast<T>(0));
            test_from_chars<T>("-"s + string(100, '0'), base, 101, errc{}, static_cast<T>(0));
            test_from_chars<T>("-"s + string(100, '0') + "11"s, base, 103, errc{}, static_cast<T>(-base - 1));
        }

        // N4713 23.20.3 [charconv.from.chars]/1 "The member ptr of the return value points to the
        // first character not matching the pattern, or has the value last if all characters match."
        test_from_chars<T>("11", base, 2, errc{}, static_cast<T>(base + 1));
        test_from_chars<T>("11@@@", base, 2, errc{}, static_cast<T>(base + 1));

        // When overflowing, we need to keep consuming valid digits, in order to return ptr correctly.
        test_from_chars<T>(string(100, '1'), base, 100, out_ran);
        test_from_chars<T>(string(100, '1') + "@@@"s, base, 100, out_ran);

        if constexpr (is_signed_v<T>) {
            test_from_chars<T>("-"s + string(100, '1'), base, 101, out_ran);
            test_from_chars<T>("-"s + string(100, '1') + "@@@"s, base, 101, out_ran);
        }
    }

    // N4713 23.20.3 [charconv.from.chars]/3 "The pattern is the expected form of the subject sequence
    // in the "C" locale for the given nonzero base, as described for strtol"
    // C11 7.22.1.4/3 "The letters from a (or A) through z (or Z) are ascribed the values 10 through 35"
    for (int i = 0; i < 26; ++i) {
        test_from_chars<T>(string(1, static_cast<char>('A' + i)), 36, 1, errc{}, static_cast<T>(10 + i));
        test_from_chars<T>(string(1, static_cast<char>('a' + i)), 36, 1, errc{}, static_cast<T>(10 + i));
    }

    // N4713 23.20.3 [charconv.from.chars]/3 "no "0x" or "0X" prefix shall appear if the value of base is 16"
    test_from_chars<T>("0x1729", 16, 1, errc{}, static_cast<T>(0)); // reads '0', stops at 'x'
    test_from_chars<T>("0X1729", 16, 1, errc{}, static_cast<T>(0)); // reads '0', stops at 'X'

    if constexpr (is_signed_v<T>) {
        test_from_chars<T>("-0x1729", 16, 2, errc{}, static_cast<T>(0)); // reads "-0", stops at 'x'
        test_from_chars<T>("-0X1729", 16, 2, errc{}, static_cast<T>(0)); // reads "-0", stops at 'X'
    }
}

template <typename T>
void test_integer() {
    test_integer_to_chars<T>();
    test_integer_from_chars<T>();
}

void all_integer_tests() {
    test_integer<char>();
    test_integer<signed char>();
    test_integer<unsigned char>();
    test_integer<short>();
    test_integer<unsigned short>();
    test_integer<int>();
    test_integer<unsigned int>();
    test_integer<long>();
    test_integer<unsigned long>();
    test_integer<long long>();
    test_integer<unsigned long long>();

    // Test overflow scenarios.
    test_from_chars<unsigned int>("4294967289", 10, 10, errc{}, 4294967289U); // not risky
    test_from_chars<unsigned int>("4294967294", 10, 10, errc{}, 4294967294U); // risky with good digit
    test_from_chars<unsigned int>("4294967295", 10, 10, errc{}, 4294967295U); // risky with max digit
    test_from_chars<unsigned int>("4294967296", 10, 10, out_ran); // risky with bad digit
    test_from_chars<unsigned int>("4294967300", 10, 10, out_ran); // beyond risky

    test_from_chars<int>("2147483639", 10, 10, errc{}, 2147483639); // not risky
    test_from_chars<int>("2147483646", 10, 10, errc{}, 2147483646); // risky with good digit
    test_from_chars<int>("2147483647", 10, 10, errc{}, 2147483647); // risky with max digit
    test_from_chars<int>("2147483648", 10, 10, out_ran); // risky with bad digit
    test_from_chars<int>("2147483650", 10, 10, out_ran); // beyond risky

    test_from_chars<int>("-2147483639", 10, 11, errc{}, -2147483639); // not risky
    test_from_chars<int>("-2147483647", 10, 11, errc{}, -2147483647); // risky with good digit
    test_from_chars<int>("-2147483648", 10, 11, errc{}, -2147483647 - 1); // risky with max digit
    test_from_chars<int>("-2147483649", 10, 11, out_ran); // risky with bad digit
    test_from_chars<int>("-2147483650", 10, 11, out_ran); // beyond risky
}

void assert_message_bits(const bool b, const char* const msg, const uint32_t bits) {
    if (!b) {
        fprintf(stderr, "%s failed for 0x%08zX\n", msg, static_cast<size_t>(bits));
        fprintf(stderr, "This is a randomized test.\n");
        fprintf(stderr, "DO NOT IGNORE/RERUN THIS FAILURE.\n");
        fprintf(stderr, "You must report it to the STL maintainers.\n");
        abort();
    }
}

void assert_message_bits(const bool b, const char* const msg, const uint64_t bits) {
    if (!b) {
        // libc++ uses long for 64-bit values.
        fprintf(stderr, "%s failed for 0x%016llX\n", msg, static_cast<unsigned long long>(bits));
        fprintf(stderr, "This is a randomized test.\n");
        fprintf(stderr, "DO NOT IGNORE/RERUN THIS FAILURE.\n");
        fprintf(stderr, "You must report it to the STL maintainers.\n");
        abort();
    }
}

constexpr uint32_t FractionBits = 10; // Tunable for test coverage vs. performance.
static_assert(FractionBits >= 1, "Must test at least 1 fraction bit.");
static_assert(FractionBits <= 23, "There are only 23 fraction bits in a float.");

constexpr uint32_t Fractions = 1U << FractionBits;
constexpr uint32_t Mask32    = ~((1U << FractionBits) - 1U);
constexpr uint64_t Mask64    = ~((1ULL << FractionBits) - 1ULL);

constexpr uint32_t PrefixesToTest = 100; // Tunable for test coverage vs. performance.
static_assert(PrefixesToTest >= 1, "Must test at least 1 prefix.");

constexpr uint32_t PrefixLimit = 2 // sign bit
                               * 255 // non-INF/NAN exponents for float
                               * (1U << (23 - FractionBits)); // fraction bits in prefix
static_assert(PrefixesToTest <= PrefixLimit, "Too many prefixes.");

template <bool IsDouble>
void test_floating_prefix(const conditional_t<IsDouble, uint64_t, uint32_t> prefix) {

    using UIntType     = conditional_t<IsDouble, uint64_t, uint32_t>;
    using FloatingType = conditional_t<IsDouble, double, float>;

    // "-1.2345678901234567e-100" or "-1.23456789e-10"
    constexpr size_t buffer_size = IsDouble ? 24 : 15;
    char buffer[buffer_size];
// TODO Enable once std::from_chars has floating point support.
#if 0
    FloatingType val;
#endif

    // Exact sizes are difficult to prove for fixed notation.
    // This must be at least (IsDouble ? 327 : 48), and I suspect that that's exact.
    // Here's a loose upper bound:
    // 1 character for a negative sign
    // + 325 (for double; 46 for float) characters in the "0.000~~~000" prefix of the min subnormal
    // + 17 (for double; 9 for float) characters for round-trip digits
    constexpr size_t fixed_buffer_size = IsDouble ? 1 + 325 + 17 : 1 + 46 + 9;
    char fixed_buffer[fixed_buffer_size];

    // worst case: negative sign + max normal + null terminator
    constexpr size_t stdio_buffer_size = 1 + (IsDouble ? 309 : 39) + 1;
    char stdio_buffer[stdio_buffer_size];

    for (uint32_t frac = 0; frac < Fractions; ++frac) {
        const UIntType bits      = prefix + frac;
        const FloatingType input = _Bit_cast<FloatingType>(bits);

        {
            const auto to_result = to_chars(buffer, end(buffer), input, chars_format::scientific);
            assert_message_bits(to_result.ec == errc{}, "to_result.ec", bits);
// TODO Enable once std::from_chars has floating point support.
#if 0
            const char* const last = to_result.ptr;

            const auto from_result = from_chars(buffer, last, val);

            assert_message_bits(from_result.ptr == last, "from_result.ptr", bits);
            assert_message_bits(from_result.ec == errc{}, "from_result.ec", bits);
            assert_message_bits(_Bit_cast<UIntType>(val) == bits, "round-trip", bits);
#endif
        }

        {
            // Also verify that to_chars() and sprintf_s() emit the same output for integers in fixed notation.
            const auto fixed_result = to_chars(fixed_buffer, end(fixed_buffer), input, chars_format::fixed);
            assert_message_bits(fixed_result.ec == errc{}, "fixed_result.ec", bits);
            const string_view fixed_sv(fixed_buffer, static_cast<size_t>(fixed_result.ptr - fixed_buffer));

            if (find(fixed_sv.begin(), fixed_sv.end(), '.') == fixed_sv.end()) {
                const int stdio_ret = sprintf_s(stdio_buffer, size(stdio_buffer), "%.0f", input);
                assert_message_bits(stdio_ret != -1, "stdio_ret", bits);
                const string_view stdio_sv(stdio_buffer);
                assert_message_bits(fixed_sv == stdio_sv, "fixed_sv", bits);
            }
        }
    }
}

template <bool IsDouble>
void test_floating_hex_prefix(const conditional_t<IsDouble, uint64_t, uint32_t> prefix) {

    using UIntType     = conditional_t<IsDouble, uint64_t, uint32_t>;
    using FloatingType = conditional_t<IsDouble, double, float>;

    // The precision is the number of hexits after the decimal point.
    // These hexits correspond to the explicitly stored fraction bits.
    // double explicitly stores 52 fraction bits. 52 / 4 == 13, so we need 13 hexits.
    // float explicitly stores 23 fraction bits. 23 / 4 == 5.75, so we need 6 hexits.

    // "-1.fffffffffffffp+1023" or "-1.fffffep+127"
    constexpr size_t buffer_size = IsDouble ? 22 : 14;
    char buffer[buffer_size];
// TODO Enable once std::from_chars has floating point support.
#if 0
    FloatingType val;
#endif

    for (uint32_t frac = 0; frac < Fractions; ++frac) {
        const UIntType bits      = prefix + frac;
        const FloatingType input = _Bit_cast<FloatingType>(bits);

        const auto to_result = to_chars(buffer, end(buffer), input, chars_format::hex);
        assert_message_bits(to_result.ec == errc{}, "(hex) to_result.ec", bits);
// TODO Enable once std::from_chars has floating point support.
#if 0
        const char* const last = to_result.ptr;

        const auto from_result = from_chars(buffer, last, val, chars_format::hex);

        assert_message_bits(from_result.ptr == last, "(hex) from_result.ptr", bits);
        assert_message_bits(from_result.ec == errc{}, "(hex) from_result.ec", bits);
        assert_message_bits(_Bit_cast<UIntType>(val) == bits, "(hex) round-trip", bits);
#endif
    }
}

template <bool IsDouble>
void test_floating_precision_prefix(const conditional_t<IsDouble, uint64_t, uint32_t> prefix) {

    using UIntType     = conditional_t<IsDouble, uint64_t, uint32_t>;
    using FloatingType = conditional_t<IsDouble, double, float>;

    // Precision for min subnormal in fixed notation. (More than enough for scientific notation.)
    constexpr int precision = IsDouble ? 1074 : 149;

    // Number of digits for max normal in fixed notation.
    constexpr int max_integer_length = IsDouble ? 309 : 39;

    // Size for fixed notation. (More than enough for scientific notation.)
    constexpr size_t charconv_buffer_size = 1 // negative sign
                                          + max_integer_length // integer digits
                                          + 1 // decimal point
                                          + precision; // fractional digits
    char charconv_buffer[charconv_buffer_size];

    constexpr size_t stdio_buffer_size = charconv_buffer_size + 1; // null terminator
    char stdio_buffer[stdio_buffer_size];

    // 1 character for a negative sign
    // + worst cases: 0x1.fffffffffffffp-1022 and 0x1.fffffep-126f
    constexpr size_t general_buffer_size = 1 + (IsDouble ? 773 : 117);
    char general_buffer[general_buffer_size];
    char general_stdio_buffer[general_buffer_size + 1]; // + null terminator

    for (uint32_t frac = 0; frac < Fractions; ++frac) {
        const UIntType bits      = prefix + frac;
        const FloatingType input = _Bit_cast<FloatingType>(bits);

        auto result = to_chars(charconv_buffer, end(charconv_buffer), input, chars_format::fixed, precision);
        assert_message_bits(result.ec == errc{}, "to_chars fixed precision", bits);
        string_view charconv_sv(charconv_buffer, static_cast<size_t>(result.ptr - charconv_buffer));

        int stdio_ret = sprintf_s(stdio_buffer, size(stdio_buffer), "%.*f", precision, input);
        assert_message_bits(stdio_ret != -1, "sprintf_s fixed precision", bits);
        string_view stdio_sv(stdio_buffer);

        assert_message_bits(charconv_sv == stdio_sv, "fixed precision output", bits);


        result = to_chars(charconv_buffer, end(charconv_buffer), input, chars_format::scientific, precision);
        assert_message_bits(result.ec == errc{}, "to_chars scientific precision", bits);
        charconv_sv = string_view(charconv_buffer, static_cast<size_t>(result.ptr - charconv_buffer));

        stdio_ret = sprintf_s(stdio_buffer, size(stdio_buffer), "%.*e", precision, input);
        assert_message_bits(stdio_ret != -1, "sprintf_s scientific precision", bits);
        stdio_sv = stdio_buffer;

        assert_message_bits(charconv_sv == stdio_sv, "scientific precision output", bits);


        result = to_chars(general_buffer, end(general_buffer), input, chars_format::general, 5000);
        assert_message_bits(result.ec == errc{}, "to_chars general precision", bits);
        charconv_sv = string_view(general_buffer, static_cast<size_t>(result.ptr - general_buffer));

        stdio_ret = sprintf_s(general_stdio_buffer, size(general_stdio_buffer), "%.5000g", input);
        assert_message_bits(stdio_ret != -1, "sprintf_s general precision", bits);
        stdio_sv = general_stdio_buffer;

        assert_message_bits(charconv_sv == stdio_sv, "general precision output", bits);
    }
}

void test_floating_prefixes(mt19937_64& mt64) {
    {
        set<uint64_t> prefixes64;

        while (prefixes64.size() < PrefixesToTest) {
            const uint64_t val = mt64();

            if ((val & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL) { // skip INF/NAN
                prefixes64.insert(val & Mask64);
            }
        }

        for (const auto& prefix : prefixes64) {
            test_floating_prefix<true>(prefix);
            test_floating_precision_prefix<true>(prefix);
        }

        test_floating_hex_prefix<true>(*prefixes64.begin()); // save time by testing fewer hexfloats
    }

    {
        set<uint32_t> prefixes32;

        while (prefixes32.size() < PrefixesToTest) {
            const uint32_t val = static_cast<uint32_t>(mt64());

            if ((val & 0x7F800000U) != 0x7F800000U) { // skip INF/NAN
                prefixes32.insert(val & Mask32);
            }
        }

        for (const auto& prefix : prefixes32) {
            test_floating_prefix<false>(prefix);
            test_floating_precision_prefix<false>(prefix);
        }

        test_floating_hex_prefix<false>(*prefixes32.begin()); // save time by testing fewer hexfloats
    }
}

// TODO Enable once std::from_chars has floating point support.
#if 0
template <typename T>
void test_floating_from_chars(const chars_format fmt) {
    test_from_chars<T>("", fmt, 0, inv_arg); // no characters
    test_from_chars<T>("@1", fmt, 0, inv_arg); // '@' is bogus
    test_from_chars<T>("z1", fmt, 0, inv_arg); // 'z' is bogus
    test_from_chars<T>(".", fmt, 0, inv_arg); // '.' without digits is bogus
    test_from_chars<T>("+1", fmt, 0, inv_arg); // '+' is bogus
    test_from_chars<T>(" 1", fmt, 0, inv_arg); // ' ' is bogus
    test_from_chars<T>("p5", fmt, 0, inv_arg); // binary-exponent-part without digits is bogus
    test_from_chars<T>("in", fmt, 0, inv_arg); // incomplete inf is bogus
    test_from_chars<T>("na", fmt, 0, inv_arg); // incomplete nan is bogus

    test_from_chars<T>("-", fmt, 0, inv_arg); // '-' followed by no characters
    test_from_chars<T>("-@1", fmt, 0, inv_arg); // '-' followed by bogus '@'
    test_from_chars<T>("-z1", fmt, 0, inv_arg); // '-' followed by bogus 'z'
    test_from_chars<T>("-.", fmt, 0, inv_arg); // '-' followed by bogus '.'
    test_from_chars<T>("-+1", fmt, 0, inv_arg); // '-' followed by bogus '+'
    test_from_chars<T>("- 1", fmt, 0, inv_arg); // '-' followed by bogus ' '
    test_from_chars<T>("-p5", fmt, 0, inv_arg); // '-' followed by bogus binary-exponent-part
    test_from_chars<T>("-in", fmt, 0, inv_arg); // '-' followed by bogus incomplete inf
    test_from_chars<T>("-na", fmt, 0, inv_arg); // '-' followed by bogus incomplete nan
    test_from_chars<T>("--1", fmt, 0, inv_arg); // '-' can't be repeated

    if (fmt != chars_format::hex) { // "e5" are valid hexits
        test_from_chars<T>("e5", fmt, 0, inv_arg); // exponent-part without digits is bogus
        test_from_chars<T>("-e5", fmt, 0, inv_arg); // '-' followed by bogus exponent-part
    }

    constexpr T inf  = numeric_limits<T>::infinity();
    constexpr T qnan = numeric_limits<T>::quiet_NaN();

    test_from_chars<T>("InF", fmt, 3, errc{}, inf);
    test_from_chars<T>("infinite", fmt, 3, errc{}, inf);
    test_from_chars<T>("iNfInItY", fmt, 8, errc{}, inf);
    test_from_chars<T>("InfinityMeow", fmt, 8, errc{}, inf);

    test_from_chars<T>("-InF", fmt, 4, errc{}, -inf);
    test_from_chars<T>("-infinite", fmt, 4, errc{}, -inf);
    test_from_chars<T>("-iNfInItY", fmt, 9, errc{}, -inf);
    test_from_chars<T>("-InfinityMeow", fmt, 9, errc{}, -inf);

    test_from_chars<T>("NaN", fmt, 3, errc{}, qnan);
    test_from_chars<T>("nanotech", fmt, 3, errc{}, qnan);
    test_from_chars<T>("nan(", fmt, 3, errc{}, qnan);
    test_from_chars<T>("nan(@)", fmt, 3, errc{}, qnan);
    test_from_chars<T>("nan(()", fmt, 3, errc{}, qnan);
    test_from_chars<T>("nan(abc", fmt, 3, errc{}, qnan);
    test_from_chars<T>("nan()", fmt, 5, errc{}, qnan);
    test_from_chars<T>("nan(abc)def", fmt, 8, errc{}, qnan);
    test_from_chars<T>("nan(_09AZaz)", fmt, 12, errc{}, qnan);
    test_from_chars<T>("nan(int)", fmt, 8, errc{}, qnan);
    test_from_chars<T>("nan(snap)", fmt, 9, errc{}, qnan);

    test_from_chars<T>("-NaN", fmt, 4, errc{}, -qnan);
    test_from_chars<T>("-nanotech", fmt, 4, errc{}, -qnan);
    test_from_chars<T>("-nan(", fmt, 4, errc{}, -qnan);
    test_from_chars<T>("-nan(@)", fmt, 4, errc{}, -qnan);
    test_from_chars<T>("-nan(()", fmt, 4, errc{}, -qnan);
    test_from_chars<T>("-nan(abc", fmt, 4, errc{}, -qnan);
    test_from_chars<T>("-nan()", fmt, 6, errc{}, -qnan);
    test_from_chars<T>("-nan(abc)def", fmt, 9, errc{}, -qnan);
    test_from_chars<T>("-nan(_09AZaz)", fmt, 13, errc{}, -qnan);
    test_from_chars<T>("-nan(int)", fmt, 9, errc{}, -qnan);
    test_from_chars<T>("-nan(snap)", fmt, 10, errc{}, -qnan);

    // The UCRT considers indeterminate NaN to be negative quiet NaN with no payload bits set.
    // It parses "nan(ind)" and "-nan(ind)" identically.
    test_from_chars<T>("nan(InD)", fmt, 8, errc{}, -qnan);
    test_from_chars<T>("-nan(InD)", fmt, 9, errc{}, -qnan);

    test_from_chars<T>("nan(SnAn)", fmt, 9, errc{}, nullopt, TestFromCharsMode::SignalingNaN);
    test_from_chars<T>("-nan(SnAn)", fmt, 10, errc{}, nullopt, TestFromCharsMode::SignalingNaN);

    switch (fmt) {
    case chars_format::general:
        test_from_chars<T>("1729", fmt, 4, errc{}, T{1729});
        test_from_chars<T>("1729e3", fmt, 6, errc{}, T{1729000});
        test_from_chars<T>("10", fmt, 2, errc{}, T{10});
        test_from_chars<T>("11.", fmt, 3, errc{}, T{11});
        test_from_chars<T>("12.13", fmt, 5, errc{}, static_cast<T>(12.13)); // avoid truncation warning
        test_from_chars<T>(".14", fmt, 3, errc{}, static_cast<T>(.14)); // avoid truncation warning
        test_from_chars<T>("20e5", fmt, 4, errc{}, T{2000000});
        test_from_chars<T>("21.e5", fmt, 5, errc{}, T{2100000});
        test_from_chars<T>("22.23e5", fmt, 7, errc{}, T{2223000});
        test_from_chars<T>(".24e5", fmt, 5, errc{}, T{24000});
        test_from_chars<T>("33e+5", fmt, 5, errc{}, T{3300000});
        test_from_chars<T>("33e-5", fmt, 5, errc{}, static_cast<T>(.00033)); // avoid truncation warning
        test_from_chars<T>("4E7", fmt, 3, errc{}, T{40000000});
        test_from_chars<T>("-00123abc", fmt, 6, errc{}, T{-123});
        test_from_chars<T>(".0045000", fmt, 8, errc{}, static_cast<T>(.0045)); // avoid truncation warning
        test_from_chars<T>("000", fmt, 3, errc{}, T{0});
        test_from_chars<T>("0e9999", fmt, 6, errc{}, T{0});
        test_from_chars<T>("0e-9999", fmt, 7, errc{}, T{0});
        test_from_chars<T>("-000", fmt, 4, errc{}, T{-0.0});
        test_from_chars<T>("-0e9999", fmt, 7, errc{}, T{-0.0});
        test_from_chars<T>("-0e-9999", fmt, 8, errc{}, T{-0.0});
        test_from_chars<T>("1e9999", fmt, 6, errc::result_out_of_range, inf);
        test_from_chars<T>("-1e9999", fmt, 7, errc::result_out_of_range, -inf);
        test_from_chars<T>("1e-9999", fmt, 7, errc::result_out_of_range, T{0});
        test_from_chars<T>("-1e-9999", fmt, 8, errc::result_out_of_range, T{-0.0});
        test_from_chars<T>("1" + string(6000, '0'), fmt, 6001, errc::result_out_of_range, inf);
        test_from_chars<T>("-1" + string(6000, '0'), fmt, 6002, errc::result_out_of_range, -inf);
        test_from_chars<T>("." + string(6000, '0') + "1", fmt, 6002, errc::result_out_of_range, T{0});
        test_from_chars<T>("-." + string(6000, '0') + "1", fmt, 6003, errc::result_out_of_range, T{-0.0});
        test_from_chars<T>("1" + string(500, '0'), fmt, 501, errc::result_out_of_range, inf);
        test_from_chars<T>("-1" + string(500, '0'), fmt, 502, errc::result_out_of_range, -inf);
        test_from_chars<T>("." + string(500, '0') + "1", fmt, 502, errc::result_out_of_range, T{0});
        test_from_chars<T>("-." + string(500, '0') + "1", fmt, 503, errc::result_out_of_range, T{-0.0});
        break;
    case chars_format::scientific:
        test_from_chars<T>("1729", fmt, 0, inv_arg);
        test_from_chars<T>("1729e3", fmt, 6, errc{}, T{1729000});
        break;
    case chars_format::fixed:
        test_from_chars<T>("1729", fmt, 4, errc{}, T{1729});
        test_from_chars<T>("1729e3", fmt, 4, errc{}, T{1729});
        break;
    case chars_format::hex:
        test_from_chars<T>("0x123", fmt, 1, errc{}, T{0});
        test_from_chars<T>("a0", fmt, 2, errc{}, T{160});
        test_from_chars<T>("a1.", fmt, 3, errc{}, T{161});
        test_from_chars<T>("a2.a3", fmt, 5, errc{}, T{162.63671875});
        test_from_chars<T>(".a4", fmt, 3, errc{}, T{0.640625});
        test_from_chars<T>("a0p5", fmt, 4, errc{}, T{5120});
        test_from_chars<T>("a1.p5", fmt, 5, errc{}, T{5152});
        test_from_chars<T>("a2.a3p5", fmt, 7, errc{}, T{5204.375});
        test_from_chars<T>(".a4p5", fmt, 5, errc{}, T{20.5});
        test_from_chars<T>("a0p+5", fmt, 5, errc{}, T{5120});
        test_from_chars<T>("a0p-5", fmt, 5, errc{}, T{5});
        test_from_chars<T>("ABCDEFP3", fmt, 8, errc{}, T{90075000});
        test_from_chars<T>("-00cdrom", fmt, 5, errc{}, T{-205});
        test_from_chars<T>(".00ef000", fmt, 8, errc{}, T{0.0036468505859375});
        test_from_chars<T>("000", fmt, 3, errc{}, T{0});
        test_from_chars<T>("0p9999", fmt, 6, errc{}, T{0});
        test_from_chars<T>("0p-9999", fmt, 7, errc{}, T{0});
        test_from_chars<T>("-000", fmt, 4, errc{}, T{-0.0});
        test_from_chars<T>("-0p9999", fmt, 7, errc{}, T{-0.0});
        test_from_chars<T>("-0p-9999", fmt, 8, errc{}, T{-0.0});
        test_from_chars<T>("1p9999", fmt, 6, errc::result_out_of_range, inf);
        test_from_chars<T>("-1p9999", fmt, 7, errc::result_out_of_range, -inf);
        test_from_chars<T>("1p-9999", fmt, 7, errc::result_out_of_range, T{0});
        test_from_chars<T>("-1p-9999", fmt, 8, errc::result_out_of_range, T{-0.0});
        test_from_chars<T>("1" + string(2000, '0'), fmt, 2001, errc::result_out_of_range, inf);
        test_from_chars<T>("-1" + string(2000, '0'), fmt, 2002, errc::result_out_of_range, -inf);
        test_from_chars<T>("." + string(2000, '0') + "1", fmt, 2002, errc::result_out_of_range, T{0});
        test_from_chars<T>("-." + string(2000, '0') + "1", fmt, 2003, errc::result_out_of_range, T{-0.0});
        test_from_chars<T>("1" + string(300, '0'), fmt, 301, errc::result_out_of_range, inf);
        test_from_chars<T>("-1" + string(300, '0'), fmt, 302, errc::result_out_of_range, -inf);
        test_from_chars<T>("." + string(300, '0') + "1", fmt, 302, errc::result_out_of_range, T{0});
        test_from_chars<T>("-." + string(300, '0') + "1", fmt, 303, errc::result_out_of_range, T{-0.0});
        break;
    }
}
#endif

template <typename T>
void test_floating_to_chars(
    const T value, const optional<chars_format> opt_fmt, const optional<int> opt_precision, const string_view correct) {

    test_common_to_chars(value, opt_fmt, opt_precision, correct);
}

void all_floating_tests(mt19937_64& mt64) {
    test_floating_prefixes(mt64);

// TODO Enable once std::from_chars has floating point support.
#if 0
    for (const auto& fmt : {chars_format::general, chars_format::scientific, chars_format::fixed, chars_format::hex}) {
        test_floating_from_chars<float>(fmt);
        test_floating_from_chars<double>(fmt);
    }

    // Test rounding.

    // See float_from_chars_test_cases.hpp in this directory.
    for (const auto& t : float_from_chars_test_cases) {
        test_from_chars<float>(t.input, t.fmt, t.correct_idx, t.correct_ec, t.correct_value);
    }

    // See double_from_chars_test_cases.hpp in this directory.
    for (const auto& t : double_from_chars_test_cases) {
        test_from_chars<double>(t.input, t.fmt, t.correct_idx, t.correct_ec, t.correct_value);
    }

    {
        // See LWG-2403. This number (exactly 0x1.fffffd00000004 in infinite precision) behaves differently
        // when parsed as double and converted to float, versus being parsed as float directly.
        const char* const lwg_2403          = "1.999999821186065729339276231257827021181583404541015625";
        constexpr float correct_float       = 0x1.fffffep0f;
        constexpr double correct_double     = 0x1.fffffdp0;
        constexpr float twice_rounded_float = 0x1.fffffcp0f;

        test_from_chars<float>(lwg_2403, chars_format::general, 56, errc{}, correct_float);
        test_from_chars<double>(lwg_2403, chars_format::general, 56, errc{}, correct_double);
        static_assert(static_cast<float>(correct_double) == twice_rounded_float);
    }

    // See floating_point_test_cases.hpp.
    for (const auto& p : floating_point_test_cases_float) {
        test_from_chars<float>(p.first, chars_format::general, strlen(p.first), errc{}, _Bit_cast<float>(p.second));
    }

    for (const auto& p : floating_point_test_cases_double) {
        test_from_chars<double>(p.first, chars_format::general, strlen(p.first), errc{}, _Bit_cast<double>(p.second));
    }
#endif
    // See float_to_chars_test_cases.hpp in this directory.
    for (const auto& t : float_to_chars_test_cases) {
        if (t.fmt == chars_format{}) {
            test_floating_to_chars(t.value, nullopt, nullopt, t.correct);
        } else {
            test_floating_to_chars(t.value, t.fmt, nullopt, t.correct);
        }
    }

    // See double_to_chars_test_cases.hpp in this directory.
    for (const auto& t : double_to_chars_test_cases) {
        if (t.fmt == chars_format{}) {
            test_floating_to_chars(t.value, nullopt, nullopt, t.correct);
        } else {
            test_floating_to_chars(t.value, t.fmt, nullopt, t.correct);
        }
    }

    // See corresponding headers in this directory.
    for (const auto& t : float_hex_precision_to_chars_test_cases) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : float_fixed_precision_to_chars_test_cases) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : float_scientific_precision_to_chars_test_cases) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : float_general_precision_to_chars_test_cases) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_hex_precision_to_chars_test_cases) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_fixed_precision_to_chars_test_cases_1) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_fixed_precision_to_chars_test_cases_2) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_fixed_precision_to_chars_test_cases_3) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_fixed_precision_to_chars_test_cases_4) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_scientific_precision_to_chars_test_cases_1) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_scientific_precision_to_chars_test_cases_2) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_scientific_precision_to_chars_test_cases_3) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_scientific_precision_to_chars_test_cases_4) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
    for (const auto& t : double_general_precision_to_chars_test_cases) {
        test_floating_to_chars(t.value, t.fmt, t.precision, t.correct);
    }
}

int main(int argc, char** argv) {
    const auto start = chrono::steady_clock::now();

    mt19937_64 mt64;

    initialize_randomness(mt64, argc, argv);

    all_integer_tests();

    all_floating_tests(mt64);

    const auto finish  = chrono::steady_clock::now();
    const long long ms = chrono::duration_cast<chrono::milliseconds>(finish - start).count();

    puts("PASS");
    printf("Randomized test cases: %zu\n", static_cast<size_t>(PrefixesToTest * Fractions));
    printf("Total time: %lld ms\n", ms);

    if (ms < 3'000) {
        puts("That was fast. Consider tuning PrefixesToTest and FractionBits to test more cases.");
    } else if (ms > 30'000) {
        puts("That was slow. Consider tuning PrefixesToTest and FractionBits to test fewer cases.");
    }
}
