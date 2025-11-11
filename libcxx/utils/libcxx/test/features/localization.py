# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import compilerMacros, Feature, programSucceeds, hasAnyLocale, programOutput, AddSubstitution
import re

features = [
    # Check for Glibc < 2.27, where the ru_RU.UTF-8 locale had
    # mon_decimal_point == ".", which our tests don't handle.
    Feature(
        name="glibc-old-ru_RU-decimal-point",
        when=lambda cfg: not "_LIBCPP_HAS_LOCALIZATION" in compilerMacros(cfg)
        or compilerMacros(cfg)["_LIBCPP_HAS_LOCALIZATION"] == "1"
        and not programSucceeds(
            cfg,
            """
            #include <locale.h>
            #include <string.h>
            int main(int, char**) {
              setlocale(LC_ALL, "ru_RU.UTF-8");
              return strcmp(localeconv()->mon_decimal_point, ",");
            }
          """,
        ),
    ),
]

# Mapping from canonical locale names (used in the tests) to possible locale
# names on various systems. Each locale is considered supported if any of the
# alternative names is supported.
_locales = {
    "en_US.UTF-8": ["en_US.UTF-8", "en_US.utf8", "English_United States.1252"],
    "fr_FR.UTF-8": ["fr_FR.UTF-8", "fr_FR.utf8", "French_France.1252"],
    "ja_JP.UTF-8": ["ja_JP.UTF-8", "ja_JP.utf8", "Japanese_Japan.923"],
    "ru_RU.UTF-8": ["ru_RU.UTF-8", "ru_RU.utf8", "Russian_Russia.1251"],
    "zh_CN.UTF-8": ["zh_CN.UTF-8", "zh_CN.utf8", "Chinese_China.936"],
    "fr_CA.ISO8859-1": ["fr_CA.ISO8859-1", "French_Canada.1252"],
    "cs_CZ.ISO8859-2": ["cs_CZ.ISO8859-2", "Czech_Czech Republic.1250"],
}
_provide_locale_conversions = {
    "fr_FR.UTF-8": ["decimal_point", "mon_thousands_sep", "thousands_sep"],
    "ru_RU.UTF-8": ["mon_thousands_sep"],
}
for locale, alts in _locales.items():
    # Note: Using alts directly in the lambda body here will bind it to the value at the
    # end of the loop. Assigning it to a default argument works around this issue.
    features.append(
        Feature(
            name="locale.{}".format(locale),
            when=lambda cfg, alts=alts: hasAnyLocale(cfg, alts),
            actions=lambda cfg, locale=locale, alts=alts: _getLocaleFlagsAction(
                cfg, locale, alts, _provide_locale_conversions[locale]
            )
            if locale in _provide_locale_conversions
            and ("_LIBCPP_HAS_WIDE_CHARACTERS" not in compilerMacros(cfg) or
                 compilerMacros(cfg)["_LIBCPP_HAS_WIDE_CHARACTERS"] == "1")
            else [],
        ),
    )

# Provide environment locale conversions through substitutions to avoid platform specific
# maintenance.
def _getLocaleFlagsAction(cfg, locale, alts, members):
    alts_list = ",".join([f'"{l}"' for l in alts])
    get_member_list = ",".join([f"lc->{m}" for m in members])

    localeconv_info = programOutput(
        cfg,
        r"""
        #if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
        #define _CRT_SECURE_NO_WARNINGS
        #endif
        #include <stdio.h>
        #include <locale.h>
        #include <stdlib.h>
        #include <wchar.h>

        // Print each requested locale conversion member on separate lines.
        int main(int, char**) {
          const char* locales[] = { %s };
          for (int loc_i = 0; loc_i < %d; ++loc_i) {
            if (!setlocale(LC_ALL, locales[loc_i])) {
              continue; // Choose first locale name that is recognized.
            }

            lconv* lc = localeconv();
            const char* members[] = { %s };
            for (size_t m_i = 0; m_i < %d; ++m_i) {
              if (!members[m_i]) {
                printf("\n"); // member value is an empty string
                continue;
              }

              size_t len = mbstowcs(nullptr, members[m_i], 0);
              if (len == static_cast<size_t>(-1)) {
                fprintf(stderr, "mbstowcs failed unexpectedly\n");
                return 1;
              }
              // Include room for null terminator. Use malloc as these features
              // are also used by lit configs that don't use -lc++ (libunwind tests).
              wchar_t* dst = (wchar_t*)malloc((len + 1) * sizeof(wchar_t));
              size_t ret = mbstowcs(dst, members[m_i], len + 1);
              if (ret == static_cast<size_t>(-1)) {
                fprintf(stderr, "mbstowcs failed unexpectedly\n");
                free(dst);
                return 1;
              }

              for (size_t i = 0; i < len; ++i) {
                if (dst[i] > 0x7F) {
                  printf("\\u%%04x", dst[i]);
                } else {
                  // c++03 does not allow basic ascii-range characters in UCNs
                  printf("%%c", (char)dst[i]);
                }
              }
              printf("\n");
              free(dst);
            }
            return 0;
          }

          return 1;
        }
        """
        % (alts_list, len(alts), get_member_list, len(members)),
    )
    valid_define_name = re.sub(r"[.-]", "_", locale).upper()
    return [
        # Provide locale conversion through a substitution.
        # Example: %{LOCALE_CONV_FR_FR_UTF_8_THOUSANDS_SEP} = L"\u202f"
        AddSubstitution(
            f"%{{LOCALE_CONV_{valid_define_name}_{member.upper()}}}",
            lambda cfg, value=value: f"'L\"{value}\"'",
        )
        for member, value in zip(members, localeconv_info.split("\n"))
    ]
