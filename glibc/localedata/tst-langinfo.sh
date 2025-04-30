#!/bin/sh
# Test nl_langinfo.
# Copyright (C) 2000-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

set -e

common_objpfx=$1
tst_langinfo_before_env=$2
run_program_env=$3
tst_langinfo_after_env=$4

# Run the test program.
cat <<"EOF" |
# Only decimal numerical escape sequences allowed in strings.
C                    ABDAY_1     Sun
C                    ABDAY_2     Mon
C                    ABDAY_3     Tue
C                    ABDAY_4     Wed
C                    ABDAY_5     Thu
C                    ABDAY_6     Fri
C                    ABDAY_7     Sat
C                    DAY_1       Sunday
C                    DAY_2       Monday
C                    DAY_3       Tuesday
C                    DAY_4       Wednesday
C                    DAY_5       Thursday
C                    DAY_6       Friday
C                    DAY_7       Saturday
C                    ABMON_1     Jan
C                    ABMON_2     Feb
C                    ABMON_3     Mar
C                    ABMON_4     Apr
C                    ABMON_5     May
C                    ABMON_6     Jun
C                    ABMON_7     Jul
C                    ABMON_8     Aug
C                    ABMON_9     Sep
C                    ABMON_10    Oct
C                    ABMON_11    Nov
C                    ABMON_12    Dec
C                    MON_1       January
C                    MON_2       February
C                    MON_3       March
C                    MON_4       April
C                    MON_5       May
C                    MON_6       June
C                    MON_7       July
C                    MON_8       August
C                    MON_9       September
C                    MON_10      October
C                    MON_11      November
C                    MON_12      December
C                    AM_STR      AM
C                    PM_STR      PM
C                    D_T_FMT     "%a %b %e %H:%M:%S %Y"
C                    D_FMT       "%m/%d/%y"
C                    T_FMT       "%H:%M:%S"
C                    T_FMT_AMPM  "%I:%M:%S %p"
C                    ABDAY_1     Sun
C                    ABDAY_2     Mon
C                    ABDAY_3     Tue
C                    ABDAY_4     Wed
C                    ABDAY_5     Thu
C                    ABDAY_6     Fri
C                    ABDAY_7     Sat
C                    DAY_1       Sunday
C                    DAY_2       Monday
C                    DAY_3       Tuesday
C                    DAY_4       Wednesday
C                    DAY_5       Thursday
C                    DAY_6       Friday
C                    DAY_7       Saturday
C                    RADIXCHAR   .
C                    THOUSEP     ""
C                    YESEXPR     ^[yY]
C                    NOEXPR      ^[nN]
en_US.ANSI_X3.4-1968 ABMON_1     Jan
en_US.ANSI_X3.4-1968 ABMON_2     Feb
en_US.ANSI_X3.4-1968 ABMON_3     Mar
en_US.ANSI_X3.4-1968 ABMON_4     Apr
en_US.ANSI_X3.4-1968 ABMON_5     May
en_US.ANSI_X3.4-1968 ABMON_6     Jun
en_US.ANSI_X3.4-1968 ABMON_7     Jul
en_US.ANSI_X3.4-1968 ABMON_8     Aug
en_US.ANSI_X3.4-1968 ABMON_9     Sep
en_US.ANSI_X3.4-1968 ABMON_10    Oct
en_US.ANSI_X3.4-1968 ABMON_11    Nov
en_US.ANSI_X3.4-1968 ABMON_12    Dec
en_US.ANSI_X3.4-1968 MON_1       January
en_US.ANSI_X3.4-1968 MON_2       February
en_US.ANSI_X3.4-1968 MON_3       March
en_US.ANSI_X3.4-1968 MON_4       April
en_US.ANSI_X3.4-1968 MON_5       May
en_US.ANSI_X3.4-1968 MON_6       June
en_US.ANSI_X3.4-1968 MON_7       July
en_US.ANSI_X3.4-1968 MON_8       August
en_US.ANSI_X3.4-1968 MON_9       September
en_US.ANSI_X3.4-1968 MON_10      October
en_US.ANSI_X3.4-1968 MON_11      November
en_US.ANSI_X3.4-1968 MON_12      December
en_US.ANSI_X3.4-1968 AM_STR      AM
en_US.ANSI_X3.4-1968 PM_STR      PM
en_US.ANSI_X3.4-1968 D_T_FMT     "%a %d %b %Y %r %Z"
en_US.ANSI_X3.4-1968 D_FMT       "%m/%d/%Y"
en_US.ANSI_X3.4-1968 T_FMT       "%r"
en_US.ANSI_X3.4-1968 T_FMT_AMPM  "%I:%M:%S %p"
en_US.ANSI_X3.4-1968 RADIXCHAR   .
en_US.ANSI_X3.4-1968 THOUSEP     ,
en_US.ANSI_X3.4-1968 YESEXPR     ^[+1yY]
en_US.ANSI_X3.4-1968 NOEXPR      ^[-0nN]
en_US.ISO-8859-1     ABMON_1     Jan
en_US.ISO-8859-1     ABMON_2     Feb
en_US.ISO-8859-1     ABMON_3     Mar
en_US.ISO-8859-1     ABMON_4     Apr
en_US.ISO-8859-1     ABMON_5     May
en_US.ISO-8859-1     ABMON_6     Jun
en_US.ISO-8859-1     ABMON_7     Jul
en_US.ISO-8859-1     ABMON_8     Aug
en_US.ISO-8859-1     ABMON_9     Sep
en_US.ISO-8859-1     ABMON_10    Oct
en_US.ISO-8859-1     ABMON_11    Nov
en_US.ISO-8859-1     ABMON_12    Dec
en_US.ISO-8859-1     MON_1       January
en_US.ISO-8859-1     MON_2       February
en_US.ISO-8859-1     MON_3       March
en_US.ISO-8859-1     MON_4       April
en_US.ISO-8859-1     MON_5       May
en_US.ISO-8859-1     MON_6       June
en_US.ISO-8859-1     MON_7       July
en_US.ISO-8859-1     MON_8       August
en_US.ISO-8859-1     MON_9       September
en_US.ISO-8859-1     MON_10      October
en_US.ISO-8859-1     MON_11      November
en_US.ISO-8859-1     MON_12      December
en_US.ISO-8859-1     AM_STR      AM
en_US.ISO-8859-1     PM_STR      PM
en_US.ISO-8859-1     D_T_FMT     "%a %d %b %Y %r %Z"
en_US.ISO-8859-1     D_FMT       "%m/%d/%Y"
en_US.ISO-8859-1     T_FMT       "%r"
en_US.ISO-8859-1     T_FMT_AMPM  "%I:%M:%S %p"
en_US.ISO-8859-1     RADIXCHAR   .
en_US.ISO-8859-1     THOUSEP     ,
en_US.ISO-8859-1     YESEXPR     ^[+1yY]
en_US.ISO-8859-1     NOEXPR      ^[-0nN]
en_US.UTF-8	     CURRENCY_SYMBOL	$
de_DE.ISO-8859-1     ABDAY_1     So
de_DE.ISO-8859-1     ABDAY_2     Mo
de_DE.ISO-8859-1     ABDAY_3     Di
de_DE.ISO-8859-1     ABDAY_4     Mi
de_DE.ISO-8859-1     ABDAY_5     Do
de_DE.ISO-8859-1     ABDAY_6     Fr
de_DE.ISO-8859-1     ABDAY_7     Sa
de_DE.ISO-8859-1     DAY_1       Sonntag
de_DE.ISO-8859-1     DAY_2       Montag
de_DE.ISO-8859-1     DAY_3       Dienstag
de_DE.ISO-8859-1     DAY_4       Mittwoch
de_DE.ISO-8859-1     DAY_5       Donnerstag
de_DE.ISO-8859-1     DAY_6       Freitag
de_DE.ISO-8859-1     DAY_7       Samstag
de_DE.ISO-8859-1     ABMON_1     Jan
de_DE.ISO-8859-1     ABMON_2     Feb
de_DE.ISO-8859-1     ABMON_3     Mär
de_DE.ISO-8859-1     ABMON_4     Apr
de_DE.ISO-8859-1     ABMON_5     Mai
de_DE.ISO-8859-1     ABMON_6     Jun
de_DE.ISO-8859-1     ABMON_7     Jul
de_DE.ISO-8859-1     ABMON_8     Aug
de_DE.ISO-8859-1     ABMON_9     Sep
de_DE.ISO-8859-1     ABMON_10    Okt
de_DE.ISO-8859-1     ABMON_11    Nov
de_DE.ISO-8859-1     ABMON_12    Dez
de_DE.ISO-8859-1     MON_1       Januar
de_DE.ISO-8859-1     MON_2       Februar
de_DE.ISO-8859-1     MON_3       März
de_DE.ISO-8859-1     MON_4       April
de_DE.ISO-8859-1     MON_5       Mai
de_DE.ISO-8859-1     MON_6       Juni
de_DE.ISO-8859-1     MON_7       Juli
de_DE.ISO-8859-1     MON_8       August
de_DE.ISO-8859-1     MON_9       September
de_DE.ISO-8859-1     MON_10      Oktober
de_DE.ISO-8859-1     MON_11      November
de_DE.ISO-8859-1     MON_12      Dezember
de_DE.ISO-8859-1     D_T_FMT     "%a %d %b %Y %T %Z"
de_DE.ISO-8859-1     D_FMT       "%d.%m.%Y"
de_DE.ISO-8859-1     T_FMT       "%T"
de_DE.ISO-8859-1     RADIXCHAR   ,
de_DE.ISO-8859-1     THOUSEP     .
de_DE.ISO-8859-1     YESEXPR     ^[+1jJyY]
de_DE.ISO-8859-1     NOEXPR      ^[-0nN]
de_DE.UTF-8          ABDAY_1     So
de_DE.UTF-8          ABDAY_2     Mo
de_DE.UTF-8          ABDAY_3     Di
de_DE.UTF-8          ABDAY_4     Mi
de_DE.UTF-8          ABDAY_5     Do
de_DE.UTF-8          ABDAY_6     Fr
de_DE.UTF-8          ABDAY_7     Sa
de_DE.UTF-8          DAY_1       Sonntag
de_DE.UTF-8          DAY_2       Montag
de_DE.UTF-8          DAY_3       Dienstag
de_DE.UTF-8          DAY_4       Mittwoch
de_DE.UTF-8          DAY_5       Donnerstag
de_DE.UTF-8          DAY_6       Freitag
de_DE.UTF-8          DAY_7       Samstag
de_DE.UTF-8          ABMON_1     Jan
de_DE.UTF-8          ABMON_2     Feb
de_DE.UTF-8          ABMON_3     MÃ¤r
de_DE.UTF-8          ABMON_4     Apr
de_DE.UTF-8          ABMON_5     Mai
de_DE.UTF-8          ABMON_6     Jun
de_DE.UTF-8          ABMON_7     Jul
de_DE.UTF-8          ABMON_8     Aug
de_DE.UTF-8          ABMON_9     Sep
de_DE.UTF-8          ABMON_10    Okt
de_DE.UTF-8          ABMON_11    Nov
de_DE.UTF-8          ABMON_12    Dez
de_DE.UTF-8          MON_1       Januar
de_DE.UTF-8          MON_2       Februar
de_DE.UTF-8          MON_3       MÃ¤rz
de_DE.UTF-8          MON_4       April
de_DE.UTF-8          MON_5       Mai
de_DE.UTF-8          MON_6       Juni
de_DE.UTF-8          MON_7       Juli
de_DE.UTF-8          MON_8       August
de_DE.UTF-8          MON_9       September
de_DE.UTF-8          MON_10      Oktober
de_DE.UTF-8          MON_11      November
de_DE.UTF-8          MON_12      Dezember
de_DE.UTF-8          D_T_FMT     "%a %d %b %Y %T %Z"
de_DE.UTF-8          D_FMT       "%d.%m.%Y"
de_DE.UTF-8          T_FMT       "%T"
de_DE.UTF-8          RADIXCHAR   ,
de_DE.UTF-8          THOUSEP     .
de_DE.UTF-8          YESEXPR     ^[+1jJyY]
de_DE.UTF-8          NOEXPR      ^[-0nN]
de_DE.UTF-8          CURRENCY_SYMBOL    â‚¬
fr_FR.ISO-8859-1     ABDAY_1     dim.
fr_FR.ISO-8859-1     ABDAY_2     lun.
fr_FR.ISO-8859-1     ABDAY_3     mar.
fr_FR.ISO-8859-1     ABDAY_4     mer.
fr_FR.ISO-8859-1     ABDAY_5     jeu.
fr_FR.ISO-8859-1     ABDAY_6     ven.
fr_FR.ISO-8859-1     ABDAY_7     sam.
fr_FR.ISO-8859-1     DAY_1       dimanche
fr_FR.ISO-8859-1     DAY_2       lundi
fr_FR.ISO-8859-1     DAY_3       mardi
fr_FR.ISO-8859-1     DAY_4       mercredi
fr_FR.ISO-8859-1     DAY_5       jeudi
fr_FR.ISO-8859-1     DAY_6       vendredi
fr_FR.ISO-8859-1     DAY_7       samedi
fr_FR.ISO-8859-1     ABMON_1     janv.
fr_FR.ISO-8859-1     ABMON_2     févr.
fr_FR.ISO-8859-1     ABMON_3     mars
fr_FR.ISO-8859-1     ABMON_4     avril
fr_FR.ISO-8859-1     ABMON_5     mai
fr_FR.ISO-8859-1     ABMON_6     juin
fr_FR.ISO-8859-1     ABMON_7     juil.
fr_FR.ISO-8859-1     ABMON_8     août
fr_FR.ISO-8859-1     ABMON_9     sept.
fr_FR.ISO-8859-1     ABMON_10    oct.
fr_FR.ISO-8859-1     ABMON_11    nov.
fr_FR.ISO-8859-1     ABMON_12    déc.
fr_FR.ISO-8859-1     MON_1       janvier
fr_FR.ISO-8859-1     MON_2       février
fr_FR.ISO-8859-1     MON_3       mars
fr_FR.ISO-8859-1     MON_4       avril
fr_FR.ISO-8859-1     MON_5       mai
fr_FR.ISO-8859-1     MON_6       juin
fr_FR.ISO-8859-1     MON_7       juillet
fr_FR.ISO-8859-1     MON_8       août
fr_FR.ISO-8859-1     MON_9       septembre
fr_FR.ISO-8859-1     MON_10      octobre
fr_FR.ISO-8859-1     MON_11      novembre
fr_FR.ISO-8859-1     MON_12      décembre
fr_FR.ISO-8859-1     D_T_FMT     "%a %d %b %Y %T %Z"
fr_FR.ISO-8859-1     D_FMT       "%d/%m/%Y"
fr_FR.ISO-8859-1     T_FMT       "%T"
fr_FR.ISO-8859-1     RADIXCHAR   ,
fr_FR.ISO-8859-1     THOUSEP     " "
fr_FR.ISO-8859-1     YESEXPR     ^[+1oOyY]
fr_FR.ISO-8859-1     NOEXPR      ^[-0nN]
fr_FR.UTF-8          CURRENCY_SYMBOL    â‚¬
ja_JP.EUC-JP         ABDAY_1     Æü
ja_JP.EUC-JP         ABDAY_2     ·î
ja_JP.EUC-JP         ABDAY_3     ²Ð
ja_JP.EUC-JP         ABDAY_4     ¿å
ja_JP.EUC-JP         ABDAY_5     ÌÚ
ja_JP.EUC-JP         ABDAY_6     ¶â
ja_JP.EUC-JP         ABDAY_7     ÅÚ
ja_JP.EUC-JP         DAY_1       ÆüÍËÆü
ja_JP.EUC-JP         DAY_2       ·îÍËÆü
ja_JP.EUC-JP         DAY_3       ²ÐÍËÆü
ja_JP.EUC-JP         DAY_4       ¿åÍËÆü
ja_JP.EUC-JP         DAY_5       ÌÚÍËÆü
ja_JP.EUC-JP         DAY_6       ¶âÍËÆü
ja_JP.EUC-JP         DAY_7       ÅÚÍËÆü
ja_JP.EUC-JP         ABMON_1     " 1·î"
ja_JP.EUC-JP         ABMON_2     " 2·î"
ja_JP.EUC-JP         ABMON_3     " 3·î"
ja_JP.EUC-JP         ABMON_4     " 4·î"
ja_JP.EUC-JP         ABMON_5     " 5·î"
ja_JP.EUC-JP         ABMON_6     " 6·î"
ja_JP.EUC-JP         ABMON_7     " 7·î"
ja_JP.EUC-JP         ABMON_8     " 8·î"
ja_JP.EUC-JP         ABMON_9     " 9·î"
ja_JP.EUC-JP         ABMON_10    "10·î"
ja_JP.EUC-JP         ABMON_11    "11·î"
ja_JP.EUC-JP         ABMON_12    "12·î"
ja_JP.EUC-JP         MON_1       "1·î"
ja_JP.EUC-JP         MON_2       "2·î"
ja_JP.EUC-JP         MON_3       "3·î"
ja_JP.EUC-JP         MON_4       "4·î"
ja_JP.EUC-JP         MON_5       "5·î"
ja_JP.EUC-JP         MON_6       "6·î"
ja_JP.EUC-JP         MON_7       "7·î"
ja_JP.EUC-JP         MON_8       "8·î"
ja_JP.EUC-JP         MON_9       "9·î"
ja_JP.EUC-JP         MON_10      "10·î"
ja_JP.EUC-JP         MON_11      "11·î"
ja_JP.EUC-JP         MON_12      "12·î"
ja_JP.EUC-JP         T_FMT_AMPM  "%p%I»þ%MÊ¬%SÉÃ"
ja_JP.EUC-JP         ERA_D_FMT   "%EY%m·î%dÆü"
ja_JP.EUC-JP         ERA_D_T_FMT "%EY%m·î%dÆü %H»þ%MÊ¬%SÉÃ"
ja_JP.EUC-JP         RADIXCHAR   .
ja_JP.EUC-JP         THOUSEP     ,
ja_JP.EUC-JP         YESEXPR     ^([+1yY£ù£Ù]|¤Ï¤¤|¥Ï¥¤)
ja_JP.EUC-JP         NOEXPR      ^([-0nN£î£Î]|¤¤¤¤¤¨|¥¤¥¤¥¨)
# Is CRNCYSTR supposed to be the national or international sign?
# ja_JP.EUC-JP         CRNCYSTR    JPY
ja_JP.EUC-JP         CODESET     EUC-JP
ja_JP.UTF-8          CURRENCY_SYMBOL    ï¿¥
EOF
${tst_langinfo_before_env} \
${run_program_env} \
LC_ALL=tt_TT ${tst_langinfo_after_env}

exit $?
