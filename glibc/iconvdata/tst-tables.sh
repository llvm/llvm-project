#!/bin/sh
# Copyright (C) 2000-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by Bruno Haible <haible@clisp.cons.org>, 2000.
#

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

# Checks that the iconv() implementation (in both directions) for the
# stateless encodings agrees with the corresponding charmap table.

common_objpfx=$1
objpfx=$2
test_program_prefix=$3

status=0

cat <<EOF |
  # Single-byte and other "small" encodings come here.
  # Keep this list in the same order as gconv-modules.
  #
  # charset name    table name          comment
  ASCII             ANSI_X3.4-1968
  ISO646-GB         BS_4730
  ISO646-CA         CSA_Z243.4-1985-1
  ISO646-CA2        CSA_Z243.4-1985-2
  ISO646-DE         DIN_66003
  ISO646-DK         DS_2089
  ISO646-ES         ES
  ISO646-ES2        ES2
  ISO646-CN         GB_1988-80
  ISO646-IT         IT
  ISO646-JP         JIS_C6220-1969-RO
  ISO646-JP-OCR-B   JIS_C6229-1984-B
  ISO646-YU         JUS_I.B1.002
  ISO646-KR         KSC5636
  ISO646-HU         MSZ_7795.3
  ISO646-CU         NC_NC00-10
  ISO646-FR         NF_Z_62-010
  ISO646-FR1        NF_Z_62-010_1973
  ISO646-NO         NS_4551-1
  ISO646-NO2        NS_4551-2
  ISO646-PT         PT
  ISO646-PT2        PT2
  ISO646-SE         SEN_850200_B
  ISO646-SE2        SEN_850200_C
  ISO-8859-1
  ISO-8859-2
  ISO-8859-3
  ISO-8859-4
  ISO-8859-5
  ISO-8859-6
  ISO-8859-7
  ISO-8859-8
  ISO-8859-9
  ISO-8859-9E
  ISO-8859-10
  ISO-8859-11
  ISO-8859-13
  ISO-8859-14
  ISO-8859-15
  ISO-8859-16
  T.61-8BIT
  ISO_6937
  #ISO_6937-2        ISO-IR-90          Handling of combining marks is broken
  KOI-8
  KOI8-R
  LATIN-GREEK
  LATIN-GREEK-1
  HP-ROMAN8
  HP-ROMAN9
  HP-TURKISH8
  HP-THAI8
  HP-GREEK8
  EBCDIC-AT-DE
  EBCDIC-AT-DE-A
  EBCDIC-CA-FR
  EBCDIC-DK-NO
  EBCDIC-DK-NO-A
  EBCDIC-ES
  EBCDIC-ES-A
  EBCDIC-ES-S
  EBCDIC-FI-SE
  EBCDIC-FI-SE-A
  EBCDIC-FR
  EBCDIC-IS-FRISS
  EBCDIC-IT
  EBCDIC-PT
  EBCDIC-UK
  EBCDIC-US
  IBM037
  IBM038
  IBM256
  IBM273
  IBM274
  IBM275
  IBM277
  IBM278
  IBM280
  IBM281
  IBM284
  IBM285
  IBM290
  IBM297
  IBM420
  IBM423
  IBM424
  IBM437
  IBM500
  IBM850
  IBM851
  IBM852
  IBM855
  IBM856
  IBM857
  IBM858
  IBM860
  IBM861
  IBM862
  IBM863
  IBM864
  IBM865
  IBM866
  IBM866NAV
  IBM868
  IBM869
  IBM870
  IBM871
  IBM875
  IBM880
  IBM891
  IBM903
  IBM904
  IBM905
  IBM918
  IBM922
  IBM1004
  IBM1026
  #IBM1046                              Differs from the AIX and JDK converters
  IBM1047
  IBM1124
  IBM1129
  IBM1160
  IBM1161
  IBM1132
  IBM1133
  IBM1162
  IBM1163
  IBM1164
  CP1125
  CP1250
  CP1251
  CP1252
  CP1253
  CP1254
  CP1255
  CP1256
  CP1257
  CP1258
  IBM874
  CP737
  CP770
  CP771
  CP772
  CP773
  CP774
  CP775
  MACINTOSH
  IEC_P27-1
  ASMO_449
  ISO-IR-99         ANSI_X3.110-1983
  ISO-IR-139        CSN_369103
  CWI
  DEC-MCS
  ECMA-CYRILLIC
  ISO-IR-153        GOST_19768-74
  GREEK-CCITT
  GREEK7
  GREEK7-OLD
  INIS
  INIS-8
  INIS-CYRILLIC
  ISO_2033          ISO_2033-1983
  ISO_5427
  ISO_5427-EXT
  #ISO_5428                             Handling of combining marks is broken
  ISO_10367-BOX
  MAC-IS
  MAC-UK
  CP10007
  NATS-DANO
  NATS-SEFI
  WIN-SAMI-2        SAMI-WS2
  ISO-IR-197
  TIS-620
  KOI8-U
  #ISIRI-3342                         This charset concept is completely broken
  VISCII
  KOI8-T
  GEORGIAN-PS
  GEORGIAN-ACADEMY
  ISO-IR-209
  MAC-SAMI
  ARMSCII-8
  TCVN5712-1
  TSCII
  PT154
  RK1048
  MIK
  BRF
  MAC-CENTRALEUROPE
  KOI8-RU
  #
  # Multibyte encodings come here
  #
  SJIS              SHIFT_JIS
  CP932             WINDOWS-31J
  #IBM932                               This converter looks quite strange
  #IBM943                               This converter looks quite strange
  EUC-KR
  CP949
  JOHAB
  BIG5
  BIG5HKSCS         BIG5-HKSCS
  EUC-JP
  EUC-JP-MS
  EUC-CN            GB2312
  GBK
  EUC-TW
  GB18030
  EUC-JISX0213
  SHIFT_JISX0213
  #
  # Stateful encodings not testable this way
  #
  #IBM930
  #IBM933
  #IBM935
  #IBM937
  #IBM939
  #ISO-2022-JP
  #ISO-2022-JP-2
  #ISO-2022-JP-3
  #ISO-2022-KR
  #ISO-2022-CN
  #ISO-2022-CN-EXT
  #UTF-7
  #
EOF
while read charset charmap; do
  if test "$charset" = GB18030; then echo "This might take a while" 1>&2; fi
  case ${charset} in \#*) continue;; esac
  printf %s "Testing ${charset}" 1>&2
  if ./tst-table.sh ${common_objpfx} ${objpfx} "${test_program_prefix}" \
      ${charset} ${charmap} < /dev/null; then
    echo 1>&2
  else
    echo "failed: ./tst-table.sh ${common_objpfx} ${objpfx} ${charset} ${charmap}"
    echo " *** FAILED ***" 1>&2
    exit 1
  fi
done

exit $?
