#!/bin/bash
# Test for some known iconv(1) hangs from bug 19519, and miscellaneous
# iconv(1) program error conditions.
# Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

codir=$1
test_wrapper_env="$2"
run_program_env="$3"

# We have to have some directories in the library path.
LIBPATH=$codir:$codir/iconvdata

# How the start the iconv(1) program.  $from is not defined/expanded yet.
ICONV='
$codir/elf/ld.so --library-path $LIBPATH --inhibit-rpath ${from}.so
$codir/iconv/iconv_prog
'
ICONV="$test_wrapper_env $run_program_env $ICONV"

# List of known hangs;
# Gathered by running an exhaustive 2 byte input search against glibc-2.28
hangarray=(
"\x00\x23;-c;ANSI_X3.110;UTF-8//TRANSLIT//IGNORE"
"\x00\xa1;-c;ARMSCII-8;UTF-8//TRANSLIT//IGNORE"
"\x00\xa1;-c;ASMO_449;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;BIG5;UTF-8//TRANSLIT//IGNORE"
"\x00\xff;-c;BIG5HKSCS;UTF-8//TRANSLIT//IGNORE"
"\x00\xff;-c;BRF;UTF-8//TRANSLIT//IGNORE"
"\x00\xff;-c;BS_4730;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;CP1250;UTF-8//TRANSLIT//IGNORE"
"\x00\x98;-c;CP1251;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;CP1252;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;CP1253;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;CP1254;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;CP1255;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;CP1257;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;CP1258;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;CP932;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;CSA_Z243.4-1985-1;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;CSA_Z243.4-1985-2;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;DEC-MCS;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;DIN_66003;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;DS_2089;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-AT-DE;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-AT-DE-A;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-CA-FR;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-DK-NO;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-DK-NO-A;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-ES;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-ES-A;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-ES-S;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-FI-SE;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-FI-SE-A;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-FR;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-IS-FRISS;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-IT;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-PT;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-UK;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;EBCDIC-US;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ES;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ES2;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;EUC-CN;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;EUC-JISX0213;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;EUC-JP;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;EUC-JP-MS;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;EUC-KR;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;EUC-TW;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;GB18030;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;GB_1988-80;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;GBK;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;GOST_19768-74;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;GREEK7;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;GREEK7-OLD;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;GREEK-CCITT;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;HP-GREEK8;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;HP-ROMAN8;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;HP-ROMAN9;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;HP-THAI8;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;HP-TURKISH8;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM038;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;IBM1004;UTF-8//TRANSLIT//IGNORE"
"\x00\xff;-c;IBM1008;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;IBM1046;UTF-8//TRANSLIT//IGNORE"
"\x00\x51;-c;IBM1132;UTF-8//TRANSLIT//IGNORE"
"\x00\xa0;-c;IBM1133;UTF-8//TRANSLIT//IGNORE"
"\x00\xce;-c;IBM1137;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;IBM1161;UTF-8//TRANSLIT//IGNORE"
"\x00\xdb;-c;IBM1162;UTF-8//TRANSLIT//IGNORE"
"\x00\x70;-c;IBM12712;UTF-8//TRANSLIT//IGNORE"
"\x00\x0f;-c;IBM1364;UTF-8"
"\x0e\x0e;-c;IBM1364;UTF-8"
"\x00\x0f;-c;IBM1371;UTF-8"
"\x0e\x0e;-c;IBM1371;UTF-8"
"\x00\x0f;-c;IBM1388;UTF-8"
"\x0e\x0e;-c;IBM1388;UTF-8"
"\x00\x0f;-c;IBM1390;UTF-8"
"\x0e\x0e;-c;IBM1390;UTF-8"
"\x00\x0f;-c;IBM1399;UTF-8"
"\x0e\x0e;-c;IBM1399;UTF-8"
"\x00\x53;-c;IBM16804;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM274;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM275;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM281;UTF-8//TRANSLIT//IGNORE"
"\x00\x57;-c;IBM290;UTF-8//TRANSLIT//IGNORE"
"\x00\x45;-c;IBM420;UTF-8//TRANSLIT//IGNORE"
"\x00\x68;-c;IBM423;UTF-8//TRANSLIT//IGNORE"
"\x00\x70;-c;IBM424;UTF-8//TRANSLIT//IGNORE"
"\x00\x53;-c;IBM4517;UTF-8//TRANSLIT//IGNORE"
"\x00\x53;-c;IBM4899;UTF-8//TRANSLIT//IGNORE"
"\x00\xa5;-c;IBM4909;UTF-8//TRANSLIT//IGNORE"
"\x00\xdc;-c;IBM4971;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM803;UTF-8//TRANSLIT//IGNORE"
"\x00\x91;-c;IBM851;UTF-8//TRANSLIT//IGNORE"
"\x00\x9b;-c;IBM856;UTF-8//TRANSLIT//IGNORE"
"\x00\xd5;-c;IBM857;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;IBM864;UTF-8//TRANSLIT//IGNORE"
"\x00\x94;-c;IBM868;UTF-8//TRANSLIT//IGNORE"
"\x00\x94;-c;IBM869;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;IBM874;UTF-8//TRANSLIT//IGNORE"
"\x00\x6a;-c;IBM875;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM880;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;IBM891;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;IBM903;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;IBM904;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM905;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;IBM9066;UTF-8//TRANSLIT//IGNORE"
"\x00\x48;-c;IBM918;UTF-8//TRANSLIT//IGNORE"
"\x00\x57;-c;IBM930;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;IBM932;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM933;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM935;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM937;UTF-8//TRANSLIT//IGNORE"
"\x00\x41;-c;IBM939;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;IBM943;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;INIS;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;INIS-8;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;INIS-CYRILLIC;UTF-8//TRANSLIT//IGNORE"
"\x00\xec;-c;ISIRI-3342;UTF-8//TRANSLIT//IGNORE"
"\x00\xec;-c;ISO_10367-BOX;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-2022-CN;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-2022-CN-EXT;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-2022-JP;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-2022-JP-2;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-2022-JP-3;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-2022-KR;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO_2033;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO_5427;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO_5427-EXT;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO_5428;UTF-8//TRANSLIT//IGNORE"
"\x00\xa4;-c;ISO_6937;UTF-8//TRANSLIT//IGNORE"
"\x00\xa0;-c;ISO_6937-2;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-8859-11;UTF-8//TRANSLIT//IGNORE"
"\x00\xa5;-c;ISO-8859-3;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-8859-6;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-8859-7;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;ISO-8859-8;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;ISO-IR-197;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;ISO-IR-209;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;IT;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;JIS_C6220-1969-RO;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;JIS_C6229-1984-B;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;JOHAB;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;JUS_I.B1.002;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;KOI-8;UTF-8//TRANSLIT//IGNORE"
"\x00\x88;-c;KOI8-T;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;KSC5636;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;LATIN-GREEK;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;LATIN-GREEK-1;UTF-8//TRANSLIT//IGNORE"
"\x00\xf6;-c;MAC-IS;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;MSZ_7795.3;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;NATS-DANO;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;NATS-SEFI;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;NC_NC00-10;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;NF_Z_62-010;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;NF_Z_62-010_1973;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;NS_4551-1;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;NS_4551-2;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;PT;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;PT2;UTF-8//TRANSLIT//IGNORE"
"\x00\x98;-c;RK1048;UTF-8//TRANSLIT//IGNORE"
"\x00\x98;-c;SEN_850200_B;UTF-8//TRANSLIT//IGNORE"
"\x00\x98;-c;SEN_850200_C;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;Shift_JISX0213;UTF-8//TRANSLIT//IGNORE"
"\x00\x80;-c;SJIS;UTF-8//TRANSLIT//IGNORE"
"\x00\x23;-c;T.61-8BIT;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;TIS-620;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;TSCII;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;UHC;UTF-8//TRANSLIT//IGNORE"
"\x00\xd8;-c;UNICODE;UTF-8//TRANSLIT//IGNORE"
"\x00\xdc;-c;UTF-16;UTF-8//TRANSLIT//IGNORE"
"\xdc\x00;-c;UTF-16BE;UTF-8//TRANSLIT//IGNORE"
"\x00\xdc;-c;UTF-16LE;UTF-8//TRANSLIT//IGNORE"
"\xff\xff;-c;UTF-7;UTF-8//TRANSLIT//IGNORE"
"\x00\x81;-c;WIN-SAMI-2;UTF-8//TRANSLIT//IGNORE"
)

# List of option combinations that *should* lead to an error
errorarray=(
# Converting from/to invalid character sets should cause error
"\x00\x00;;INVALID;INVALID"
"\x00\x00;;INVALID;UTF-8"
"\x00\x00;;UTF-8;INVALID"
)

# Requires $twobyte input, $c flag, $from, and $to to be set; sets $ret
execute_test ()
{
  eval PROG=\"$ICONV\"
  echo -en "$twobyte" \
    | timeout -k 4 3 $PROG $c -f $from -t "$to" &>/dev/null
  ret=$?
}

check_hangtest_result ()
{
  if [ "$ret" -eq "124" ] || [ "$ret" -eq "137" ]; then # timeout/hang
    result="HANG"
  else
    if [ "$ret" -eq "139" ]; then # segfault
      result="SEGFAULT"
    else
      if [ "$ret" -gt "127" ]; then # unexpected error
        result="UNEXPECTED"
      else
        result="OK"
      fi
    fi
  fi

  echo -n "$result: from: \"$from\", to: \"$to\","
  echo    " input \"$twobyte\", flags \"$c\""

  if [ "$result" != "OK" ]; then
    exit 1
  fi
}

for hangcommand in "${hangarray[@]}"; do
  twobyte="$(echo "$hangcommand" | cut -d";" -f 1)"
  c="$(echo "$hangcommand" | cut -d";" -f 2)"
  from="$(echo "$hangcommand" | cut -d";" -f 3)"
  to="$(echo "$hangcommand" | cut -d";" -f 4)"
  execute_test
  check_hangtest_result
done

check_errtest_result ()
{
  if [ "$ret" -eq "1" ]; then # we errored out as expected
    result="PASS"
  else
    result="FAIL"
  fi
  echo -n "$result: from: \"$from\", to: \"$to\","
  echo    " input \"$twobyte\", flags \"$c\", return code $ret"

  if [ "$result" != "PASS" ]; then
    exit 1
  fi
}

for errorcommand in "${errorarray[@]}"; do
  twobyte="$(echo "$errorcommand" | cut -d";" -f 1)"
  c="$(echo "$errorcommand" | cut -d";" -f 2)"
  from="$(echo "$errorcommand" | cut -d";" -f 3)"
  to="$(echo "$errorcommand" | cut -d";" -f 4)"
  execute_test
  check_errtest_result
done
