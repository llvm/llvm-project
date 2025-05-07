// RUN: %clang_cc1 %s -emit-llvm -triple s390x-none-zos -fexec-charset IBM-1047 -o - | FileCheck %s
// RUN: %clang %s -emit-llvm -S -target s390x-ibm-zos -o - | FileCheck %s

const char *UpperCaseLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
// CHECK: c"\C1\C2\C3\C4\C5\C6\C7\C8\C9\D1\D2\D3\D4\D5\D6\D7\D8\D9\E2\E3\E4\E5\E6\E7\E8\E9\00"

const char *LowerCaseLetters = "abcdefghijklmnopqrstuvwxyz";
//CHECK: c"\81\82\83\84\85\86\87\88\89\91\92\93\94\95\96\97\98\99\A2\A3\A4\A5\A6\A7\A8\A9\00"

const char *Digits = "0123456789";
// CHECK: c"\F0\F1\F2\F3\F4\F5\F6\F7\F8\F9\00"

const char *SpecialCharacters = " .<(+|&!$*);^-/,%%_>`:#@=";
// CHECK: c"@KLMNOPZ[\\]^_`akllmnyz{|~\00"

const char *EscapeCharacters = "\a\b\f\n\r\t\v\\\'\"\?";
//CHECK: c"/\16\0C\15\0D\05\0B\E0}\7Fo\00"

const char *InvalidEscape = "\y\z";
//CHECK: c"oo\00"

const char *HexCharacters = "\x12\x13\x14";
//CHECK: c"\12\13\14\00"

const char *OctalCharacters = "\141\142\143";
//CHECK: c"abc\00"

const char singleChar = 'a';
//CHECK: i8 -127

const char *UcnCharacters = "\u00E2\u00AC\U000000DF";
//CHECK: c"B\B0Y\00"

const char *Unicode = "ÿ";
//CHECK: c"\DF\00"
