// RUN: %clang %s -std=c++17 -emit-llvm -S -target s390x-ibm-zos -o - | FileCheck %s

const char *RawString = R"(Hello\n)";
//CHECK: c"\C8\85\93\93\96\E0\95\00"

const char *MultiLineRawString = R"(
Hello
There)";
//CHECK: c"\15\C8\85\93\93\96\15\E3\88\85\99\85\00"

char UnicodeChar8 = u8'1';
//CHECK: i8 49
char16_t UnicodeChar16 = u'1';
//CHECK: i16 49
char32_t UnicodeChar32 = U'1';
//CHECK: i32 49

const char *EscapeCharacters8 = u8"\a\b\f\n\r\t\v\\\'\"\?";
//CHECK: c"\07\08\0C\0A\0D\09\0B\\'\22?\00"

const char16_t *EscapeCharacters16 = u"\a\b\f\n\r\t\v\\\'\"\?";
//CHECK: [12 x i16] [i16 7, i16 8, i16 12, i16 10, i16 13, i16 9, i16 11, i16 92, i16 39, i16 34, i16 63, i16 0]

const char32_t *EscapeCharacters32 = U"\a\b\f\n\r\t\v\\\'\"\?";
//CHECK: [12 x i32] [i32 7, i32 8, i32 12, i32 10, i32 13, i32 9, i32 11, i32 92, i32 39, i32 34, i32 63, i32 0]

const char *UnicodeString8 = u8"Hello";
//CHECK: c"Hello\00"
const char16_t *UnicodeString16 = u"Hello";
//CHECK: [6 x i16] [i16 72, i16 101, i16 108, i16 108, i16 111, i16 0]
const char32_t *UnicodeString32 = U"Hello";
//CHECK: [6 x i32] [i32 72, i32 101, i32 108, i32 108, i32 111, i32 0]

const char *UnicodeRawString8 = u8R"("Hello\")";
//CHECK: c"\22Hello\\\22\00"
const char16_t *UnicodeRawString16 = uR"("Hello\")";
//CHECK: [9 x i16] [i16 34, i16 72, i16 101, i16 108, i16 108, i16 111, i16 92, i16 34, i16 0]
const char32_t *UnicodeRawString32 = UR"("Hello\")";
//CHECK: [9 x i32] [i32 34, i32 72, i32 101, i32 108, i32 108, i32 111, i32 92, i32 34, i32 0]

const char *UnicodeUCNString8 = u8"\u00E2\u00AC\U000000DF";
//CHECK: c"\C3\A2\C2\AC\C3\9F\00"
const char16_t *UnicodeUCNString16 = u"\u00E2\u00AC\U000000DF";
//CHECK: [4 x i16] [i16 226, i16 172, i16 223, i16 0]
const char32_t *UnicodeUCNString32 = U"\u00E2\u00AC\U000000DF";
//CHECK: [4 x i32] [i32 226, i32 172, i32 223, i32 0]
