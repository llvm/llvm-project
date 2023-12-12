// RUN: %check_clang_tidy %s modernize-use-digit-separator %t

int NotFormattedInteger = 1234567;
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: unformatted representation of integer literal '1234567' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234'567

int MinusNotFormattedInteger = -1234567;
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: unformatted representation of integer literal '1234567' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234'567

int BinaryNotFormattedInteger = 0b11101101;
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: unformatted representation of integer literal '0b11101101' [modernize-use-digit-separator]
// CHECK-FIXES: 0b1110'1101

int MinusBinaryNotFormattedInteger = -0b11101101;
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: unformatted representation of integer literal '0b11101101' [modernize-use-digit-separator]
// CHECK-FIXES: 0b1110'1101

int OctNotFormattedInteger = 037512;
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: unformatted representation of integer literal '037512' [modernize-use-digit-separator]
// CHECK-FIXES: 037'512

int MinusOctNotFormattedInteger = -037512;
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: unformatted representation of integer literal '037512' [modernize-use-digit-separator]
// CHECK-FIXES: 037'512

int HexNotFormattedInteger = 0x4f356;
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: unformatted representation of integer literal '0x4f356' [modernize-use-digit-separator]
// CHECK-FIXES: 0x4'F356

int MinusHexNotFormattedInteger = -0x4f356;
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: unformatted representation of integer literal '0x4f356' [modernize-use-digit-separator]
// CHECK-FIXES: 0x4'F356

unsigned int UnsignedNotFormattedInteger = 10004U;
// CHECK-MESSAGES: :[[@LINE-1]]:44: warning: unformatted representation of integer literal '10004U' [modernize-use-digit-separator]
// CHECK-FIXES: 10'004

unsigned int MinusUnsignedNotFormattedInteger = -10004U;
// CHECK-MESSAGES: :[[@LINE-1]]:50: warning: unformatted representation of integer literal '10004U' [modernize-use-digit-separator]
// CHECK-FIXES: 10'004

unsigned int UnsignedNotFormattedInteger1 = 100045u;
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: unformatted representation of integer literal '100045u' [modernize-use-digit-separator]
// CHECK-FIXES: 100'045

unsigned int MinusUnsignedNotFormattedInteger1 = -100045u;
// CHECK-MESSAGES: :[[@LINE-1]]:51: warning: unformatted representation of integer literal '100045u' [modernize-use-digit-separator]
// CHECK-FIXES: 100'045

long LongNotFormattedInteger = 123456789101112L;
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: unformatted representation of integer literal '123456789101112L' [modernize-use-digit-separator]
// CHECK-FIXES: 123'456'789'101'112

long MinusLongNotFormattedInteger = -123456789101112L;
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: unformatted representation of integer literal '123456789101112L' [modernize-use-digit-separator]
// CHECK-FIXES: 123'456'789'101'112

long LongNotFormattedInteger1 = 12345678910111213l;
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: unformatted representation of integer literal '12345678910111213l' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213

long MinusLongNotFormattedInteger1 = -12345678910111213l;
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: unformatted representation of integer literal '12345678910111213l' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213

float NotFormattedFloat = 1234.56789;
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: unformatted representation of integer literal '1234.56789' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.567'89

float MinusNotFormattedFloat = -1234.56789;
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: unformatted representation of integer literal '1234.56789' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.567'89

// FIXME:
// error: expected ';' after top level declarator [clang-diagnostic-error]
//   80 | int FormattedInteger = 1'234'567;
//      |                         ^
//      |                         ;
//int FormattedInteger = 1'234'567;