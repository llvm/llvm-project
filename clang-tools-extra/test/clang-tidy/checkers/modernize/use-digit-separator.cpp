// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-use-digit-separator %t


// Long not formatted literals

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
// CHECK-FIXES: 10'004U

unsigned int MinusUnsignedNotFormattedInteger = -10004U;
// CHECK-MESSAGES: :[[@LINE-1]]:50: warning: unformatted representation of integer literal '10004U' [modernize-use-digit-separator]
// CHECK-FIXES: 10'004U

unsigned int UnsignedNotFormattedInteger1 = 100045u;
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: unformatted representation of integer literal '100045u' [modernize-use-digit-separator]
// CHECK-FIXES: 100'045u

unsigned int MinusUnsignedNotFormattedInteger1 = -100045u;
// CHECK-MESSAGES: :[[@LINE-1]]:51: warning: unformatted representation of integer literal '100045u' [modernize-use-digit-separator]
// CHECK-FIXES: 100'045u

long LongNotFormattedInteger = 123456789101112L;
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: unformatted representation of integer literal '123456789101112L' [modernize-use-digit-separator]
// CHECK-FIXES: 123'456'789'101'112L

long MinusLongNotFormattedInteger = -123456789101112L;
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: unformatted representation of integer literal '123456789101112L' [modernize-use-digit-separator]
// CHECK-FIXES: 123'456'789'101'112L

long LongNotFormattedInteger1 = 12345678910111213l;
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: unformatted representation of integer literal '12345678910111213l' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213l

long MinusLongNotFormattedInteger1 = -12345678910111213l;
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: unformatted representation of integer literal '12345678910111213l' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213l

unsigned long UnsignedLongNotFormattedInteger1 = 12345678910111213Ul;
// CHECK-MESSAGES: :[[@LINE-1]]:50: warning: unformatted representation of integer literal '12345678910111213Ul' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213Ul

unsigned long MinusUnsignedLongNotFormattedInteger1 = -12345678910111213Ul;
// CHECK-MESSAGES: :[[@LINE-1]]:56: warning: unformatted representation of integer literal '12345678910111213Ul' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213Ul

float NotFormattedFloat = 1234.56789;
// CHECK-MESSAGES: :[[@LINE-1]]:27: warning: unformatted representation of float literal '1234.56789' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.567'89

float MinusNotFormattedFloat = -1234.56789;
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: unformatted representation of float literal '1234.56789' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.567'89

double PostfixNotFormattedFloat = 1234.569F;
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: unformatted representation of float literal '1234.569F' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.569F

double MinusPostfixNotFormattedFloat = -1234.569F;
// CHECK-MESSAGES: :[[@LINE-1]]:41: warning: unformatted representation of float literal '1234.569F' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.569F

double PostfixNotFormattedFloat1 = 1234.569f;
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: unformatted representation of float literal '1234.569f' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.569f

double MinusPostfixNotFormattedFloat1 = -1234.569f;
// CHECK-MESSAGES: :[[@LINE-1]]:42: warning: unformatted representation of float literal '1234.569f' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.569f

double ScientificNotFormattedFloat = 1.2345678E10;
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: unformatted representation of float literal '1.2345678E10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E10

double MinusScientificNotFormattedFloat = -1.2345678E10;
// CHECK-MESSAGES: :[[@LINE-1]]:44: warning: unformatted representation of float literal '1.2345678E10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E10

double ScientificNotFormattedFloat1 = 1.2345678e10;
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: unformatted representation of float literal '1.2345678e10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e10

double MinusScientificNotFormattedFloat1 = -1.2345678e10;
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: unformatted representation of float literal '1.2345678e10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e10

double ScientificNotFormattedFloat2 = 1.2345678E+10;
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: unformatted representation of float literal '1.2345678E+10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E+10

double MinusScientificNotFormattedFloat2 = -1.2345678E+10;
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: unformatted representation of float literal '1.2345678E+10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E+10

double ScientificNotFormattedFloat3 = 1.2345678e+10;
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: unformatted representation of float literal '1.2345678e+10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e+10

double MinusScientificNotFormattedFloat3 = -1.2345678e+10;
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: unformatted representation of float literal '1.2345678e+10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e+10

double PostfixScientificNotFormattedFloat = 1.2345678E10F;
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: unformatted representation of float literal '1.2345678E10F' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E10F

double PostfixScientificNotFormattedFloat1 = 1.2345678e10f;
// CHECK-MESSAGES: :[[@LINE-1]]:46: warning: unformatted representation of float literal '1.2345678e10f' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e10f

double PostfixScientificNotFormattedFloat2 = -1.2345678E+10f;
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: unformatted representation of float literal '1.2345678E+10f' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E+10f

double PostfixScientificNotFormattedFloat4 = -1.2345678e-10F;
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: unformatted representation of float literal '1.2345678e-10F' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e-10F


// Short literals

int ShortInteger = 123;
int MinusShortInteger = -123;
int ShortBinaryInteger = 0b10;
int MinusShortBinaryInteger = 0b10;
int ShortOctInteger = 037;
int MinusShortOctInteger = -037;
int ShortHexInteger = 0x3F0;
int MinusShortHexInteger = -0x3F0;
unsigned int UnsignedShortInteger = 123U;
unsigned int MinusUnsignedShortInteger = -123U;
unsigned int UnsignedShortInteger1 = 123u;
unsigned int MinusUnsignedShortInteger1 = -123u;
long LongShortInteger = 123L;
long MinusLongShortInteger = -123L;
long LongShortInteger1 = 123l;
long MinusLongShortInteger1 = -123l;
unsigned long UnsignedLongShortInteger = 123uL;
unsigned long MinusUnsignedLongShortInteger = -123uL;
float ShortFloat = 1.23;
float MinusShortFloat = -1.23;
float PostfixShortFloat = 1.23F;
float MinusPostfixShortFloat = -1.23F;
float PostfixShortFloat1 = 1.23f;
float MinusPostfixShortFloat1 = -1.23f;
float ScientificShortFloat = 1.23E10;
float MinusScientificShortFloat = -1.23E10;
float ScientificShortFloat1 = 1.23e10;
float MinusScientificShortFloat1 = -1.23e10;
float ScientificShortFloat2 = 1.23E+10;
float MinusScientificShortFloat2 = -1.23E+10;
float ScientificShortFloat3 = 1.23e+10;
float MinusScientificShortFloat3 = -1.23e+10;
float ScientificShortFloat4 = 1.23E-10;
float MinusScientificShortFloat4 = -1.23E-10;
float ScientificShortFloat5 = 1.23e+10;
float MinusScientificShortFloat5 = -1.23e-10;
float PostfixScientificShortFloat = 1.23E10F;
float PostfixScientificShortFloat1 = 1.23e10f;
float PostfixScientificShortFloat2 = 1.23E+10f;
float PostfixScientificShortFloat3 = 1.23e-10F;


// Formatted literals

int FormattedInteger = 1'234'567;
int MinusFormattedInteger = -1'234'567;
int BinaryFormattedInteger = 0b1110'1101;
int MinusBinaryFormattedInteger = -0b1110'1101;
int OctFormattedInteger = 037'512;
int MinusOctFormattedInteger = -037'512;
int HexFormattedInteger = 0x4'F356;
int MinusHexFormattedInteger = -0x4'F356;
unsigned int UnsignedFormattedInteger = 10'004U;
unsigned int MinusUnsignedFormattedInteger = -10'004U;
unsigned int UnsignedFormattedInteger1 = 100'045u;
unsigned int MinusUnsignedFormattedInteger1 = -100'045u;
long LongFormattedInteger = 123'456'789'101'112L;
long MinusLongFormattedInteger = -123'456'789'101'112L;
long LongFormattedInteger1 = 12'345'678'910'111'213l;
long MinusLongFormattedInteger1 = -12'345'678'910'111'213l;
unsigned long UnsignedLongFormattedInteger1 = 12'345'678'910'111'213Ul;
unsigned long MinusUnsignedLongFormattedInteger1 = -12'345'678'910'111'213Ul;
float FormattedFloat = 1'234.567'89;
float MinusFormattedFloat = -1'234.567'89;
double PostfixFormattedFloat = 1'234.569F;
double MinusPostfixFormattedFloat = -1'234.569F;
double PostfixFormattedFloat1 = 1'234.569f;
double MinusPostfixFormattedFloat1 = -1'234.569f;
double ScientificFormattedFloat = 1.234'567'8E10;
double MinusScientificFormattedFloat = -1.234'567'8E10;
double ScientificFormattedFloat1 = 1.234'567'8e10;
double MinusScientificFormattedFloat1 = -1.234'567'8e10;
double ScientificFormattedFloat2 = 1.234'567'8E+10;
double MinusScientificFormattedFloat2 = -1.234'567'8E+10;
double ScientificFormattedFloat3 = 1.234'567'8e+10;
double MinusScientificFormattedFloat3 = -1.234'567'8e+10;
double PostfixScientificFormattedFloat = 1.234'567'8E10F;
double PostfixScientificFormattedFloat1 = 1.234'567'8e10f;
double PostfixScientificFormattedFloat2 = -1.234'567'8E+10f;
double PostfixScientificFormattedFloat4 = -1.234'567'8e-10F;


// Long wrong formatted literals

int WrongFormattedInteger = 1'2345'6'7;
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: unformatted representation of integer literal '1'2345'6'7' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234'567

int MinusWrongFormattedInteger = -1'2'3'4'5'6'7;
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: unformatted representation of integer literal '1'2'3'4'5'6'7' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234'567

int BinaryWrongFormattedInteger = 0b111'01101;
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: unformatted representation of integer literal '0b111'01101' [modernize-use-digit-separator]
// CHECK-FIXES: 0b1110'1101

int MinusBinaryWrongFormattedInteger = -0b11'10'11'01;
// CHECK-MESSAGES: :[[@LINE-1]]:41: warning: unformatted representation of integer literal '0b11'10'11'01' [modernize-use-digit-separator]
// CHECK-FIXES: 0b1110'1101

int OctWrongFormattedInteger = 0'37512;
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: unformatted representation of integer literal '0'37512' [modernize-use-digit-separator]
// CHECK-FIXES: 037'512

int MinusOctWrongFormattedInteger = -037'5'12;
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: unformatted representation of integer literal '037'5'12' [modernize-use-digit-separator]
// CHECK-FIXES: 037'512

int HexWrongFormattedInteger = 0x4f3'56;
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: unformatted representation of integer literal '0x4f3'56' [modernize-use-digit-separator]
// CHECK-FIXES: 0x4'F356

int MinusHexWrongFormattedInteger = -0x4'f3'5'6;
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: unformatted representation of integer literal '0x4'f3'5'6' [modernize-use-digit-separator]
// CHECK-FIXES: 0x4'F356

unsigned int UnsignedWrongFormattedInteger = 1'0004U;
// CHECK-MESSAGES: :[[@LINE-1]]:46: warning: unformatted representation of integer literal '1'0004U' [modernize-use-digit-separator]
// CHECK-FIXES: 10'004U

unsigned int MinusUnsignedWrongFormattedInteger = -100'0'4U;
// CHECK-MESSAGES: :[[@LINE-1]]:52: warning: unformatted representation of integer literal '100'0'4U' [modernize-use-digit-separator]
// CHECK-FIXES: 10'004U

unsigned int UnsignedWrongFormattedInteger1 = 10'0045u;
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: unformatted representation of integer literal '10'0045u' [modernize-use-digit-separator]
// CHECK-FIXES: 100'045u

unsigned int MinusUnsignedWrongFormattedInteger1 = -10'00'45u;
// CHECK-MESSAGES: :[[@LINE-1]]:53: warning: unformatted representation of integer literal '10'00'45u' [modernize-use-digit-separator]
// CHECK-FIXES: 100'045u

long LongWrongFormattedInteger = 1234'56'789'1011'12L;
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: unformatted representation of integer literal '1234'56'789'1011'12L' [modernize-use-digit-separator]
// CHECK-FIXES: 123'456'789'101'112L

long MinusLongWrongFormattedInteger = -1'234567891'011'1'2L;
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: unformatted representation of integer literal '1'234567891'011'1'2L' [modernize-use-digit-separator]
// CHECK-FIXES: 123'456'789'101'112L

long LongWrongFormattedInteger1 = 12345'67'89101'11213l;
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: unformatted representation of integer literal '12345'67'89101'11213l' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213l

long MinusLongWrongFormattedInteger1 = -1234567891'0111213l;
// CHECK-MESSAGES: :[[@LINE-1]]:41: warning: unformatted representation of integer literal '1234567891'0111213l' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213l

unsigned long UnsignedLongWrongFormattedInteger1 = 1234567'89101112'13Ul;
// CHECK-MESSAGES: :[[@LINE-1]]:52: warning: unformatted representation of integer literal '1234567'89101112'13Ul' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213Ul

unsigned long MinusUnsignedLongWrongFormattedInteger1 = -1'2'34'567'89'10'11'12'1'3Ul;
// CHECK-MESSAGES: :[[@LINE-1]]:58: warning: unformatted representation of integer literal '1'2'34'567'89'10'11'12'1'3Ul' [modernize-use-digit-separator]
// CHECK-FIXES: 12'345'678'910'111'213Ul

float WrongFormattedFloat = 1'234.56789;
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: unformatted representation of float literal '1'234.56789' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.567'89

float MinusWrongFormattedFloat = -1234.56'789;
// CHECK-MESSAGES: :[[@LINE-1]]:35: warning: unformatted representation of float literal '1234.56'789' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.567'89

double PostfixWrongFormattedFloat = 123'4.5'69F;
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: unformatted representation of float literal '123'4.5'69F' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.569F

double MinusPostfixWrongFormattedFloat = -12'34.569F;
// CHECK-MESSAGES: :[[@LINE-1]]:43: warning: unformatted representation of float literal '12'34.569F' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.569F

double PostfixWrongFormattedFloat1 = 1'2'34.569f;
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: unformatted representation of float literal '1'2'34.569f' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.569f

double MinusPostfixWrongFormattedFloat1 = -12'3'4.5'69f;
// CHECK-MESSAGES: :[[@LINE-1]]:44: warning: unformatted representation of float literal '12'3'4.5'69f' [modernize-use-digit-separator]
// CHECK-FIXES: 1'234.569f

double ScientificWrongFormattedFloat = 1.23'456'78E1'0;
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: unformatted representation of float literal '1.23'456'78E1'0' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E10

double MinusScientificWrongFormattedFloat = -1.234'5678E10;
// CHECK-MESSAGES: :[[@LINE-1]]:46: warning: unformatted representation of float literal '1.234'5678E10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E10

double ScientificWrongFormattedFloat1 = 1.234'5'67'8e10;
// CHECK-MESSAGES: :[[@LINE-1]]:41: warning: unformatted representation of float literal '1.234'5'67'8e10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e10

double MinusScientificWrongFormattedFloat1 = -1.2345678e1'0;
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: unformatted representation of float literal '1.2345678e1'0' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e10

double ScientificWrongFormattedFloat2 = 1.23456'78E+1'0;
// CHECK-MESSAGES: :[[@LINE-1]]:41: warning: unformatted representation of float literal '1.23456'78E+1'0' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E+10

double MinusScientificWrongFormattedFloat2 = -1.23'456'78E+10;
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: unformatted representation of float literal '1.23'456'78E+10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E+10

double ScientificWrongFormattedFloat3 = 1.234'56'78e+10;
// CHECK-MESSAGES: :[[@LINE-1]]:41: warning: unformatted representation of float literal '1.234'56'78e+10' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e+10

double MinusScientificWrongFormattedFloat3 = -1.2'3'4'5'678e+1'0;
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: unformatted representation of float literal '1.2'3'4'5'678e+1'0' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e+10

double PostfixScientificWrongFormattedFloat = 1.23456'78E1'0F;
// CHECK-MESSAGES: :[[@LINE-1]]:47: warning: unformatted representation of float literal '1.23456'78E1'0F' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E10F

double PostfixScientificWrongFormattedFloat1 = 1.23'45'6'7'8e10f;
// CHECK-MESSAGES: :[[@LINE-1]]:48: warning: unformatted representation of float literal '1.23'45'6'7'8e10f' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e10f

double PostfixScientificWrongFormattedFloat2 = -1.2'34'56'7'8E+10f;
// CHECK-MESSAGES: :[[@LINE-1]]:49: warning: unformatted representation of float literal '1.2'34'56'7'8E+10f' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8E+10f

double PostfixScientificWrongFormattedFloat4 = -1.23'456'78e-1'0F;
// CHECK-MESSAGES: :[[@LINE-1]]:49: warning: unformatted representation of float literal '1.23'456'78e-1'0F' [modernize-use-digit-separator]
// CHECK-FIXES: 1.234'567'8e-10F
