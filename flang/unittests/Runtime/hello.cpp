// Basic sanity tests of I/O API; exhaustive testing will be done in Fortran

#include "testing.h"
#include "../../runtime/descriptor.h"
#include "../../runtime/io-api.h"
#include <cstring>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

static bool test(const char *format, const char *expect, std::string &&got) {
  std::string want{expect};
  want.resize(got.length(), ' ');
  if (got != want) {
    Fail() << '\'' << format << "' failed;\n     got '" << got
           << "',\nexpected '" << want << "'\n";
    return false;
  }
  return true;
}

static void hello() {
  char buffer[32];
  const char *format{"(6HHELLO,,A6,2X,I3,1X,'0x',Z8,1X,L1)"};
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, sizeof buffer, format, std::strlen(format))};
  IONAME(OutputAscii)(cookie, "WORLD", 5);
  IONAME(OutputInteger64)(cookie, 678);
  IONAME(OutputInteger64)(cookie, 0xfeedface);
  IONAME(OutputLogical)(cookie, true);
  if (auto status{IONAME(EndIoStatement)(cookie)}) {
    Fail() << "hello: '" << format << "' failed, status "
           << static_cast<int>(status) << '\n';
  } else {
    test(format, "HELLO, WORLD  678 0xFEEDFACE T",
        std::string{buffer, sizeof buffer});
  }
}

static void multiline() {
  char buffer[5][32];
  StaticDescriptor<1> staticDescriptor[2];
  Descriptor &whole{staticDescriptor[0].descriptor()};
  SubscriptValue extent[]{5};
  whole.Establish(TypeCode{CFI_type_char}, sizeof buffer[0], &buffer, 1, extent,
      CFI_attribute_pointer);
  whole.Dump();
  whole.Check();
  Descriptor &section{staticDescriptor[1].descriptor()};
  SubscriptValue lowers[]{0}, uppers[]{4}, strides[]{1};
  section.Establish(whole.type(), whole.ElementBytes(), nullptr, 1, extent,
      CFI_attribute_pointer);
  if (auto error{
          CFI_section(&section.raw(), &whole.raw(), lowers, uppers, strides)}) {
    Fail() << "multiline: CFI_section failed: " << error << '\n';
    return;
  }
  section.Dump();
  section.Check();
  const char *format{
      "('?abcde,',T1,'>',T9,A,TL12,A,TR25,'<'//G0,17X,'abcd',1(2I4))"};
  auto cookie{IONAME(BeginInternalArrayFormattedOutput)(
      section, format, std::strlen(format))};
  IONAME(OutputAscii)(cookie, "WORLD", 5);
  IONAME(OutputAscii)(cookie, "HELLO", 5);
  IONAME(OutputInteger64)(cookie, 789);
  for (int j{666}; j <= 999; j += 111) {
    IONAME(OutputInteger64)(cookie, j);
  }
  if (auto status{IONAME(EndIoStatement)(cookie)}) {
    Fail() << "multiline: '" << format << "' failed, status "
           << static_cast<int>(status) << '\n';
  } else {
    test(format,
        ">HELLO, WORLD                  <"
        "                                "
        "789                 abcd 666 777"
        " 888 999                        "
        "                                ",
        std::string{buffer[0], sizeof buffer});
  }
}

static void listInputTest() {
  static const char input[]{",1*,(5.,6..)"};
  auto cookie{IONAME(BeginInternalListInput)(input, sizeof input - 1)};
  float z[6];
  for (int j{0}; j < 6; ++j) {
    z[j] = -(j + 1);
  }
  for (int j{0}; j < 6; j += 2) {
    if (!IONAME(InputComplex32)(cookie, &z[j])) {
      Fail() << "InputComplex32 failed\n";
    }
  }
  auto status{IONAME(EndIoStatement)(cookie)};
  if (status) {
    Fail() << "Failed complex list-directed input, status "
           << static_cast<int>(status) << '\n';
  } else {
    char output[33];
    output[32] = '\0';
    cookie = IONAME(BeginInternalListOutput)(output, 32);
    for (int j{0}; j < 6; j += 2) {
      if (!IONAME(OutputComplex32)(cookie, z[j], z[j + 1])) {
        Fail() << "OutputComplex32 failed\n";
      }
    }
    status = IONAME(EndIoStatement)(cookie);
    static const char expect[33]{" (-1.,-2.) (-3.,-4.) (5.,6.)    "};
    if (status) {
      Fail() << "Failed complex list-directed output, status "
             << static_cast<int>(status) << '\n';
    } else if (std::strncmp(output, expect, 33) != 0) {
      Fail() << "Failed complex list-directed output, expected '" << expect
             << "', but got '" << output << "'\n";
    }
  }
}

static void descrOutputTest() {
  char buffer[9];
  // Formatted
  const char *format{"(2A4)"};
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, sizeof buffer, format, std::strlen(format))};
  StaticDescriptor<1> staticDescriptor;
  Descriptor &desc{staticDescriptor.descriptor()};
  SubscriptValue extent[]{2};
  char data[2][4];
  std::memcpy(data[0], "ABCD", 4);
  std::memcpy(data[1], "EFGH", 4);
  desc.Establish(TypeCode{CFI_type_char}, sizeof data[0], &data, 1, extent);
  desc.Dump();
  desc.Check();
  IONAME(OutputDescriptor)(cookie, desc);
  if (auto status{IONAME(EndIoStatement)(cookie)}) {
    Fail() << "descrOutputTest: '" << format << "' failed, status "
           << static_cast<int>(status) << '\n';
  } else {
    test("descrOutputTest(formatted)", "ABCDEFGH ",
        std::string{buffer, sizeof buffer});
  }
  // List-directed
  cookie = IONAME(BeginInternalListOutput)(buffer, sizeof buffer);
  IONAME(OutputDescriptor)(cookie, desc);
  if (auto status{IONAME(EndIoStatement)(cookie)}) {
    Fail() << "descrOutputTest: list-directed failed, status "
           << static_cast<int>(status) << '\n';
  } else {
    test("descrOutputTest(list)", " ABCDEFGH",
        std::string{buffer, sizeof buffer});
  }
}

static void realTest(const char *format, double x, const char *expect) {
  char buffer[800];
  auto cookie{IONAME(BeginInternalFormattedOutput)(
      buffer, sizeof buffer, format, std::strlen(format))};
  IONAME(OutputReal64)(cookie, x);
  if (auto status{IONAME(EndIoStatement)(cookie)}) {
    Fail() << '\'' << format << "' failed, status " << static_cast<int>(status)
           << '\n';
  } else {
    test(format, expect, std::string{buffer, sizeof buffer});
  }
}

static void realInTest(
    const char *format, const char *data, std::uint64_t want) {
  auto cookie{IONAME(BeginInternalFormattedInput)(
      data, std::strlen(data), format, std::strlen(format))};
  union {
    double x;
    std::uint64_t raw;
  } u;
  u.raw = 0;
  IONAME(EnableHandlers)(cookie, true, true, true, true, true);
  IONAME(InputReal64)(cookie, u.x);
  char iomsg[65];
  iomsg[0] = '\0';
  iomsg[sizeof iomsg - 1] = '\0';
  IONAME(GetIoMsg)(cookie, iomsg, sizeof iomsg - 1);
  auto status{IONAME(EndIoStatement)(cookie)};
  if (status) {
    Fail() << '\'' << format << "' failed reading '" << data << "', status "
           << static_cast<int>(status) << " iomsg '" << iomsg << "'\n";
  } else if (u.raw != want) {
    Fail() << '\'' << format << "' failed reading '" << data << "', want 0x";
    Fail().write_hex(want) << ", got 0x" << u.raw << '\n';
  }
}

int main() {
  StartTests();

  hello();
  multiline();

  static const char *zeroes[][2]{
      {"(E32.17,';')", "         0.00000000000000000E+00;"},
      {"(F32.17,';')", "             0.00000000000000000;"},
      {"(G32.17,';')", "          0.0000000000000000    ;"},
      {"(DC,E32.17,';')", "         0,00000000000000000E+00;"},
      {"(DC,F32.17,';')", "             0,00000000000000000;"},
      {"(DC,G32.17,';')", "          0,0000000000000000    ;"},
      {"(D32.17,';')", "         0.00000000000000000D+00;"},
      {"(E32.17E1,';')", "          0.00000000000000000E+0;"},
      {"(G32.17E1,';')", "           0.0000000000000000   ;"},
      {"(E32.17E0,';')", "          0.00000000000000000E+0;"},
      {"(G32.17E0,';')", "          0.0000000000000000    ;"},
      {"(1P,E32.17,';')", "         0.00000000000000000E+00;"},
      {"(1PE32.17,';')", "         0.00000000000000000E+00;"}, // no comma
      {"(1P,F32.17,';')", "             0.00000000000000000;"},
      {"(1P,G32.17,';')", "          0.0000000000000000    ;"},
      {"(2P,E32.17,';')", "         00.0000000000000000E+00;"},
      {"(-1P,E32.17,';')", "         0.00000000000000000E+00;"},
      {"(G0,';')", "0.;"}, {}};
  for (int j{0}; zeroes[j][0]; ++j) {
    realTest(zeroes[j][0], 0.0, zeroes[j][1]);
  }

  static const char *ones[][2]{
      {"(E32.17,';')", "         0.10000000000000000E+01;"},
      {"(F32.17,';')", "             1.00000000000000000;"},
      {"(G32.17,';')", "          1.0000000000000000    ;"},
      {"(E32.17E1,';')", "          0.10000000000000000E+1;"},
      {"(G32.17E1,';')", "           1.0000000000000000   ;"},
      {"(E32.17E0,';')", "          0.10000000000000000E+1;"},
      {"(G32.17E0,';')", "          1.0000000000000000    ;"},
      {"(E32.17E4,';')", "       0.10000000000000000E+0001;"},
      {"(G32.17E4,';')", "        1.0000000000000000      ;"},
      {"(1P,E32.17,';')", "         1.00000000000000000E+00;"},
      {"(1PE32.17,';')", "         1.00000000000000000E+00;"}, // no comma
      {"(1P,F32.17,';')", "            10.00000000000000000;"},
      {"(1P,G32.17,';')", "          1.0000000000000000    ;"},
      {"(ES32.17,';')", "         1.00000000000000000E+00;"},
      {"(2P,E32.17,';')", "         10.0000000000000000E-01;"},
      {"(2P,G32.17,';')", "          1.0000000000000000    ;"},
      {"(-1P,E32.17,';')", "         0.01000000000000000E+02;"},
      {"(-1P,G32.17,';')", "          1.0000000000000000    ;"},
      {"(G0,';')", "1.;"}, {}};
  for (int j{0}; ones[j][0]; ++j) {
    realTest(ones[j][0], 1.0, ones[j][1]);
  }

  realTest("(E32.17,';')", -1.0, "        -0.10000000000000000E+01;");
  realTest("(F32.17,';')", -1.0, "            -1.00000000000000000;");
  realTest("(G32.17,';')", -1.0, "         -1.0000000000000000    ;");
  realTest("(G0,';')", -1.0, "-1.;");

  volatile union {
    double d;
    std::uint64_t n;
  } u;
  u.n = 0x8000000000000000; // -0
  realTest("(E9.1,';')", u.d, " -0.0E+00;");
  realTest("(F4.0,';')", u.d, " -0.;");
  realTest("(G8.0,';')", u.d, "-0.0E+00;");
  realTest("(G8.1,';')", u.d, " -0.    ;");
  realTest("(G0,';')", u.d, "-0.;");
  u.n = 0x7ff0000000000000; // +Inf
  realTest("(E9.1,';')", u.d, "      Inf;");
  realTest("(F9.1,';')", u.d, "      Inf;");
  realTest("(G9.1,';')", u.d, "      Inf;");
  realTest("(SP,E9.1,';')", u.d, "     +Inf;");
  realTest("(SP,F9.1,';')", u.d, "     +Inf;");
  realTest("(SP,G9.1,';')", u.d, "     +Inf;");
  realTest("(G0,';')", u.d, "Inf;");
  u.n = 0xfff0000000000000; // -Inf
  realTest("(E9.1,';')", u.d, "     -Inf;");
  realTest("(F9.1,';')", u.d, "     -Inf;");
  realTest("(G9.1,';')", u.d, "     -Inf;");
  realTest("(G0,';')", u.d, "-Inf;");
  u.n = 0x7ff0000000000001; // NaN
  realTest("(E9.1,';')", u.d, "      NaN;");
  realTest("(F9.1,';')", u.d, "      NaN;");
  realTest("(G9.1,';')", u.d, "      NaN;");
  realTest("(G0,';')", u.d, "NaN;");
  u.n = 0xfff0000000000001; // NaN (sign irrelevant)
  realTest("(E9.1,';')", u.d, "      NaN;");
  realTest("(F9.1,';')", u.d, "      NaN;");
  realTest("(G9.1,';')", u.d, "      NaN;");
  realTest("(SP,E9.1,';')", u.d, "      NaN;");
  realTest("(SP,F9.1,';')", u.d, "      NaN;");
  realTest("(SP,G9.1,';')", u.d, "      NaN;");
  realTest("(G0,';')", u.d, "NaN;");

  u.n = 0x3fb999999999999a; // 0.1 rounded
  realTest("(E62.55,';')", u.d,
      " 0.1000000000000000055511151231257827021181583404541015625E+00;");
  realTest("(E0.0,';')", u.d, "0.E+00;");
  realTest("(E0.55,';')", u.d,
      "0.1000000000000000055511151231257827021181583404541015625E+00;");
  realTest("(E0,';')", u.d, ".1E+00;");
  realTest("(F58.55,';')", u.d,
      " 0.1000000000000000055511151231257827021181583404541015625;");
  realTest("(F0.0,';')", u.d, "0.;");
  realTest("(F0.55,';')", u.d,
      ".1000000000000000055511151231257827021181583404541015625;");
  realTest("(F0,';')", u.d, ".1;");
  realTest("(G62.55,';')", u.d,
      " 0.1000000000000000055511151231257827021181583404541015625    ;");
  realTest("(G0.0,';')", u.d, "0.;");
  realTest("(G0.55,';')", u.d,
      ".1000000000000000055511151231257827021181583404541015625;");
  realTest("(G0,';')", u.d, ".1;");

  u.n = 0x3ff8000000000000; // 1.5
  realTest("(E9.2,';')", u.d, " 0.15E+01;");
  realTest("(F4.1,';')", u.d, " 1.5;");
  realTest("(G7.1,';')", u.d, " 2.    ;");
  realTest("(RN,E8.1,';')", u.d, " 0.2E+01;");
  realTest("(RN,F3.0,';')", u.d, " 2.;");
  realTest("(RN,G7.0,';')", u.d, " 0.E+01;");
  realTest("(RN,G7.1,';')", u.d, " 2.    ;");
  realTest("(RD,E8.1,';')", u.d, " 0.1E+01;");
  realTest("(RD,F3.0,';')", u.d, " 1.;");
  realTest("(RD,G7.0,';')", u.d, " 0.E+01;");
  realTest("(RD,G7.1,';')", u.d, " 1.    ;");
  realTest("(RU,E8.1,';')", u.d, " 0.2E+01;");
  realTest("(RU,G7.0,';')", u.d, " 0.E+01;");
  realTest("(RU,G7.1,';')", u.d, " 2.    ;");
  realTest("(RZ,E8.1,';')", u.d, " 0.1E+01;");
  realTest("(RZ,F3.0,';')", u.d, " 1.;");
  realTest("(RZ,G7.0,';')", u.d, " 0.E+01;");
  realTest("(RZ,G7.1,';')", u.d, " 1.    ;");
  realTest("(RC,E8.1,';')", u.d, " 0.2E+01;");
  realTest("(RC,F3.0,';')", u.d, " 2.;");
  realTest("(RC,G7.0,';')", u.d, " 0.E+01;");
  realTest("(RC,G7.1,';')", u.d, " 2.    ;");

  // TODO continue F and G editing tests on these data

  u.n = 0xbff8000000000000; // -1.5
  realTest("(E9.2,';')", u.d, "-0.15E+01;");
  realTest("(RN,E8.1,';')", u.d, "-0.2E+01;");
  realTest("(RD,E8.1,';')", u.d, "-0.2E+01;");
  realTest("(RU,E8.1,';')", u.d, "-0.1E+01;");
  realTest("(RZ,E8.1,';')", u.d, "-0.1E+01;");
  realTest("(RC,E8.1,';')", u.d, "-0.2E+01;");

  u.n = 0x4004000000000000; // 2.5
  realTest("(E9.2,';')", u.d, " 0.25E+01;");
  realTest("(RN,E8.1,';')", u.d, " 0.2E+01;");
  realTest("(RD,E8.1,';')", u.d, " 0.2E+01;");
  realTest("(RU,E8.1,';')", u.d, " 0.3E+01;");
  realTest("(RZ,E8.1,';')", u.d, " 0.2E+01;");
  realTest("(RC,E8.1,';')", u.d, " 0.3E+01;");

  u.n = 0xc004000000000000; // -2.5
  realTest("(E9.2,';')", u.d, "-0.25E+01;");
  realTest("(RN,E8.1,';')", u.d, "-0.2E+01;");
  realTest("(RD,E8.1,';')", u.d, "-0.3E+01;");
  realTest("(RU,E8.1,';')", u.d, "-0.2E+01;");
  realTest("(RZ,E8.1,';')", u.d, "-0.2E+01;");
  realTest("(RC,E8.1,';')", u.d, "-0.3E+01;");

  u.n = 1; // least positive nonzero subnormal
  realTest("(E32.17,';')", u.d, "         0.49406564584124654-323;");
  realTest("(ES32.17,';')", u.d, "         4.94065645841246544-324;");
  realTest("(EN32.17,';')", u.d, "         4.94065645841246544-324;");
  realTest("(E759.752,';')", u.d,
      " 0."
      "494065645841246544176568792868221372365059802614324764425585682500675507"
      "270208751865299836361635992379796564695445717730926656710355939796398774"
      "796010781878126300713190311404527845817167848982103688718636056998730723"
      "050006387409153564984387312473397273169615140031715385398074126238565591"
      "171026658556686768187039560310624931945271591492455329305456544401127480"
      "129709999541931989409080416563324524757147869014726780159355238611550134"
      "803526493472019379026810710749170333222684475333572083243193609238289345"
      "836806010601150616980975307834227731832924790498252473077637592724787465"
      "608477820373446969953364701797267771758512566055119913150489110145103786"
      "273816725095583738973359899366480994116420570263709027924276754456522908"
      "75386825064197182655334472656250-323;");
  realTest("(G0,';')", u.d, ".5-323;");
  realTest("(E757.750,';')", u.d,
      " 0."
      "494065645841246544176568792868221372365059802614324764425585682500675507"
      "270208751865299836361635992379796564695445717730926656710355939796398774"
      "796010781878126300713190311404527845817167848982103688718636056998730723"
      "050006387409153564984387312473397273169615140031715385398074126238565591"
      "171026658556686768187039560310624931945271591492455329305456544401127480"
      "129709999541931989409080416563324524757147869014726780159355238611550134"
      "803526493472019379026810710749170333222684475333572083243193609238289345"
      "836806010601150616980975307834227731832924790498252473077637592724787465"
      "608477820373446969953364701797267771758512566055119913150489110145103786"
      "273816725095583738973359899366480994116420570263709027924276754456522908"
      "753868250641971826553344726562-323;");
  realTest("(RN,E757.750,';')", u.d,
      " 0."
      "494065645841246544176568792868221372365059802614324764425585682500675507"
      "270208751865299836361635992379796564695445717730926656710355939796398774"
      "796010781878126300713190311404527845817167848982103688718636056998730723"
      "050006387409153564984387312473397273169615140031715385398074126238565591"
      "171026658556686768187039560310624931945271591492455329305456544401127480"
      "129709999541931989409080416563324524757147869014726780159355238611550134"
      "803526493472019379026810710749170333222684475333572083243193609238289345"
      "836806010601150616980975307834227731832924790498252473077637592724787465"
      "608477820373446969953364701797267771758512566055119913150489110145103786"
      "273816725095583738973359899366480994116420570263709027924276754456522908"
      "753868250641971826553344726562-323;");
  realTest("(RD,E757.750,';')", u.d,
      " 0."
      "494065645841246544176568792868221372365059802614324764425585682500675507"
      "270208751865299836361635992379796564695445717730926656710355939796398774"
      "796010781878126300713190311404527845817167848982103688718636056998730723"
      "050006387409153564984387312473397273169615140031715385398074126238565591"
      "171026658556686768187039560310624931945271591492455329305456544401127480"
      "129709999541931989409080416563324524757147869014726780159355238611550134"
      "803526493472019379026810710749170333222684475333572083243193609238289345"
      "836806010601150616980975307834227731832924790498252473077637592724787465"
      "608477820373446969953364701797267771758512566055119913150489110145103786"
      "273816725095583738973359899366480994116420570263709027924276754456522908"
      "753868250641971826553344726562-323;");
  realTest("(RU,E757.750,';')", u.d,
      " 0."
      "494065645841246544176568792868221372365059802614324764425585682500675507"
      "270208751865299836361635992379796564695445717730926656710355939796398774"
      "796010781878126300713190311404527845817167848982103688718636056998730723"
      "050006387409153564984387312473397273169615140031715385398074126238565591"
      "171026658556686768187039560310624931945271591492455329305456544401127480"
      "129709999541931989409080416563324524757147869014726780159355238611550134"
      "803526493472019379026810710749170333222684475333572083243193609238289345"
      "836806010601150616980975307834227731832924790498252473077637592724787465"
      "608477820373446969953364701797267771758512566055119913150489110145103786"
      "273816725095583738973359899366480994116420570263709027924276754456522908"
      "753868250641971826553344726563-323;");
  realTest("(RC,E757.750,';')", u.d,
      " 0."
      "494065645841246544176568792868221372365059802614324764425585682500675507"
      "270208751865299836361635992379796564695445717730926656710355939796398774"
      "796010781878126300713190311404527845817167848982103688718636056998730723"
      "050006387409153564984387312473397273169615140031715385398074126238565591"
      "171026658556686768187039560310624931945271591492455329305456544401127480"
      "129709999541931989409080416563324524757147869014726780159355238611550134"
      "803526493472019379026810710749170333222684475333572083243193609238289345"
      "836806010601150616980975307834227731832924790498252473077637592724787465"
      "608477820373446969953364701797267771758512566055119913150489110145103786"
      "273816725095583738973359899366480994116420570263709027924276754456522908"
      "753868250641971826553344726563-323;");

  u.n = 0x10000000000000; // least positive nonzero normal
  realTest("(E723.716,';')", u.d,
      " 0."
      "222507385850720138309023271733240406421921598046233183055332741688720443"
      "481391819585428315901251102056406733973103581100515243416155346010885601"
      "238537771882113077799353200233047961014744258363607192156504694250373420"
      "837525080665061665815894872049117996859163964850063590877011830487479978"
      "088775374994945158045160505091539985658247081864511353793580499211598108"
      "576605199243335211435239014879569960959128889160299264151106346631339366"
      "347758651302937176204732563178148566435087212282863764204484681140761391"
      "147706280168985324411002416144742161856716615054015428508471675290190316"
      "132277889672970737312333408698898317506783884692609277397797285865965494"
      "10913690954061364675687023986783152906809846172109246253967285156250-"
      "307;");
  realTest("(G0,';')", u.d, ".22250738585072014-307;");

  u.n = 0x7fefffffffffffffuLL; // greatest finite
  realTest("(E32.17,';')", u.d, "         0.17976931348623157+309;");
  realTest("(E317.310,';')", u.d,
      " 0."
      "179769313486231570814527423731704356798070567525844996598917476803157260"
      "780028538760589558632766878171540458953514382464234321326889464182768467"
      "546703537516986049910576551282076245490090389328944075868508455133942304"
      "583236903222948165808559332123348274797826204144723168738177180919299881"
      "2504040261841248583680+309;");
  realTest("(ES317.310,';')", u.d,
      " 1."
      "797693134862315708145274237317043567980705675258449965989174768031572607"
      "800285387605895586327668781715404589535143824642343213268894641827684675"
      "467035375169860499105765512820762454900903893289440758685084551339423045"
      "832369032229481658085593321233482747978262041447231687381771809192998812"
      "5040402618412485836800+308;");
  realTest("(EN319.310,';')", u.d,
      " 179."
      "769313486231570814527423731704356798070567525844996598917476803157260780"
      "028538760589558632766878171540458953514382464234321326889464182768467546"
      "703537516986049910576551282076245490090389328944075868508455133942304583"
      "236903222948165808559332123348274797826204144723168738177180919299881250"
      "4040261841248583680000+306;");
  realTest("(G0,';')", u.d, ".17976931348623157+309;");

  realTest("(F5.3,';')", 25., "*****;");
  realTest("(F5.3,';')", 2.5, "2.500;");
  realTest("(F5.3,';')", 0.25, "0.250;");
  realTest("(F5.3,';')", 0.025, "0.025;");
  realTest("(F5.3,';')", 0.0025, "0.003;");
  realTest("(F5.3,';')", 0.00025, "0.000;");
  realTest("(F5.3,';')", 0.000025, "0.000;");
  realTest("(F5.3,';')", -25., "*****;");
  realTest("(F5.3,';')", -2.5, "*****;");
  realTest("(F5.3,';')", -0.25, "-.250;");
  realTest("(F5.3,';')", -0.025, "-.025;");
  realTest("(F5.3,';')", -0.0025, "-.003;");
  realTest("(F5.3,';')", -0.00025, "-.000;");
  realTest("(F5.3,';')", -0.000025, "-.000;");

  realInTest("(F18.0)", "                 0", 0x0);
  realInTest("(F18.0)", "                  ", 0x0);
  realInTest("(F18.0)", "                -0", 0x8000000000000000);
  realInTest("(F18.0)", "                01", 0x3ff0000000000000);
  realInTest("(F18.0)", "                 1", 0x3ff0000000000000);
  realInTest("(F18.0)", "              125.", 0x405f400000000000);
  realInTest("(F18.0)", "              12.5", 0x4029000000000000);
  realInTest("(F18.0)", "              1.25", 0x3ff4000000000000);
  realInTest("(F18.0)", "             01.25", 0x3ff4000000000000);
  realInTest("(F18.0)", "              .125", 0x3fc0000000000000);
  realInTest("(F18.0)", "             0.125", 0x3fc0000000000000);
  realInTest("(F18.0)", "             .0625", 0x3fb0000000000000);
  realInTest("(F18.0)", "            0.0625", 0x3fb0000000000000);
  realInTest("(F18.0)", "               125", 0x405f400000000000);
  realInTest("(F18.1)", "               125", 0x4029000000000000);
  realInTest("(F18.2)", "               125", 0x3ff4000000000000);
  realInTest("(F18.3)", "               125", 0x3fc0000000000000);
  realInTest("(-1P,F18.0)", "               125", 0x4093880000000000); // 1250
  realInTest("(1P,F18.0)", "               125", 0x4029000000000000); // 12.5
  realInTest("(BZ,F18.0)", "              125 ", 0x4093880000000000); // 1250
  realInTest("(BZ,F18.0)", "       125 . e +1 ", 0x42a6bcc41e900000); // 1.25e13
  realInTest("(DC,F18.0)", "              12,5", 0x4029000000000000);

  listInputTest();
  descrOutputTest();

  return EndTests();
}
