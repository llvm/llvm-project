// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest -forest-abbrev=0 | FileCheck %s --implicit-check-not=ambiguous
auto list = {
  0,      // CHECK: := integer-literal
  0b1011, // CHECK: := integer-literal
  0777,   // CHECK: := integer-literal
  42_u,   // CHECK: := user-defined-integer-literal
  0LL,    // CHECK: := integer-literal
  0h,     // CHECK: := user-defined-integer-literal
  0.,     // CHECK: := floating-point-literal
  .2,     // CHECK: := floating-point-literal
  2e1,    // CHECK: := floating-point-literal
  0x42d,  // CHECK: := integer-literal
  0x42_d, // CHECK: := user-defined-integer-literal
  0x42ds, // CHECK: := user-defined-integer-literal
  0x1.2p2,// CHECK: := floating-point-literal
  
  "",               // CHECK: literal := string-literal
  L"",              // CHECK: literal := string-literal
  u8"",             // CHECK: literal := string-literal
  u"",              // CHECK: literal := string-literal
  U"",              // CHECK: literal := string-literal
  R"()",            // CHECK: literal := string-literal
  uR"()",           // CHECK: literal := string-literal
  "a" "b",          // CHECK: literal := string-literal
  u8"a" "b",        // CHECK: literal := string-literal
  u"a" u"b",        // CHECK: literal := string-literal
  "a"_u "b",        // CHECK: user-defined-literal := user-defined-string-literal
  "a"_u u"b",       // CHECK: user-defined-literal := user-defined-string-literal
  R"(a)" "\n",      // CHECK: literal := string-literal
  R"c(a)c"_u u"\n", // CHECK: user-defined-literal := user-defined-string-literal

  'a',      // CHECK: := character-literal
  'abc',    // CHECK: := character-literal
  'abcdef', // CHECK: := character-literal
  u'a',     // CHECK: := character-literal
  U'a',     // CHECK: := character-literal
  L'a',     // CHECK: := character-literal
  L'abc',   // CHECK: := character-literal
  U'\u1234',// CHECK: := character-literal
  '\u1234', // CHECK: := character-literal
  u'a'_u,   // CHECK: := user-defined-character-literal
};

