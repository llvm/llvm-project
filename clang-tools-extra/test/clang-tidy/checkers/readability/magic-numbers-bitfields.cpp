// RUN: %check_clang_tidy %s readability-magic-numbers %t \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-magic-numbers.IgnoredIntegerValues: "1;2;10;100;"}}' \
// RUN: --

struct HardwareGateway {
   /*
    * The configuration suppresses the warnings for the bitfields...
    */
   unsigned int Some: 5;
   unsigned int Bits: 7;
   unsigned int: 7;
   unsigned int: 0;
   unsigned int Rest: 13;

   /*
    * ... but other fields trigger the warning.
    */
   unsigned int Another[3];
   // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 3 is a magic number; consider replacing it with a named constant [readability-magic-numbers]
};

