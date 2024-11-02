// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck -check-prefix=CHECK-MACHINE %s
const char*
\N{GREEK_SMALL_LETTER-OMICRON} = // expected-error {{'GREEK_SMALL_LETTER-OMICRON' is not a valid Unicode character name}} \
                                 // expected-note {{sensitive to case and whitespaces}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-2]]:4-[[@LINE-2]]:30}:"GREEK SMALL LETTER OMICRON"

"\N{zero width no break space}" // expected-error {{'zero width no break space' is not a valid Unicode character name}} \
                               // expected-note {{sensitive to case and whitespaces}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-2]]:5-[[@LINE-2]]:30}:"ZERO WIDTH NO-BREAK SPACE"

"abc\N{MAN IN A BUSINESS SUIT LEVITATING}" // expected-error {{'MAN IN A BUSINESS SUIT LEVITATING' is not a valid Unicode character name}} \
                                           // expected-note {{did you mean MAN IN BUSINESS SUIT LEVITATING ('🕴' U+1F574)?}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-2]]:8-[[@LINE-2]]:41}:"MAN IN BUSINESS SUIT LEVITATING"

"\N{AAA}" // expected-error {{'AAA' is not a valid Unicode character name}} \
          // expected-note 5{{did you mean}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-2]]:5-[[@LINE-2]]:8}:"ANT"
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-3]]:5-[[@LINE-3]]:8}:"ARC"
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-4]]:5-[[@LINE-4]]:8}:"AXE"
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-5]]:5-[[@LINE-5]]:8}:"BAT"
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-6]]:5-[[@LINE-6]]:8}:"CAT"

"\N{BLACKCHESSBISHOP}" // expected-error {{'BLACKCHESSBISHOP' is not a valid Unicode character name}} \
                       // expected-note {{sensitive to case and whitespaces}}
// CHECK-MACHINE: fix-it:"{{.*}}":{[[@LINE-2]]:5-[[@LINE-2]]:21}:"BLACK CHESS BISHOP"

;


