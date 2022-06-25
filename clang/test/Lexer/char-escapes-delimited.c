// RUN: %clang_cc1 -x c++ -std=gnu++11 -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -x c -std=gnu11 -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -x c++ -std=gnu++11 -fwchar-type=short -fno-signed-wchar -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -x c -std=gnu11 -fwchar-type=short -fno-signed-wchar -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -x c++ -std=c++17 -ftrigraphs -fsyntax-only -pedantic -verify -DTRIGRAPHS=1 %s

const char *errors =
    "\u{}"  // expected-error {{delimited escape sequence cannot be empty}}
    "\u{"   // expected-error {{expected '}'}}
    "\u{h}" // expected-error {{invalid digit 'h' in escape sequence}}
    "\x{}"  // expected-error {{delimited escape sequence cannot be empty}}
    "\x{"   // expected-error {{expected '}'}}
    "\x{h}" // expected-error {{invalid digit 'h' in escape sequence}}
    "\o{}"  // expected-error {{delimited escape sequence cannot be empty}}
    "\o{"   // expected-error {{expected '}'}}
    "\o"    // expected-error {{expected '{' after '\o' escape sequence}}
    "\o{8}" // expected-error {{invalid digit '8' in escape sequence}}
    "\U{8}" // expected-error {{\U used with no following hex digits}}
    ;

void ucn(void) {
  char a = '\u{1234}'; // expected-error {{character too large for enclosing character literal type}}
                       // expected-warning@-1 {{delimited escape sequences are a Clang extension}}

  unsigned b = U'\u{1234}'; // expected-warning {{extension}}

#ifdef __cplusplus
  unsigned b2 = U'\u{1}'; // expected-warning {{extension}}
#else
  unsigned b2 = U'\u{1}';     //expected-error {{universal character name refers to a control character}}
#endif

  unsigned c = U'\u{000000000001234}'; // expected-warning {{extension}}
  unsigned d = U'\u{111111111}';       //expected-error {{hex escape sequence out of range}}
}

void hex(void) {
  char a = '\x{1}';             // expected-warning {{extension}}
  char b = '\x{abcdegggggabc}'; // expected-error 5{{invalid digit 'g' in escape sequence}}
  char c = '\x{ff1}';           // expected-error {{hex escape sequence out of range}}

#if __WCHAR_MAX__ > 0xFFFF
  unsigned d = L'\x{FFFFFFFF}';  // expected-warning {{extension}}
  unsigned e = L'\x{100000000}'; // expected-error {{hex escape sequence out of range}}
#else
  unsigned f = L'\x{FFFF}';   // expected-warning {{extension}}
  unsigned g = L'\x{10000}';  // expected-error {{hex escape sequence out of range}}
#endif
  unsigned h = U'\x{FFFFFFFF}';  // expected-warning {{extension}}
  unsigned i = U'\x{100000000}'; // expected-error {{hex escape sequence out of range}}
}

void octal(void) {
  char a = '\o{1}';              // expected-warning {{extension}}
  char b = '\o{12345678881238}'; // expected-error 4{{invalid digit '8' in escape sequence}}
  char c = '\o{777}';            // //expected-error {{octal escape sequence out of range}}
#if __WCHAR_MAX__ > 0xFFFF
  unsigned d = L'\o{37777777777}'; // expected-warning {{extension}}
  unsigned e = L'\o{40000000000}'; // expected-error {{octal escape sequence out of range}}
#else
  unsigned d = L'\o{177777}'; // expected-warning {{extension}}
  unsigned e = L'\o{200000}'; // expected-error {{octal escape sequence out of range}}
#endif
}

void concat(void) {
  (void)"\x{" "12}"; // expected-error {{expected '}'}}
  (void)"\u{" "12}"; // expected-error {{expected '}'}}
  (void)"\o{" "12}"; // expected-error {{expected '}'}}

  (void)"\x{12" "}"; // expected-error {{expected '}'}}
  (void)"\u{12" "}"; // expected-error {{expected '}'}}
  (void)"\o{12" "}"; // expected-error {{expected '}'}}
}

void named(void) {
  char a = '\N{LOTUS}'; // expected-error{{character too large for enclosing character literal type}} \
                        // expected-warning {{extension}}

  char b  = '\N{DOLLAR SIGN}'; // expected-warning {{extension}}
  char b_ = '\N{ DOL-LAR _SIGN }'; // expected-error {{' DOL-LAR _SIGN ' is not a valid Unicode character name}} \
                               // expected-note {{characters names in Unicode escape sequences are sensitive to case and whitespaces}}

  char c = '\N{NOTATHING}'; // expected-error {{'NOTATHING' is not a valid Unicode character name}} \
                            // expected-note 5{{did you mean}}
  char d = '\N{}';          // expected-error {{delimited escape sequence cannot be empty}}
  char e = '\N{';           // expected-error {{incomplete universal character name}}

  unsigned f = L'\N{GREEK CAPITAL LETTER DELTA}'; // expected-warning {{extension}}

  unsigned g = u'\N{LOTUS}'; // expected-error {{character too large for enclosing character literal type}} \
                             // expected-warning {{extension}}

  unsigned h = U'\N{LOTUS}';                      // expected-warning {{extension}}
  unsigned i = u'\N{GREEK CAPITAL LETTER DELTA}'; // expected-warning {{extension}}
  char j = '\NN';                                 // expected-error {{expected '{' after '\N' escape sequence}}
  unsigned k = u'\N{LOTUS';                       // expected-error {{incomplete universal character name}}
}

void separators(void) {
  (void)"\x{12'3}"; // expected-error {{invalid digit ''' in escape sequence}}
  (void)"\u{12'3}"; // expected-error {{invalid digit ''' in escape sequence}}
  (void)"\o{12'3}"; // expected-error {{invalid digit ''' in escape sequence}}

  '\x{12'3'}';   // expected-error {{expected '}'}}
                 // expected-error@-1 2{{expected ';'}}
                 // expected-warning@-2 3{{expression result unused}}
}

#if L'\N{GREEK CAPITAL LETTER GAMMA}' != L'Γ' // expected-warning {{extension}}
#error "oh no!"
#endif

#ifdef TRIGRAPHS
static_assert('\N??<DOLLAR SIGN??>' == '$'); // expected-warning 2{{trigraph converted}} \
                                             // expected-warning {{named escape sequences are a Clang extension}}
#endif
