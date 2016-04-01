// RUN: %check_clang_tidy %s misc-suspicious-missing-comma %t

const char* Cartoons[] = {
  "Bugs Bunny",
  "Homer Simpson",
  "Mickey Mouse",
  "Bart Simpson",
  "Charlie Brown"  // There is a missing comma here.
  "Fred Flintstone",
  "Popeye",
};
// CHECK-MESSAGES: :[[@LINE-4]]:3: warning: suspicious string literal, probably missing a comma [misc-suspicious-missing-comma]

const wchar_t* Colors[] = {
  L"Red", L"Yellow", L"Blue", L"Green", L"Purple", L"Rose", L"White", L"Black"
};

// The following array should not trigger any warnings.
const char* HttpCommands[] = {
  "GET / HTTP/1.0\r\n"
  "\r\n",

  "GET /index.html HTTP/1.0\r\n"
  "\r\n",

  "GET /favicon.ico HTTP/1.0\r\n"
  "header: dummy"
  "\r\n",
};

// This array is too small to trigger a warning.
const char* SmallArray[] = {
  "a" "b", "c"
};
