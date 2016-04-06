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

// The following array should not trigger any warnings. There is more than 5
// elements, but they are all concatenated string literals.
const char* HttpCommands[] = {
  "GET / HTTP/1.0\r\n"
  "\r\n",

  "GET /index.html HTTP/1.0\r\n"
  "\r\n",

  "GET /favicon.ico HTTP/1.0\r\n"
  "header: dummy"
  "\r\n",

  "GET /index.html-en HTTP/1.0\r\n"
  "\r\n",

  "GET /index.html-fr HTTP/1.0\r\n"
  "\r\n",

  "GET /index.html-es HTTP/1.0\r\n"
  "\r\n",
};

// This array is too small to trigger a warning.
const char* SmallArray[] = {
  "a" "b", "c"
};

// Parentheses should be enough to avoid warnings.
const char* ParentheseArray[] = {
  ("a" "b"), "c",
  ("d"
   "e"
   "f"),
  "g", "h", "i", "j", "k", "l"
};

// Indentation should be enough to avoid warnings.
const char* CorrectlyIndentedArray[] = {
  "This is a long message "
      "which is spanning over multiple lines."
      "And this should be fine.",
  "a", "b", "c", "d", "e", "f",
  "g", "h", "i", "j", "k", "l"
};

const char* IncorrectlyIndentedArray[] = {
  "This is a long message "
  "which is spanning over multiple lines."
      "And this should be fine.",
  "a", "b", "c", "d", "e", "f",
  "g", "h", "i", "j", "k", "l"
};
// CHECK-MESSAGES: :[[@LINE-6]]:3: warning: suspicious string literal, probably missing a comma [misc-suspicious-missing-comma]

const char* TooManyConcatenatedTokensArray[] = {
  "Dummy line",
  "Dummy line",
  "a" "b" "c" "d" "e" "f",
  "g" "h" "i" "j" "k" "l",
  "Dummy line",
  "Dummy line",
  "Dummy line",
  "Dummy line",
};
