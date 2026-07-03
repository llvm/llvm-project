// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection %s 2>&1 | FileCheck %s

template <class... Ts>
void escape(Ts&...);
void clang_analyzer_dump(int);
void clang_analyzer_dumpInvalidationHistory(int);

void escapeParam(int param) {
  clang_analyzer_dump(param);                    // expected-warning {{reg_$}}
  clang_analyzer_dumpInvalidationHistory(param); // expected-warning {{reg_$}}
  // CHECK: chains.cpp:[[@LINE-2]]:3: warning: reg_$[[ID1:[0-9]+]]<int param> [debug.ExprInspection]
  // CHECK: chains.cpp:[[@LINE-2]]:3: warning: reg_$[[ID1:[0-9]+]]<int param> [debug.ExprInspection]

  escape(param);

  clang_analyzer_dump(param);                    // expected-warning    {{inv_$}}
  clang_analyzer_dumpInvalidationHistory(param); // expected-warning-re {{{{inv_\$.+ -> reg_\$.+}}}}
  // CHECK: chains.cpp:[[@LINE-2]]:3: warning: inv_$[[ID2:[0-9]+]]{int, LC[[#]], conservative-call, S[[#]], prev=reg_$[[ID1]]<int param>, #[[#]]} [debug.ExprInspection]
  // CHECK: chains.cpp:[[@LINE-2]]:3: warning: inv_$[[ID2:[0-9]+]]{int, LC[[#]], conservative-call, S[[#]], prev=reg_$[[ID1]]<int param>, #[[#]]} -> reg_$[[ID1]]<int param> [debug.ExprInspection]

  escape(param);

  clang_analyzer_dump(param);                    // expected-warning {{inv_$}}
  clang_analyzer_dumpInvalidationHistory(param); // expected-warning-re {{{{inv_\$.+ -> inv_\$.+ -> reg_\$.+}}}}
  // CHECK: chains.cpp:[[@LINE-2]]:3: warning: inv_$[[ID3:[0-9]+]]{int, LC[[#]], conservative-call, S[[#]], prev=inv_$[[ID2]], #[[#]]} [debug.ExprInspection]
  // CHECK: chains.cpp:[[@LINE-2]]:3: warning: inv_$[[ID3:[0-9]+]]{int, LC[[#]], conservative-call, S[[#]], prev=inv_$[[ID2]], #[[#]]} -> inv_$[[ID2:[0-9]+]]{int, LC[[#]], conservative-call, S[[#]], prev=reg_$[[ID1]]<int param>, #[[#]]} -> reg_$[[ID1]]<int param> [debug.ExprInspection]
}
