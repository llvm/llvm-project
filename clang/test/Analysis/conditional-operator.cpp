// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config eagerly-assume=false %s -analyzer-output=text -verify

void clang_analyzer_eval(bool);

// Test that the analyzer does not crash on GNU extension operator "?:".
void NoCrashTest(int x, int y) {
	int w = x ?: y;
}

void OperatorEvaluationTest(int y) {
  int x = 1;
	int w = x ?: y;  // expected-note {{'?' condition is true}}
	
	// TODO: We are not precise when processing the "?:" operator in C++.
  clang_analyzer_eval(w == 1); // expected-warning{{UNKNOWN}}
                               // expected-note@-1{{UNKNOWN}}
}
