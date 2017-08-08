// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:IgnoredFilesPattern="moc_|ui_|dbus_|.*_automoc" -verify %s

void f1() {
  int *p1 = new int[1];
  int *p2 = new int[1];
  if (p1) {
    delete [] p1; // expected-note{{Similar code using 'p1' here}}
    p1 = nullptr;
  }
  if (p2) {
    delete [] p1; // expected-warning{{Potential copy-paste error; did you really mean to use 'p1' here?}}
    p2 = nullptr;
  }
}
