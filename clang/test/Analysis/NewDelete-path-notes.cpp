// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=cplusplus.NewDelete,unix.Malloc \
// RUN:   -analyzer-config add-pop-up-notes=false \
// RUN:   -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=cplusplus.NewDelete,unix.Malloc \
// RUN:   -analyzer-config c++-allocator-inlining=true \
// RUN:   -analyzer-config add-pop-up-notes=false \
// RUN:   -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=cplusplus.NewDelete,unix.Malloc \
// RUN:   -analyzer-config add-pop-up-notes=false \
// RUN:   -analyzer-output=plist %s -o %t.plist
// RUN: %normalize_plist <%t.plist | diff -ub \
// RUN:   %S/Inputs/expected-plists/NewDelete-path-notes.cpp.plist -

void test() {
  int *p = new int;
  // expected-note@-1 {{Memory is allocated}}
  if (p) // expected-note {{Taking true branch}}
    delete p;
    // expected-note@-1 {{Memory is released}}

  delete p; // expected-warning {{Attempt to free released memory}}
  // expected-note@-1 {{Attempt to free released memory}}
}

struct Odd {
	void kill() {
		delete this; // expected-note {{Memory is released}}
	}
};

void test(Odd *odd) {
	odd->kill(); // expected-note{{Calling 'Odd::kill'}}
               // expected-note@-1 {{Returning; memory was released}}
	delete odd; // expected-warning {{Attempt to free released memory}}
              // expected-note@-1 {{Attempt to free released memory}}
}

