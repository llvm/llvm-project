// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.NewDelete,unix.Malloc -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.NewDelete,unix.Malloc -analyzer-output=text -analyzer-config c++-allocator-inlining=true -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.NewDelete,unix.Malloc -analyzer-output=plist %s -o %t.plist
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/NewDelete-path-notes.cpp.plist -

void test() {
  int *p = new int;
  // expected-note@-1 {{Memory is allocated}}
  if (p)
    // expected-note@-1 {{Taking true branch}}
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

