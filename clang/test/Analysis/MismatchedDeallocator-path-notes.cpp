// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.MismatchedDeallocator -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.MismatchedDeallocator -analyzer-output=plist %s -o %t.plist
// RUN: tail -n +11 %t.plist | diff -u -w -I "<string>/" -I "<string>.:" -I "version" - %S/copypaste/Inputs/expected-plists/MismatchedDeallocator-path-notes.cpp.plist

void changePointee(int *p);
int *allocIntArray(unsigned c) {
  return new int[c]; // expected-note {{Memory is allocated}}
}
void test() {
  int *p = allocIntArray(1); // expected-note {{Calling 'allocIntArray'}}
  // expected-note@-1 {{Returned allocated memory}}
  changePointee(p);
  delete p; // expected-warning {{Memory allocated by 'new[]' should be deallocated by 'delete[]', not 'delete'}}
  // expected-note@-1 {{Memory allocated by 'new[]' should be deallocated by 'delete[]', not 'delete'}}
}
