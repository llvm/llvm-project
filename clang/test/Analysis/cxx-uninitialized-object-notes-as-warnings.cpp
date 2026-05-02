// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:NotesAsWarnings=true \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:CheckPointeeInitialization=true \
// RUN:   -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus.UninitializedObject \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:NotesAsWarnings=true \
// RUN:   -analyzer-config optin.cplusplus.UninitializedObject:CheckPointeeInitialization=true \
// RUN:   -std=c++11 -verify %s -DHEAP_ALLOCATION

#ifdef HEAP_ALLOCATION
#define INIT(CLS, ARGS) new CLS ARGS
#else
#define INIT(CLS, ARGS) (void) CLS ARGS
#endif

class NotesAsWarningsTest {
  int a;
  int b;
  int dontGetFilteredByNonPedanticMode = 0;

public:
  NotesAsWarningsTest() {} // expected-warning{{uninitialized field 'this->a'}}
  // expected-warning@-1{{uninitialized field 'this->b'}}
};

void fNotesAsWarningsTest() {
  INIT(NotesAsWarningsTest, ());
}
