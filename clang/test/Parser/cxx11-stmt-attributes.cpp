// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++11 %s

void foo(int i) {

  [[unknown_attribute]] ; // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  [[unknown_attribute]] { } // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  [[unknown_attribute]] if (0) { } // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  [[unknown_attribute]] for (;;); // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  [[unknown_attribute]] do { // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
    [[unknown_attribute]] continue; // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  } while (0);
  [[unknown_attribute]] while (0); // expected-warning {{unknown attribute 'unknown_attribute' ignored}}

  [[unknown_attribute]] switch (i) { // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
    [[unknown_attribute]] case 0: // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
    [[unknown_attribute]] default: // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
      [[unknown_attribute]] break; // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  }

  [[unknown_attribute]] goto here; // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  [[unknown_attribute]] here: // expected-warning {{unknown attribute 'unknown_attribute' ignored}}

  [[unknown_attribute]] try { // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
  } catch (...) {
  }

  [[unknown_attribute]] return; // expected-warning {{unknown attribute 'unknown_attribute' ignored}}
	 

  alignas(8) ; // expected-warning {{attribute aligned cannot be specified on a statement}}
  [[noreturn]] { } // expected-warning {{attribute noreturn cannot be specified on a statement}}
  [[noreturn]] if (0) { } // expected-warning {{attribute noreturn cannot be specified on a statement}}
  [[noreturn]] for (;;); // expected-warning {{attribute noreturn cannot be specified on a statement}}
  [[noreturn]] do { // expected-warning {{attribute noreturn cannot be specified on a statement}}
    [[unavailable]] continue; // expected-warning {{unknown attribute 'unavailable' ignored}}
  } while (0);
  [[unknown_attributqqq]] while (0); // expected-warning {{unknown attribute 'unknown_attributqqq' ignored}}
	// TODO: remove 'qqq' part and enjoy 'empty loop body' warning here (DiagnoseEmptyLoopBody)

  [[unknown_attribute]] while (0); // expected-warning {{unknown attribute 'unknown_attribute' ignored}}

  [[unused]] switch (i) { // expected-warning {{unknown attribute 'unused' ignored}}
    [[uuid]] case 0: // expected-warning {{unknown attribute 'uuid' ignored}}
    [[visibility]] default: // expected-warning {{unknown attribute 'visibility' ignored}}
      [[carries_dependency]] break; // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
  }

  [[fastcall]] goto there; // expected-warning {{unknown attribute 'fastcall' ignored}}
  [[noinline]] there: // expected-warning {{unknown attribute 'noinline' ignored}}

  [[lock_returned]] try { // expected-warning {{unknown attribute 'lock_returned' ignored}}
  } catch (...) {
  }

  [[weakref]] return; // expected-warning {{unknown attribute 'weakref' ignored}}

  [[carries_dependency]] ; // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
  [[carries_dependency]] { } // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
  [[carries_dependency]] if (0) { } // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
  [[carries_dependency]] for (;;); // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
  [[carries_dependency]] do { // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
    [[carries_dependency]] continue; // expected-warning {{attribute carries_dependency cannot be specified on a statement}} ignored}}
  } while (0);
  [[carries_dependency]] while (0); // expected-warning {{attribute carries_dependency cannot be specified on a statement}}

  [[carries_dependency]] switch (i) { // expected-warning {{attribute carries_dependency cannot be specified on a statement}} ignored}}
    [[carries_dependency]] case 0: // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
    [[carries_dependency]] default: // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
      [[carries_dependency]] break; // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
  }

  [[carries_dependency]] goto here; // expected-warning {{attribute carries_dependency cannot be specified on a statement}}

  [[carries_dependency]] try { // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
  } catch (...) {
  }

  [[carries_dependency]] return; // expected-warning {{attribute carries_dependency cannot be specified on a statement}}
}
