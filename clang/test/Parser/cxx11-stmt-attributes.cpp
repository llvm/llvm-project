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
	 

  alignas(8) ; // expected-error {{'alignas' attribute cannot be applied to a statement}}
  [[noreturn]] { } // expected-error {{'noreturn' attribute cannot be applied to a statement}}
  [[noreturn]] if (0) { } // expected-error {{'noreturn' attribute cannot be applied to a statement}}
  [[noreturn]] for (;;); // expected-error {{'noreturn' attribute cannot be applied to a statement}}
  [[noreturn]] do { // expected-error {{'noreturn' attribute cannot be applied to a statement}}
    [[unavailable]] continue; // expected-warning {{unknown attribute 'unavailable' ignored}}
  } while (0);
  [[unknown_attributqqq]] while (0); // expected-warning {{unknown attribute 'unknown_attributqqq' ignored}}
	// TODO: remove 'qqq' part and enjoy 'empty loop body' warning here (DiagnoseEmptyLoopBody)

  [[unknown_attribute]] while (0); // expected-warning {{unknown attribute 'unknown_attribute' ignored}}

  [[unused]] switch (i) { // expected-warning {{unknown attribute 'unused' ignored}}
    [[uuid]] case 0: // expected-warning {{unknown attribute 'uuid' ignored}}
    [[visibility]] default: // expected-warning {{unknown attribute 'visibility' ignored}}
      [[carries_dependency]] break; // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
  }

  [[fastcall]] goto there; // expected-warning {{unknown attribute 'fastcall' ignored}}
  [[noinline]] there: // expected-warning {{unknown attribute 'noinline' ignored}}

  [[lock_returned]] try { // expected-warning {{unknown attribute 'lock_returned' ignored}}
  } catch (...) {
  }

  [[weakref]] return; // expected-warning {{unknown attribute 'weakref' ignored}}

  [[carries_dependency]] ; // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
  [[carries_dependency]] { } // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
  [[carries_dependency]] if (0) { } // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
  [[carries_dependency]] for (;;); // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
  [[carries_dependency]] do { // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
    [[carries_dependency]] continue; // expected-error {{'carries_dependency' attribute cannot be applied to a statement}} ignored}}
  } while (0);
  [[carries_dependency]] while (0); // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}

  [[carries_dependency]] switch (i) { // expected-error {{'carries_dependency' attribute cannot be applied to a statement}} ignored}}
    [[carries_dependency]] case 0: // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
    [[carries_dependency]] default: // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
      [[carries_dependency]] break; // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
  }

  [[carries_dependency]] goto here; // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}

  [[carries_dependency]] try { // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}
  } catch (...) {
  }

  [[carries_dependency]] return; // expected-error {{'carries_dependency' attribute cannot be applied to a statement}}

  {
    [[ ]] // expected-error {{an attribute list cannot appear here}}
#pragma STDC FP_CONTRACT ON // expected-error {{can only appear at file scope or at the start of a compound statement}}
#pragma clang fp contract(fast) // expected-error {{can only appear at file scope or at the start of a compound statement}}
  }
}
