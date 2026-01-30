// RUN: %clang_cc1 -std=c23 -triple x86_64 -verify -fsyntax-only -Wfortify-source -Wno-string-plus-int %s
// RUN: %clang_cc1 -std=c23 -triple x86_64 -verify -fsyntax-only -Wfortify-source -Wno-string-plus-int -fexperimental-new-constant-interpreter %s

void test(char *c) {
  // We test offsets 0 to 4.
  // 0: a no op of course
  // 1,2: these result in a different length string than the buffer size
  // 3: the last position: this hits ptr+object_size being a valid pointer,
  //    but not dereferencable
  // 4: completely invalid pointer
  __builtin_strcat(c, "42" + 0);
  __builtin_strcat(c, "42" + 1);
  __builtin_strcat(c, "42" + 2);
  __builtin_strcat(c, "42" + 3);
  __builtin_strcat(c, "42" + 4);
  _Static_assert(__builtin_strlen("42" + 0) == 2);

  // A test without a null terminator, this captures incorrect size computation
  // and incorrectly specifying the buffer size to strlen.
  char buffer[1];
  static const char test_buffer[] = {'4','2'};
  __builtin_strcpy(buffer, test_buffer + 0);
  __builtin_strcpy(buffer, test_buffer + 1);
  // Note: these show that we will not issue a fortify warning when the source
  // buffer is not null terminated.
  __builtin_strcpy(buffer, test_buffer + 2);
  __builtin_strcpy(buffer, test_buffer + 3);

  // Verifying strlen computes from the correct starting point.
  _Static_assert(__builtin_strlen("42" + 1) == 1);
  _Static_assert(__builtin_strlen("42" + 2) == 0);
  _Static_assert(__builtin_strlen("42" + 3));
  // expected-error@-1 {{static assertion expression is not an integral constant expression}}
  _Static_assert(__builtin_strlen("42" + 4));
  // expected-error@-1 {{static assertion expression is not an integral constant expression}}
  // expected-note@-2 {{cannot refer to element 4 of array of 3 elements in a constant expression}}
}
