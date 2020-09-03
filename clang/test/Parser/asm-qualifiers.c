// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify %s

void qualifiers(void) {
  asm("");
  asm volatile("");
  asm inline("");
  asm goto("" ::::foo);
foo:;
}

void unknown_qualifiers(void) {
  asm noodle(""); // expected-error {{expected 'volatile', 'inline', 'goto', or '('}}
  asm goto noodle("" ::::foo); // expected-error {{expected 'volatile', 'inline', 'goto', or '('}}
  asm volatile noodle inline(""); // expected-error {{expected 'volatile', 'inline', 'goto', or '('}}
foo:;
}

void underscores(void) {
  __asm__("");
  __asm__ __volatile__("");
  __asm__ __inline__("");
  // Note: goto is not supported with underscore prefix+suffix.
  __asm__ goto("" ::::foo);
foo:;
}

void permutations(void) {
  asm goto inline volatile("" ::::foo);
  asm goto inline("");
  asm goto volatile inline("" ::::foo);
  asm goto volatile("");
  asm inline goto volatile("" ::::foo);
  asm inline goto("" ::::foo);
  asm inline volatile goto("" ::::foo);
  asm inline volatile("");
  asm volatile goto("" ::::foo);
  asm volatile inline goto("" ::::foo);
  asm volatile inline("");
foo:;
}

void duplicates(void) {
  asm volatile volatile("");             // expected-error {{duplicate asm qualifier 'volatile'}}
  __asm__ __volatile__ __volatile__(""); // expected-error {{duplicate asm qualifier 'volatile'}}
  asm inline inline("");                 // expected-error {{duplicate asm qualifier 'inline'}}
  __asm__ __inline__ __inline__("");     // expected-error {{duplicate asm qualifier 'inline'}}
  asm goto goto("" ::::foo);             // expected-error {{duplicate asm qualifier 'goto'}}
  __asm__ goto goto("" ::::foo);         // expected-error {{duplicate asm qualifier 'goto'}}
foo:;
}

// globals
asm ("");
// <rdar://problem/7574870>
asm volatile (""); // expected-error {{meaningless 'volatile' on asm outside function}}
asm inline (""); // expected-error {{meaningless 'inline' on asm outside function}}
asm goto (""::::noodle); // expected-error {{meaningless 'goto' on asm outside function}}
// expected-error@-1 {{expected ')'}}
// expected-note@-2 {{to match this '('}}
