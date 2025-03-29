// RUN: %clang_cc1 %s -verify -fno-builtin

#define _zero_call_used_regs(...) __attribute__((zero_call_used_regs(__VA_ARGS__)))

void failure1(void) _zero_call_used_regs();                   // expected-error {{takes one argument}}
void failure2(void) _zero_call_used_regs("used", "used-gpr"); // expected-error {{takes one argument}}
void failure3(void) _zero_call_used_regs(0);                  // expected-error {{expected string literal}}
void failure4(void) _zero_call_used_regs("hello");            // expected-warning {{argument not supported: hello}}

void success1(void) _zero_call_used_regs("skip");
void success2(void) _zero_call_used_regs("used-gpr-arg");
void success3(void) _zero_call_used_regs("used-gpr");
void success4(void) _zero_call_used_regs("used-arg");
void success5(void) _zero_call_used_regs("used");
void success6(void) _zero_call_used_regs("all-gpr-arg");
void success7(void) _zero_call_used_regs("all-gpr");
void success8(void) _zero_call_used_regs("all-arg");
void success9(void) _zero_call_used_regs("all");
