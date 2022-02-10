// RUN: %clang_cc1 %s -verify -fno-builtin

#define _zero_call_used_regs(...) __attribute__((zero_call_used_regs(__VA_ARGS__)))

void failure() _zero_call_used_regs();                   // expected-error {{takes one argument}}
void failure() _zero_call_used_regs("used", "used-gpr"); // expected-error {{takes one argument}}
void failure() _zero_call_used_regs(0);                  // expected-error {{requires a string}}
void failure() _zero_call_used_regs("hello");            // expected-warning {{argument not supported: hello}}

void success() _zero_call_used_regs("skip");
void success() _zero_call_used_regs("used-gpr-arg");
void success() _zero_call_used_regs("used-gpr");
void success() _zero_call_used_regs("used-arg");
void success() _zero_call_used_regs("used");
void success() _zero_call_used_regs("all-gpr-arg");
void success() _zero_call_used_regs("all-gpr");
void success() _zero_call_used_regs("all-arg");
void success() _zero_call_used_regs("all");
