// RUN: %clang_cc1 -triple s390x-none-zos -fzos-extensions %s -verify

#pragma csect(code, "abc")
#pragma csect(code, "csectcode") // expected-warning {{duplicate '#pragma csect' is ignored}}
#pragma csect(static, "abc") // expected-error {{external name 'abc' in pragma csect conflicts with another csect name}}

#pragma csect(static, "csectstatic")
#pragma csect(test,"csecttest")
#pragma csect(xyz,"csectxyz") // expected-warning {{expected identifier in '#pragma csect' - ignored}}

int main() { return 0; }

void def() {}
