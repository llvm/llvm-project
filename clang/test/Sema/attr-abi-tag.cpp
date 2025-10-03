// RUN: %clang_cc1 -verify -fsyntax-only %s

[[gnu::abi_tag("")]]    void f1();  // expected-error {{ABI tag cannot be empty}}
[[gnu::abi_tag("9A")]]  void f2();  // expected-error {{character '9' is not allowed at the start of an ABI tag}}
[[gnu::abi_tag("0")]]   void f3();  // expected-error {{character '0' is not allowed at the start of an ABI tag}}
[[gnu::abi_tag("猫A")]] void f4();  // expected-error {{character '猫' is not allowed in ABI tags}}
[[gnu::abi_tag("A𨭎")]] void f5();  // expected-error {{character '𨭎' is not allowed in ABI tags}}
[[gnu::abi_tag("AB")]]  void f6();
[[gnu::abi_tag("A1")]]  void f7();
