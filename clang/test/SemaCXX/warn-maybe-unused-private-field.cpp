// RUN: %clang_cc1 -fsyntax-only -Wunused-private-field -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -Wunused-private-field -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -Wunused-private-field -verify -std=c++23 %s

class MyClass {
	// Marking an unused field with [[maybe_unused]] shouldn't result in a
	// warning
	[[maybe_unused]]
	unsigned field1;
	signed field2; // expected-warning{{private field 'field2' is not used}}
};
