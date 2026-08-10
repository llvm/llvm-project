// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s -triple x86_64-pc-win32

#pragma const_seg(".my_const") // expected-note 2 {{#pragma entered here}}
extern const int a;
const int a = 1; // expected-note 2 {{declared here}}
#pragma data_seg(".my_const") // expected-note {{#pragma entered here}}
int b = 1; // expected-error {{'b' causes a section type conflict with 'a'}}
#pragma data_seg()
int c = 1;
__declspec(allocate(".my_const")) int d = 1; // expected-error {{'d' causes a section type conflict with 'a'}}
#pragma data_seg("\u") // expected-error {{\u used with no following hex digits}}
#pragma data_seg("a" L"b") // expected-warning {{expected non-wide string literal in '#pragma data_seg'}}

#pragma section(".my_seg", execute) // expected-note 2 {{#pragma entered her}}
__declspec(allocate(".my_seg")) int int_my_seg;
#pragma code_seg(".my_seg")
void fn_my_seg(void){}

__declspec(allocate(".bad_seg")) int int_bad_seg = 1; // expected-note {{declared here}}
#pragma code_seg(".bad_seg") // expected-note {{#pragma entered here}}
void fn_bad_seg(void){} // expected-error {{'fn_bad_seg' causes a section type conflict with 'int_bad_seg'}}

#pragma bss_seg // expected-warning {{missing '(' after '#pragma bss_seg' - ignoring}}
#pragma bss_seg(L".my_seg") // expected-warning {{expected push, pop or a string literal for the section name in '#pragma bss_seg' - ignored}}
#pragma bss_seg(1) // expected-warning {{expected push, pop or a string literal for the section name in '#pragma bss_seg' - ignored}}
#pragma bss_seg(push)
#pragma bss_seg(push, ".my_seg")
#pragma bss_seg(push, 1) // expected-warning {{expected a stack label or a string literal for the section name in '#pragma bss_seg'}}
#pragma bss_seg ".my_seg" // expected-warning {{missing '(' after '#pragma bss_seg' - ignoring}}
#pragma bss_seg(push, my_label, 1) // expected-warning {{expected a string literal for the section name in '#pragma bss_seg' - ignored}}
#pragma bss_seg(".my_seg", 1) // expected-warning {{missing ')' after '#pragma bss_seg' - ignoring}}
#pragma bss_seg(".my_seg" // expected-warning {{missing ')' after '#pragma bss_seg' - ignoring}}

#pragma section // expected-warning {{missing '(' after '#pragma section' - ignoring}}
#pragma section( // expected-warning {{expected a string literal for the section name in '#pragma section' - ignored}}
#pragma section(L".my_seg") // expected-warning {{expected a string literal for the section name in '#pragma section' - ignored}}
#pragma section(".my_seg" // expected-warning {{missing ')' after '#pragma section' - ignoring}}
#pragma section(".my_seg" 1  // expected-warning {{missing ')' after '#pragma section' - ignoring}}
#pragma section(".my_seg",  // expected-warning {{expected action or ')' in '#pragma section' - ignored}}
#pragma section(".my_seg", read) // expected-error {{this causes a section type conflict with a prior #pragma section}}
#pragma section(".my_seg", bogus) // expected-warning {{unknown action 'bogus' for '#pragma section' - ignored}}
#pragma section(".my_seg", nopage) // expected-warning {{known but unsupported action 'nopage' for '#pragma section' - ignored}}
#pragma section(".my_seg", read, write) // expected-error {{this causes a section type conflict with a prior #pragma section}}
#pragma section(".my_seg", read, write, 1) //  expected-warning {{expected action or ')' in '#pragma section' - ignored}}

#pragma bss_seg(".drectve") // expected-warning{{#pragma bss_seg(".drectve") has undefined behavior, use #pragma comment(linker, ...) instead}}
#pragma code_seg(".drectve") // expected-warning{{#pragma code_seg(".drectve") has undefined behavior, use #pragma comment(linker, ...) instead}} 
#pragma const_seg(".drectve")  // expected-warning{{#pragma const_seg(".drectve") has undefined behavior, use #pragma comment(linker, ...) instead}}
#pragma data_seg(".drectve")  // expected-warning{{#pragma data_seg(".drectve") has undefined behavior, use #pragma comment(linker, ...) instead}}
#pragma code_seg(".my_seg")

// cl.exe doesn't warn on this, so match that.
// (Technically it ICEs on this particular example, but if it's on a class then
// it just doesn't warn.)
__declspec(code_seg(".drectve")) void drectve_fn(void) {}

// This shouldn't warn; mingw users likely want to use
//     __attribute__((section(".drective")))
//     const char LinkerFlags[] = "-export:foo -export:bar";
// for interop with gcc.
__attribute__((section(".drectve"))) int drectve_int;
