// RUN: %clang_cc1 %s -fsyntax-only --embed-dir=%S/Inputs -verify -std=c23
// RUN: %clang_cc1 %s -fsyntax-only --embed-dir=%S/Inputs -verify -std=c23 -fexperimental-new-constant-interpreter

static constexpr unsigned char data[] = {
#embed "big_char.txt"
};

static constexpr char data1[] = {
#embed "big_char.txt" // expected-error {{constexpr initializer evaluates to 255 which is not exactly representable in type 'const char'}}
};

static constexpr int data2[] = {
#embed "big_char.txt"
};

static constexpr unsigned data3[] = {
#embed "big_char.txt" suffix(, -1) // expected-error {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned int'}}
};

static constexpr int data4[] = {
#embed "big_char.txt" suffix(, -1)
};

static constexpr float data5[] = {
#embed "big_char.txt" suffix(, -1)
};
