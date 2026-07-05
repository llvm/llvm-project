// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -Wsign-conversion -verify=unsigned,both %s
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -Wconversion -verify=unsigned,both %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -verify -Wsign-conversion -verify=win32,both %s

// PR35200
enum X { A,B,C};
int f(enum X x) {
  return x; // unsigned-warning {{implicit conversion changes signedness: 'enum X' to 'int'}}
}

enum SE1 { N1 = -1 }; // Always a signed underlying type.
enum E1 { P1 };       // Unsigned underlying type except on Windows.

// ensure no regression with enum to sign (related to enum-enum-conversion.c)
int f1(enum E1 E) {
  return E; // unsigned-warning {{implicit conversion changes signedness: 'enum E1' to 'int'}}
}

enum E1 f2(int E) {
  return E; // unsigned-warning {{implicit conversion changes signedness: 'int' to 'enum E1'}}
}

int f3(enum SE1 E) {
  return E; // shouldn't warn
}

enum SE1 f4(int E) {
  return E; // shouldn't warn
}

unsigned f5(enum E1 E) {
  return E; // win32-warning {{implicit conversion changes signedness: 'enum E1' to 'unsigned int'}}
}

enum E1 f6(unsigned E) {
  return E; // win32-warning {{implicit conversion changes signedness: 'unsigned int' to 'enum E1'}}
}

unsigned f7(enum SE1 E) {
  return E; // both-warning {{implicit conversion changes signedness: 'enum SE1' to 'unsigned int'}}
}

enum SE1 f8(unsigned E) {
  return E; // both-warning {{implicit conversion changes signedness: 'unsigned int' to 'enum SE1'}}
}
