// RUN: %check_clang_tidy %s cppcoreguidelines-narrowing-conversions %t -std=c++20 -- -Wno-everything

void narrow_integer_to_signed_integer_is_ok_since_cxx20() {
  char c;
  short s;
  int i;
  long l;
  long long ll;

  unsigned char uc;
  unsigned short us;
  unsigned int ui;
  unsigned long ul;
  unsigned long long ull;

  c = c;
  c = s;
  c = i;
  c = l;
  c = ll;

  c = uc;
  c = us;
  c = ui;
  c = ul;
  c = ull;

  i = c;
  i = s;
  i = i;
  i = l;
  i = ll;

  i = uc;
  i = us;
  i = ui;
  i = ul;
  i = ull;

  ll = c;
  ll = s;
  ll = i;
  ll = l;
  ll = ll;

  ll = uc;
  ll = us;
  ll = ui;
  ll = ul;
  ll = ull;
}

void narrow_constant_to_signed_integer_is_ok_since_cxx20() {
  char c1 = -128;
  char c2 = 127;
  char c3 = -129;
  char c4 = 128;

  short s1 = -32768;
  short s2 = 32767;
  short s3 = -32769;
  short s4 = 32768;
}
