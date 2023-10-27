// RUN: %clang_analyze_cc1 -analyzer-checker=optin.portability.UnixAPI \
// RUN:    -triple x86_64-pc-linux-gnu -x c %s

// Don't crash!
// expected-no-diagnostics
const __int128_t a = ( (__int128_t)1 << 64 );
const _BitInt(72) b = ( 1 << 72 );

void int128() {
  2 >> a;
}

void withbitint() {
  2 >> b;
}
