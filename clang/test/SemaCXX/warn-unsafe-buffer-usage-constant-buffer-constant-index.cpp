// RUN: %clang_cc1 -std=c++20 -Wno-everything -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -verify %s

void char_literal() {
  if ("abc"[2] == 'c')
    return;
  if ("def"[3] == '0')
    return;
}

void const_size_buffer_arithmetic() {
  char kBuf[64] = {};
  const char* p = kBuf + 1;
}

// expected-no-diagnostics
