// Test the -cc1 flag. There's no -fno-... option in -cc1 invocations,
// just the positive option.

// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -verify=OFF %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -verify=ON %s \
// RUN:            -fsafe-buffer-usage-suggestions

// Test driver flags. Now there's both -f... and -fno-... to worry about.

// RUN: %clang -fsyntax-only -std=c++20 -Wunsafe-buffer-usage \
// RUN:        -Xclang -verify=OFF %s
// RUN: %clang -fsyntax-only -std=c++20 -Wunsafe-buffer-usage \
// RUN:        -fsafe-buffer-usage-suggestions \
// RUN:        -Xclang -verify=ON %s
// RUN: %clang -fsyntax-only -std=c++20 -Wunsafe-buffer-usage \
// RUN:        -fno-safe-buffer-usage-suggestions \
// RUN:        -Xclang -verify=OFF %s

// In case of driver flags, last flag takes precedence.

// RUN: %clang -fsyntax-only -std=c++20 -Wunsafe-buffer-usage \
// RUN:        -fsafe-buffer-usage-suggestions \
// RUN:        -fno-safe-buffer-usage-suggestions \
// RUN:        -Xclang -verify=OFF %s
// RUN: %clang -fsyntax-only -std=c++20 -Wunsafe-buffer-usage \
// RUN:        -fno-safe-buffer-usage-suggestions \
// RUN:        -fsafe-buffer-usage-suggestions \
// RUN:        -Xclang -verify=ON %s

// Passing through -Xclang.

// RUN: %clang -fsyntax-only -std=c++20 -Wunsafe-buffer-usage \
// RUN:        -Xclang -fsafe-buffer-usage-suggestions \
// RUN:        -Xclang -verify=ON %s

// -Xclang flags take precedence over driver flags.

// RUN: %clang -fsyntax-only -std=c++20 -Wunsafe-buffer-usage \
// RUN:        -Xclang -fsafe-buffer-usage-suggestions \
// RUN:        -fno-safe-buffer-usage-suggestions \
// RUN:        -Xclang -verify=ON %s
// RUN: %clang -fsyntax-only -std=c++20 -Wunsafe-buffer-usage \
// RUN:        -fno-safe-buffer-usage-suggestions \
// RUN:        -Xclang -fsafe-buffer-usage-suggestions \
// RUN:        -Xclang -verify=ON %s

[[clang::unsafe_buffer_usage]] void bar(int *);

void foo(int *x) { // \
  // ON-warning{{'x' is an unsafe pointer used for buffer access}}
  // FIXME: Better "OFF" warning?
  x[5] = 10; // \
  // ON-note    {{used in buffer access here}} \
  // OFF-warning{{unsafe buffer access}} \
  // OFF-note   {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

  x += 5;  // \
  // ON-note    {{used in pointer arithmetic here}} \
  // OFF-warning{{unsafe pointer arithmetic}} \
  // OFF-note   {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

  bar(x);  // \
  // ON-warning{{function introduces unsafe buffer manipulation}} \
  // OFF-warning{{function introduces unsafe buffer manipulation}} \
  // OFF-note   {{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
}
