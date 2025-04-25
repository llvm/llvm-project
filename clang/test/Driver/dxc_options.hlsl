// RUN: %clang_dxc \
// RUN: -fcolor-diagnostics \
// RUN: -fno-color-diagnostics \
// RUN: -fdiagnostics-color \
// RUN: -fno-diagnostics-color \
// RUN: -fdiagnostics-color=auto \
// RUN: -Tlib_6_7 -fdriver-only -- %s 2>&1 |count 0
