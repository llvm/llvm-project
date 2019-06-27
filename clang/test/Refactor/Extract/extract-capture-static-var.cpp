

void captureStaticVars() {
  static int x;
  int y = x;
  x += 1;
// CHECK1: extracted(int x) {\nint y = x;\n}\n\n"
// CHECK1: extracted(int &x) {\nx += 1;\n}
// CHECK1: extracted() {\nstatic int x;\n  int y = x;\n  x += 1;\n}\n\n"
}

// RUN: clang-refactor-test perform -action extract -selected=%s:5:3-5:12 -selected=%s:6:3-6:9 -selected=%s:4:3-6:9 %s | FileCheck --check-prefix=CHECK1 %s
