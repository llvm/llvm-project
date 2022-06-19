// RUN: %clang_cc1 -std=c17 -fsyntax-only -verify=ext -Wno-unused %s
// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify=compat -Wpre-c2x-compat -Wno-unused %s
// RUN: %clang_cc1 -fsyntax-only -verify=cpp -Wno-unused -x c++ %s

#if 18446744073709551615uwb // ext-warning {{'_BitInt' suffix for literals is a C2x extension}} \
                               compat-warning {{'_BitInt' suffix for literals is incompatible with C standards before C2x}} \
                               cpp-error {{invalid suffix 'uwb' on integer constant}}
#endif

void func(void) {
  18446744073709551615wb; // ext-warning {{'_BitInt' suffix for literals is a C2x extension}} \
                             compat-warning {{'_BitInt' suffix for literals is incompatible with C standards before C2x}} \
                             cpp-error {{invalid suffix 'wb' on integer constant}}
}
