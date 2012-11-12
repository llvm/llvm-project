// RUN: %clang_cc1 -fsyntax-only -fmessage-length 100 %s 2>&1 | FileCheck -strict-whitespace %s
// REQUIRES: asserts

int main() {
    "╔#x#p)6╥)╤╜К$√Ю>U ъh╤№├Ў|Я рже╧╗gЯY|`?Є;;╞┐Vj╟\\∙АЗ√▌кW9·╨в:╠КOаE°█Рy?SKкyж╣З╪рi&n";
}

// CHECK-NOT:Assertion
