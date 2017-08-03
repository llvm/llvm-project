void extractEntireIfWhenSelectedBody(int x) {
  if (x == 1)
  {
    int z = x + 2;
  }
  else if (x == 3)
  {
    int y = x + 1;
  }
  // CHECK1: Initiated the 'extract' action at [[@LINE-8]]:3 -> [[@LINE-1]]:4
}

// RUN: clang-refactor-test initiate -action extract -selected=%s:3:3-5:4 -selected=%s:7:3-9:4 -selected=%s:4:5-8:19 -selected=%s:6:12-9:4 %s | FileCheck --check-prefix=CHECK1 %s

void extractEntireSourceConstructWhenSelectedBody(int x) {
  switch (x)
   {
  case 0:
    break;
  case 1:
    break;
  }
// CHECK2: Initiated the 'extract' action at [[@LINE-7]]:3 -> [[@LINE-1]]:4
// RUN: clang-refactor-test initiate -action extract -selected=%s:17:4-22:4 %s | FileCheck --check-prefix=CHECK2 %s

  for (int i = 0; i < x; ++i)
   {
    break;
  }
// CHECK3: Initiated the 'extract' action at [[@LINE-4]]:3 -> [[@LINE-1]]:4
// RUN: clang-refactor-test initiate -action extract -selected=%s:27:4-29:4 %s | FileCheck --check-prefix=CHECK3 %s

  while (x < 0)
   {
    break;
  }
// CHECK4: Initiated the 'extract' action at [[@LINE-4]]:3 -> [[@LINE-1]]:4
// RUN: clang-refactor-test initiate -action extract -selected=%s:34:4-36:4 %s | FileCheck --check-prefix=CHECK4 %s

  do
   {
    break;
  }
  while (x != 0);
// CHECK5: Initiated the 'extract' action at [[@LINE-5]]:3 -> [[@LINE-1]]:17
// RUN: clang-refactor-test initiate -action extract -selected=%s:41:4-43:4 %s | FileCheck --check-prefix=CHECK5 %s
}

void extractJustTheCompoundStatement() {
  {
    {
      int x = 0;
    }
  }
// CHECK6: Initiated the 'extract' action at [[@LINE-4]]:5 -> [[@LINE-2]]:6
// RUN: clang-refactor-test initiate -action extract -selected=%s:51:5-53:6 %s | FileCheck --check-prefix=CHECK6 %s
}

void extractSwitch(int x) {
  switch (x) {
    extractSwitch(x - 1);

    case 0:
      extractSwitch(x + 1);
      break;

    // comment
    case 1:
      extractSwitch(x + 2);
      break;

    default:
      extractSwitch(x + 2);
      break;

  }
// CHECK7: Initiated the 'extract' action at [[@LINE-17]]:3 -> [[@LINE-1]]:4
// RUN: clang-refactor-test initiate -action extract -selected=%s:61:3-64:27 -selected=%s:63:5-63:11 -selected=%s:68:5-69:27 -selected=%s:72:5-72:13 %s | FileCheck --check-prefix=CHECK7 %s
}

class AClass {
  void method();

  void extractWholeCallWhenJustMethodSelected() {
    method();
  }
};
// CHECK8: Initiated the 'extract' action at [[@LINE-3]]:5 -> [[@LINE-3]]:13
// RUN: clang-refactor-test initiate -action extract -selected=%s:85:5-85:6 -selected=%s:85:5-85:11 %s | FileCheck --check-prefix=CHECK8 %s

void extractWholeCallWhenJustMethodSelected() {
  AClass a;
  a.method();
}
// CHECK9: Initiated the 'extract' action at [[@LINE-2]]:3 -> [[@LINE-2]]:13
// RUN: clang-refactor-test initiate -action extract -selected=%s:93:3-93:7 -selected=%s:93:5-93:11 %s | FileCheck --check-prefix=CHECK9 %s
;
void avoidExtractingTooMuch(bool boolean) { // CHECK10: void extracted() {\nint x = 2;\n    // avoid-{{.*}}-end:+1:15\n    int y = x;\n}\n\n" [[@LINE]]:1
  if (boolean) {
    // avoid-too-much-begin:+1:1 // CHECK10: "extracted();" [[@LINE+1]]:5 -> [[@LINE+3]]:15
    int x = 2;
    // avoid-too-much-end:+1:15
    int y = x;
  } else {
    int z = 3;
  }

  // switch-casesel-begin: +4:3 // switch-casesel-end: +4:4 // CHECK10: void extracted(bool boolean) {\nswitch ((int)boolean) {\n case 0:\n avoidExtractingTooMuch(boolean);\n avoidExtractingTooMuch(boolean);\n break;\n }\n}
  // switch-case0-begin: +4:5 // switch-case0-end: +4:36 // CHECK10: void extracted(bool boolean) {\navoidExtractingTooMuch(boolean);\n}
  // switch-case1-begin: +3:5 // switch-case1-end: +4:36 // CHECK10: void extracted(bool boolean) {\navoidExtractingTooMuch(boolean);\n    avoidExtractingTooMuch(boolean);\n}\n\n"
  switch ((int)boolean) {
  case 0:
    avoidExtractingTooMuch(boolean);
    avoidExtractingTooMuch(boolean);
    break;
  }
  // CHECK10: "extracted(boolean)" [[@LINE-4]]:5 -> [[@LINE-3]]:36
}

// RUN: clang-refactor-test perform -action extract -selected=avoid-too-much -selected=switch-casesel -selected=switch-case0 -selected=switch-case1 %s | FileCheck --check-prefix=CHECK10 %s
