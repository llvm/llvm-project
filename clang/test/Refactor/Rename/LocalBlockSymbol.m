__auto_type escaping1 = ^ {
  struct Local {  // LOCAL1: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:2:10 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=LOCAL1 %s

    int x;// ESCAPES1: rename [[@LINE]]
          // NOESCAPE1: rename local [[@LINE-1]]
// RUN: clang-refactor-test rename-initiate -at=%s:5:9 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=ESCAPES1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:5:9 -new-name=name %s -fblocks -x objective-c | FileCheck --check-prefix=NOESCAPE1 %s
  };
  struct Local result;
  return result;
};

__auto_type escaping2 = ^ () { // no prototype,
  struct Local {  // LOCAL2: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:15:10 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=LOCAL2 %s

    int x;// ESCAPES2: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:18:9 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=ESCAPES2 %s
  };
  struct Local result;
  return result;
};

__auto_type outer1 = ^ {
  __auto_type escaping3 = ^ (int x) { // prototype with some arguments.
    struct Local {  // LOCAL3: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:27:12 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=LOCAL3 %s

      int x;// ESCAPES3: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:30:11 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=ESCAPES3 %s
    };
    struct Local result;
    return result;
  };
  return escaping3(0);
};

void outer2() {
  __auto_type escaping1 = ^ {
    struct Local {
      int x;// LOCAL4: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:42:11 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=LOCAL4 %s
    };
    struct Local result;
    return result;
  };
}

__auto_type normalBlock = ^int (void) {
  struct Local {  // LOCAL5: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:51:10 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=LOCAL5 %s

    int x;// LOCAL6: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:54:9 -new-name=name %s -fblocks -x objective-c-header | FileCheck --check-prefix=LOCAL6 %s
  };
  return 0;
};
