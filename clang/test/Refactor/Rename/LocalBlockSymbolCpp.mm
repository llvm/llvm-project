auto escaping1 = ^ {
  struct Global {  // ESCAPES1: rename [[@LINE]]
                   // NOESCAPE1: rename local [[@LINE-1]]
// RUN: clang-refactor-test rename-initiate -at=%s:2:10 -new-name=name %s -std=c++14 -fblocks -x objective-c++-header | FileCheck --check-prefix=ESCAPES1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:2:10 -new-name=name %s -std=c++14 -fblocks -x objective-c++ | FileCheck --check-prefix=NOESCAPE1 %s
  };
  return Global();
};

void outer1() {
  auto escaping1 = ^ {
    struct Local {  // LOCAL1: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:12:12 -new-name=name %s -std=c++14 -fblocks -x objective-c++-header | FileCheck --check-prefix=LOCAL1 %s
    };
    return Local();
  };
}

auto normalBlock = ^int () {
  struct Local {  // LOCAL2: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:20:10 -new-name=name %s -std=c++14 -fblocks -x objective-c++-header | FileCheck --check-prefix=LOCAL2 %s
  };
  return 0;
};
