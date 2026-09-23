// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-compatibility \
// RUN:   -Werror=microsoft-enum-typedef -verify=werror %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-compatibility \
// RUN:   -Wno-microsoft-enum-typedef -verify=nowarn %s

// nowarn-no-diagnostics

typedef enum Underlying { Value } Alias;

struct Use {
  enum Alias *member; // werror-error {{using an 'enum' specifier with a typedef name is a Microsoft extension}}
};
