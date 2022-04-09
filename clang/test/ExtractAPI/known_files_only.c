// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%/t@g" %t/reference.output.json.in >> \
// RUN: %t/reference.output.json
// RUN: %clang -extract-api --product-name=GlobalRecord -target arm64-apple-macosx \
// RUN: %t/input1.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input1.h
int num;
#include "input2.h"

//--- input2.h
// Ensure that these symbols are not emitted in the Symbol Graph.
#define HELLO 1
char not_emitted;
void foo(int);
struct Foo { int a; };

//--- reference.output.json.in
{
  "metadata": {
    "formatVersion": {
      "major": 0,
      "minor": 5,
      "patch": 3
    },
    "generator": "?"
  },
  "module": {
    "name": "GlobalRecord",
    "platform": {
      "architecture": "arm64",
      "operatingSystem": {
        "minimumVersion": {
          "major": 11,
          "minor": 0,
          "patch": 0
        },
        "name": "macosx"
      },
      "vendor": "apple"
    }
  },
  "relationships": [],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "num"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@num"
      },
      "kind": {
        "displayName": "Global Variable",
        "identifier": "c.var"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input1.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "num"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "num"
          }
        ],
        "title": "num"
      },
      "pathComponents": [
        "num"
      ]
    }
  ]
}
