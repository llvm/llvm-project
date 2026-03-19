// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api --product-name=BoolTest --pretty-sgf -triple arm64-apple-macosx \
// RUN:   %t/input.h -o %t/output.json

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
#include <stdbool.h>
bool Foo;

bool IsFoo(bool Bar);
// expected-no-diagnostics

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
    "name": "BoolTest",
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
          "preciseIdentifier": "c:b",
          "spelling": "bool"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Foo"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@Foo"
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
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Foo"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Foo"
          }
        ],
        "title": "Foo"
      },
      "pathComponents": [
        "Foo"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:b",
          "spelling": "bool"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "IsFoo"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:b",
          "spelling": "bool"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "internalParam",
          "spelling": "Bar"
        },
        {
          "kind": "text",
          "spelling": ");"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:b",
                "spelling": "bool"
              },
              {
                "kind": "text",
                "spelling": " "
              },
              {
                "kind": "internalParam",
                "spelling": "Bar"
              }
            ],
            "name": "Bar"
          }
        ],
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:b",
            "spelling": "bool"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@F@IsFoo"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "IsFoo"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "IsFoo"
          }
        ],
        "title": "IsFoo"
      },
      "pathComponents": [
        "IsFoo"
      ]
    }
  ]
}
