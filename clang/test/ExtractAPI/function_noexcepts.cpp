// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api -triple arm64-apple-macosx \
// RUN:   -x c++-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
void getFoo() noexcept;

void getBar() noexcept(true);

void getFooBar() noexcept(false);
/// expected-no-diagnostics

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
    "name": "",
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
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "getFoo"
        },
        {
          "kind": "text",
          "spelling": "()"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "keyword",
          "spelling": "noexcept"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@F@getFoo#"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c++.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 0
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "getFoo"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "getFoo"
          }
        ],
        "title": "getFoo"
      },
      "pathComponents": [
        "getFoo"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "getBar"
        },
        {
          "kind": "text",
          "spelling": "()"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "keyword",
          "spelling": "noexcept"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "keyword",
          "spelling": "true"
        },
        {
          "kind": "text",
          "spelling": ");"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@F@getBar#"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c++.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "getBar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "getBar"
          }
        ],
        "title": "getBar"
      },
      "pathComponents": [
        "getBar"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "getFooBar"
        },
        {
          "kind": "text",
          "spelling": "()"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "keyword",
          "spelling": "noexcept"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "keyword",
          "spelling": "false"
        },
        {
          "kind": "text",
          "spelling": ");"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@F@getFooBar#"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c++.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "getFooBar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "getFooBar"
          }
        ],
        "title": "getFooBar"
      },
      "pathComponents": [
        "getFooBar"
      ]
    }
  ]
}
