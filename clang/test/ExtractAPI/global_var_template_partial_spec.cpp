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
template<typename X, typename Y> int Foo = 0;

template<typename Z> int Foo<int, Z> = 0;
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
          "kind": "keyword",
          "spelling": "template"
        },
        {
          "kind": "text",
          "spelling": "<"
        },
        {
          "kind": "keyword",
          "spelling": "typename"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "genericParameter",
          "spelling": "X"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "keyword",
          "spelling": "typename"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "genericParameter",
          "spelling": "Y"
        },
        {
          "kind": "text",
          "spelling": "> "
        },
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
          "spelling": "Foo"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@Foo"
      },
      "kind": {
        "displayName": "Global Variable Template",
        "identifier": "c++.var"
      },
      "location": {
        "position": {
          "character": 38,
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
      ],
      "swiftGenerics": {
        "parameters": [
          {
            "depth": 0,
            "index": 0,
            "name": "X"
          },
          {
            "depth": 0,
            "index": 1,
            "name": "Y"
          }
        ]
      }
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "template"
        },
        {
          "kind": "text",
          "spelling": "<"
        },
        {
          "kind": "keyword",
          "spelling": "typename"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "genericParameter",
          "spelling": "Z"
        },
        {
          "kind": "text",
          "spelling": "> "
        },
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
          "spelling": "Foo"
        },
        {
          "kind": "text",
          "spelling": "<"
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:t0.0",
          "spelling": "Z"
        },
        {
          "kind": "text",
          "spelling": ">;"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c++",
        "precise": "c:@VP>1#T@Foo>#I#t0.0"
      },
      "kind": {
        "displayName": "Global Variable Template Partial Specialization",
        "identifier": "c++.var"
      },
      "location": {
        "position": {
          "character": 26,
          "line": 3
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
      ],
      "swiftGenerics": {
        "parameters": [
          {
            "depth": 0,
            "index": 0,
            "name": "Z"
          }
        ]
      }
    }
  ]
}
