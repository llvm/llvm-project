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
class Foo {
  template<typename T> void Bar(T Fizz);

  template<> void Bar<int>(int Fizz);
};

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
  "relationships": [
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@FT@>1#TBar#t0.0#v#",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Foo@F@Bar<#I>#I#",
      "target": "c:@S@Foo",
      "targetFallback": "Foo"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "class"
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
        "precise": "c:@S@Foo"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "c++.class"
      },
      "location": {
        "position": {
          "character": 7,
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
      "accessLevel": "private",
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
          "spelling": "T"
        },
        {
          "kind": "text",
          "spelling": "> "
        },
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
          "spelling": "Bar"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:t0.0",
          "spelling": "T"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "internalParam",
          "spelling": "Fizz"
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
                "preciseIdentifier": "c:t0.0",
                "spelling": "T"
              },
              {
                "kind": "text",
                "spelling": " "
              },
              {
                "kind": "internalParam",
                "spelling": "Fizz"
              }
            ],
            "name": "Fizz"
          }
        ],
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
        "precise": "c:@S@Foo@FT@>1#TBar#t0.0#v#"
      },
      "kind": {
        "displayName": "Method Template",
        "identifier": "c++.method"
      },
      "location": {
        "position": {
          "character": 29,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Bar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Bar"
          }
        ],
        "title": "Bar"
      },
      "pathComponents": [
        "Foo",
        "Bar"
      ],
      "swiftGenerics": {
        "parameters": [
          {
            "depth": 0,
            "index": 0,
            "name": "T"
          }
        ]
      }
    },
    {
      "accessLevel": "private",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "template"
        },
        {
          "kind": "text",
          "spelling": "<> "
        },
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
          "spelling": "Bar"
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
          "spelling": ">("
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
          "kind": "internalParam",
          "spelling": "Fizz"
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
                "preciseIdentifier": "c:I",
                "spelling": "int"
              },
              {
                "kind": "text",
                "spelling": " "
              },
              {
                "kind": "internalParam",
                "spelling": "Fizz"
              }
            ],
            "name": "Fizz"
          }
        ],
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
        "precise": "c:@S@Foo@F@Bar<#I>#I#"
      },
      "kind": {
        "displayName": "Method Template Specialization",
        "identifier": "c++.method"
      },
      "location": {
        "position": {
          "character": 19,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Bar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Bar"
          }
        ],
        "title": "Bar"
      },
      "pathComponents": [
        "Foo",
        "Bar"
      ]
    }
  ]
}
