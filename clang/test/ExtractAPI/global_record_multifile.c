// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang -extract-api --product-name=GlobalRecord -target arm64-apple-macosx \
// RUN: %t/input1.h %t/input2.h %t/input3.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input1.h
int num;

//--- input2.h
/**
 * \brief Add two numbers.
 * \param [in]  x   A number.
 * \param [in]  y   Another number.
 * \param [out] res The result of x + y.
 */
void add(const int x, const int y, int *res);

//--- input3.h
char unavailable __attribute__((unavailable));

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
        },
        {
          "kind": "text",
          "spelling": ";"
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
          "spelling": "add"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "keyword",
          "spelling": "const"
        },
        {
          "kind": "text",
          "spelling": " "
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
          "spelling": "x"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "keyword",
          "spelling": "const"
        },
        {
          "kind": "text",
          "spelling": " "
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
          "spelling": "y"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " * "
        },
        {
          "kind": "internalParam",
          "spelling": "res"
        },
        {
          "kind": "text",
          "spelling": ");"
        }
      ],
      "docComment": {
        "lines": [
          {
            "range": {
              "end": {
                "character": 4,
                "line": 1
              },
              "start": {
                "character": 4,
                "line": 1
              }
            },
            "text": ""
          },
          {
            "range": {
              "end": {
                "character": 27,
                "line": 2
              },
              "start": {
                "character": 3,
                "line": 2
              }
            },
            "text": " \\brief Add two numbers."
          },
          {
            "range": {
              "end": {
                "character": 30,
                "line": 3
              },
              "start": {
                "character": 3,
                "line": 3
              }
            },
            "text": " \\param [in]  x   A number."
          },
          {
            "range": {
              "end": {
                "character": 36,
                "line": 4
              },
              "start": {
                "character": 3,
                "line": 4
              }
            },
            "text": " \\param [in]  y   Another number."
          },
          {
            "range": {
              "end": {
                "character": 41,
                "line": 5
              },
              "start": {
                "character": 3,
                "line": 5
              }
            },
            "text": " \\param [out] res The result of x + y."
          },
          {
            "range": {
              "end": {
                "character": 4,
                "line": 6
              },
              "start": {
                "character": 1,
                "line": 6
              }
            },
            "text": " "
          }
        ]
      },
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "keyword",
                "spelling": "const"
              },
              {
                "kind": "text",
                "spelling": " "
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
                "spelling": "x"
              }
            ],
            "name": "x"
          },
          {
            "declarationFragments": [
              {
                "kind": "keyword",
                "spelling": "const"
              },
              {
                "kind": "text",
                "spelling": " "
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
                "spelling": "y"
              }
            ],
            "name": "y"
          },
          {
            "declarationFragments": [
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:I",
                "spelling": "int"
              },
              {
                "kind": "text",
                "spelling": " * "
              },
              {
                "kind": "internalParam",
                "spelling": "res"
              }
            ],
            "name": "res"
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
        "interfaceLanguage": "c",
        "precise": "c:@F@add"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input2.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "add"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "add"
          }
        ],
        "title": "add"
      },
      "pathComponents": [
        "add"
      ]
    }
  ]
}
