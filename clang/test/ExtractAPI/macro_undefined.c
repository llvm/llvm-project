// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang -extract-api --product-name=Macros -target arm64-apple-macosx \
// RUN: -x objective-c-header %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
#define HELLO 1
#define FUNC_GEN(NAME, ...) void NAME(__VA_ARGS__);
FUNC_GEN(foo)
FUNC_GEN(bar, const int *, unsigned);
#undef FUNC_GEN
// Undefining a not previously defined macro should not result in a crash.
#undef FOO

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
    "name": "Macros",
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
          "spelling": "foo"
        },
        {
          "kind": "text",
          "spelling": "()"
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
        "interfaceLanguage": "objective-c",
        "precise": "c:@F@foo"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "objective-c.func"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "foo"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "foo"
          }
        ],
        "title": "foo"
      },
      "pathComponents": [
        "foo"
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
          "spelling": "bar"
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
          "spelling": " * "
        },
        {
          "kind": "internalParam",
          "spelling": ""
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "internalParam",
          "spelling": ""
        },
        {
          "kind": "text",
          "spelling": ")"
        }
      ],
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
                "spelling": " * "
              },
              {
                "kind": "internalParam",
                "spelling": ""
              }
            ],
            "name": ""
          },
          {
            "declarationFragments": [
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:i",
                "spelling": "unsigned int"
              },
              {
                "kind": "text",
                "spelling": " "
              },
              {
                "kind": "internalParam",
                "spelling": ""
              }
            ],
            "name": ""
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
        "interfaceLanguage": "objective-c",
        "precise": "c:@F@bar"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "objective-c.func"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "bar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "bar"
          }
        ],
        "title": "bar"
      },
      "pathComponents": [
        "bar"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "#define"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "HELLO"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@8@macro@HELLO"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "objective-c.macro"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "HELLO"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "HELLO"
          }
        ],
        "title": "HELLO"
      },
      "pathComponents": [
        "HELLO"
      ]
    }
  ]
}
