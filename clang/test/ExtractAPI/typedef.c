// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang -extract-api --product-name=Typedef -target arm64-apple-macosx \
// RUN: -x objective-c-header %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
typedef int MyInt;

typedef struct Bar *BarPtr;

void foo(BarPtr value);

void baz(BarPtr *value);

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
    "name": "Typedef",
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
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:input.h@T@BarPtr",
          "spelling": "BarPtr"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "internalParam",
          "spelling": "value"
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
                "preciseIdentifier": "c:input.h@T@BarPtr",
                "spelling": "BarPtr"
              },
              {
                "kind": "text",
                "spelling": " "
              },
              {
                "kind": "internalParam",
                "spelling": "value"
              }
            ],
            "name": "value"
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
        "precise": "c:@F@foo"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "objective-c.func"
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
          "spelling": "baz"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:input.h@T@BarPtr",
          "spelling": "BarPtr"
        },
        {
          "kind": "text",
          "spelling": " * "
        },
        {
          "kind": "internalParam",
          "spelling": "value"
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
                "preciseIdentifier": "c:input.h@T@BarPtr",
                "spelling": "BarPtr"
              },
              {
                "kind": "text",
                "spelling": " * "
              },
              {
                "kind": "internalParam",
                "spelling": "value"
              }
            ],
            "name": "value"
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
        "precise": "c:@F@baz"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "objective-c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "baz"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "baz"
          }
        ],
        "title": "baz"
      },
      "pathComponents": [
        "baz"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "typedef"
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
          "kind": "identifier",
          "spelling": "MyInt"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@T@MyInt"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "objective-c.typealias"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 0
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyInt"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyInt"
          }
        ],
        "title": "MyInt"
      },
      "pathComponents": [
        "MyInt"
      ],
      "type": "c:I"
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "typedef"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "keyword",
          "spelling": "struct"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:@S@Bar",
          "spelling": "Bar"
        },
        {
          "kind": "text",
          "spelling": " * "
        },
        {
          "kind": "identifier",
          "spelling": "BarPtr"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@T@BarPtr"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "objective-c.typealias"
      },
      "location": {
        "position": {
          "character": 20,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "BarPtr"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "BarPtr"
          }
        ],
        "title": "BarPtr"
      },
      "pathComponents": [
        "BarPtr"
      ],
      "type": "c:*$@S@Bar"
    }
  ]
}
