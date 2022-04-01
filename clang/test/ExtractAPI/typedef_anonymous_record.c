// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%/t@g" %t/reference.output.json.in >> \
// RUN: %t/reference.output.json
// RUN: %clang -extract-api --product-name=TypedefChain -target arm64-apple-macosx \
// RUN: -x objective-c-header %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
typedef struct { } MyStruct;
typedef MyStruct MyStructStruct;
typedef MyStructStruct MyStructStructStruct;

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
    "name": "TypedefChain",
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
          "kind": "identifier",
          "spelling": "MyStruct"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:@SA@MyStruct"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "objective-c.struct"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "title": "MyStruct"
      },
      "pathComponents": [
        "MyStruct"
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
          "preciseIdentifier": "c:@SA@MyStruct",
          "spelling": "MyStruct"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyStructStruct"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@T@MyStructStruct"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "objective-c.typealias"
      },
      "location": {
        "position": {
          "character": 18,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyStructStruct"
          }
        ],
        "title": "MyStructStruct"
      },
      "pathComponents": [
        "MyStructStruct"
      ],
      "type": "c:@SA@MyStruct"
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
          "preciseIdentifier": "c:input.h@T@MyStructStruct",
          "spelling": "MyStructStruct"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyStructStructStruct"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@T@MyStructStructStruct"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "objective-c.typealias"
      },
      "location": {
        "position": {
          "character": 24,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyStructStructStruct"
          }
        ],
        "title": "MyStructStructStruct"
      },
      "pathComponents": [
        "MyStructStructStruct"
      ],
      "type": "c:input.h@T@MyStructStruct"
    }
  ]
}
