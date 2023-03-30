// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang -extract-api -x objective-c-header -target arm64-apple-macosx \
// RUN: %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
@protocol MyProtocol
@end

@interface MyInterface
@property(copy, readwrite) id<MyProtocol> obj1;
@property(readwrite) id<MyProtocol> *obj2;
@end
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
      "source": "c:objc(cs)MyInterface(py)obj1",
      "target": "c:objc(cs)MyInterface",
      "targetFallback": "MyInterface"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)MyInterface(py)obj2",
      "target": "c:objc(cs)MyInterface",
      "targetFallback": "MyInterface"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@interface"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyInterface"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)MyInterface"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "objective-c.class"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyInterface"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyInterface"
          }
        ],
        "title": "MyInterface"
      },
      "pathComponents": [
        "MyInterface"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@property"
        },
        {
          "kind": "text",
          "spelling": " ("
        },
        {
          "kind": "keyword",
          "spelling": "atomic"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "keyword",
          "spelling": "copy"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "keyword",
          "spelling": "readwrite"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:Qoobjc(pl)MyProtocol",
          "spelling": "id<MyProtocol>"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "obj1"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)MyInterface(py)obj1"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "objective-c.property"
      },
      "location": {
        "position": {
          "character": 43,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "obj1"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "obj1"
          }
        ],
        "title": "obj1"
      },
      "pathComponents": [
        "MyInterface",
        "obj1"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@property"
        },
        {
          "kind": "text",
          "spelling": " ("
        },
        {
          "kind": "keyword",
          "spelling": "atomic"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "keyword",
          "spelling": "assign"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "keyword",
          "spelling": "unsafe_unretained"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "keyword",
          "spelling": "readwrite"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:Qoobjc(pl)MyProtocol",
          "spelling": "id<MyProtocol>"
        },
        {
          "kind": "text",
          "spelling": " * "
        },
        {
          "kind": "identifier",
          "spelling": "obj2"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)MyInterface(py)obj2"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "objective-c.property"
      },
      "location": {
        "position": {
          "character": 38,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "obj2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "obj2"
          }
        ],
        "title": "obj2"
      },
      "pathComponents": [
        "MyInterface",
        "obj2"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@protocol"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyProtocol"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(pl)MyProtocol"
      },
      "kind": {
        "displayName": "Protocol",
        "identifier": "objective-c.protocol"
      },
      "location": {
        "position": {
          "character": 11,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyProtocol"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyProtocol"
          }
        ],
        "title": "MyProtocol"
      },
      "pathComponents": [
        "MyProtocol"
      ]
    }
  ]
}
