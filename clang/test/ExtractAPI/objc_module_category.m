// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang -extract-api -x objective-c-header \
// RUN: -target arm64-apple-macosx \
// RUN: %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
#import "Foundation.h"

/// Doc comment 1
@interface NSString (Category1)
-(void)method1;
@end

/// Doc comment 2
@interface NSString (Category2)
-(void)method2;
@end

//--- Foundation.h
@interface NSString
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
      "kind": "extensionTo",
      "source": "c:objc(cy)NSString@Category1",
      "target": "c:objc(cs)NSString",
      "targetFallback": "NSString"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)NSString(im)method1",
      "target": "c:objc(cy)NSString@Category1",
      "targetFallback": "Category1"
    },
    {
      "kind": "extensionTo",
      "source": "c:objc(cy)NSString@Category2",
      "target": "c:objc(cs)NSString",
      "targetFallback": "NSString"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)NSString(im)method2",
      "target": "c:objc(cy)NSString@Category2",
      "targetFallback": "Category2"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cy)NSString@Category1"
      },
      "kind": {
        "displayName": "Module Extension",
        "identifier": "objective-c.module.extension"
      }
    },
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
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:objc(cs)NSString",
          "spelling": "NSString"
        },
        {
          "kind": "text",
          "spelling": " ("
        },
        {
          "kind": "identifier",
          "spelling": "Category1"
        },
        {
          "kind": "text",
          "spelling": ")"
        }
      ],
      "docComment": {
        "lines": [
          {
            "range": {
              "end": {
                "character": 17,
                "line": 2
              },
              "start": {
                "character": 4,
                "line": 2
              }
            },
            "text": "Doc comment 1"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cy)NSString@Category1"
      },
      "kind": {
        "displayName": "Class Extension",
        "identifier": "objective-c.class.extension"
      },
      "location": {
        "position": {
          "character": 11,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Category1"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Category1"
          }
        ],
        "title": "NSString (Category1)"
      },
      "pathComponents": [
        "Category1"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "method1"
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
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)NSString(im)method1"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 0,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "method1"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "method1"
          }
        ],
        "title": "method1"
      },
      "pathComponents": [
        "Category1",
        "method1"
      ]
    },
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
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:objc(cs)NSString",
          "spelling": "NSString"
        },
        {
          "kind": "text",
          "spelling": " ("
        },
        {
          "kind": "identifier",
          "spelling": "Category2"
        },
        {
          "kind": "text",
          "spelling": ")"
        }
      ],
      "docComment": {
        "lines": [
          {
            "range": {
              "end": {
                "character": 17,
                "line": 7
              },
              "start": {
                "character": 4,
                "line": 7
              }
            },
            "text": "Doc comment 2"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cy)NSString@Category2"
      },
      "kind": {
        "displayName": "Class Extension",
        "identifier": "objective-c.class.extension"
      },
      "location": {
        "position": {
          "character": 11,
          "line": 8
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Category2"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Category2"
          }
        ],
        "title": "NSString (Category2)"
      },
      "pathComponents": [
        "Category2"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "method2"
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
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)NSString(im)method2"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 0,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "method2"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "method2"
          }
        ],
        "title": "method2"
      },
      "pathComponents": [
        "Category2",
        "method2"
      ]
    }
  ]
}
