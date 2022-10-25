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
@protocol Protocol;

@interface Interface
@end

@interface Interface (Category) <Protocol>
@property int Property;
- (void)InstanceMethod;
+ (void)ClassMethod;
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
      "source": "c:objc(cs)Interface(im)InstanceMethod",
      "target": "c:objc(cs)Interface"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Interface(cm)ClassMethod",
      "target": "c:objc(cs)Interface"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Interface(py)Property",
      "target": "c:objc(cs)Interface"
    },
    {
      "kind": "conformsTo",
      "source": "c:objc(cs)Interface",
      "target": "c:objc(pl)Protocol",
      "targetFallback": "Protocol"
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
          "spelling": "Interface"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Interface"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "objective-c.class"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Interface"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Interface"
          }
        ],
        "title": "Interface"
      },
      "pathComponents": [
        "Interface"
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
          "spelling": "InstanceMethod"
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
        "precise": "c:objc(cs)Interface(im)InstanceMethod"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 8
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "InstanceMethod"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "InstanceMethod"
          }
        ],
        "title": "InstanceMethod"
      },
      "pathComponents": [
        "Interface",
        "InstanceMethod"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "+ ("
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
          "spelling": "ClassMethod"
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
        "precise": "c:objc(cs)Interface(cm)ClassMethod"
      },
      "kind": {
        "displayName": "Type Method",
        "identifier": "objective-c.type.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "ClassMethod"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "+ "
          },
          {
            "kind": "identifier",
            "spelling": "ClassMethod"
          }
        ],
        "title": "ClassMethod"
      },
      "pathComponents": [
        "Interface",
        "ClassMethod"
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
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Property"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Interface(py)Property"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "objective-c.property"
      },
      "location": {
        "position": {
          "character": 15,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Property"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Property"
          }
        ],
        "title": "Property"
      },
      "pathComponents": [
        "Interface",
        "Property"
      ]
    }
  ]
}
