// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api -triple arm64-apple-macosx -x objective-c-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
@protocol Protocol
@property(class) int myProtocolTypeProp;
@property int myProtocolInstanceProp;
@end

@interface Interface
@property(class) int myInterfaceTypeProp;
@property int myInterfaceInstanceProp;
@end

@interface Interface (Category) <Protocol>
@property(class) int myCategoryTypeProp;
@property int myCategoryInstanceProp;
@end
// expected-no-diagnostics

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
      "source": "c:objc(cs)Interface(cpy)myInterfaceTypeProp",
      "target": "c:objc(cs)Interface",
      "targetFallback": "Interface"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Interface(py)myInterfaceInstanceProp",
      "target": "c:objc(cs)Interface",
      "targetFallback": "Interface"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Interface(cpy)myCategoryTypeProp",
      "target": "c:objc(cs)Interface",
      "targetFallback": "Interface"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Interface(py)myCategoryInstanceProp",
      "target": "c:objc(cs)Interface",
      "targetFallback": "Interface"
    },
    {
      "kind": "conformsTo",
      "source": "c:objc(cs)Interface",
      "target": "c:objc(pl)Protocol",
      "targetFallback": "Protocol"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(pl)Protocol(cpy)myProtocolTypeProp",
      "target": "c:objc(pl)Protocol",
      "targetFallback": "Protocol"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(pl)Protocol(py)myProtocolInstanceProp",
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
          "line": 6
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
          "kind": "keyword",
          "spelling": "@property"
        },
        {
          "kind": "text",
          "spelling": " ("
        },
        {
          "kind": "keyword",
          "spelling": "class"
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
          "spelling": "myInterfaceTypeProp"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Interface(cpy)myInterfaceTypeProp"
      },
      "kind": {
        "displayName": "Type Property",
        "identifier": "objective-c.type.property"
      },
      "location": {
        "position": {
          "character": 22,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "myInterfaceTypeProp"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "myInterfaceTypeProp"
          }
        ],
        "title": "myInterfaceTypeProp"
      },
      "pathComponents": [
        "Interface",
        "myInterfaceTypeProp"
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
          "spelling": "myInterfaceInstanceProp"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Interface(py)myInterfaceInstanceProp"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "objective-c.property"
      },
      "location": {
        "position": {
          "character": 15,
          "line": 8
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "myInterfaceInstanceProp"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "myInterfaceInstanceProp"
          }
        ],
        "title": "myInterfaceInstanceProp"
      },
      "pathComponents": [
        "Interface",
        "myInterfaceInstanceProp"
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
          "spelling": "class"
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
          "spelling": "myCategoryTypeProp"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Interface(cpy)myCategoryTypeProp"
      },
      "kind": {
        "displayName": "Type Property",
        "identifier": "objective-c.type.property"
      },
      "location": {
        "position": {
          "character": 22,
          "line": 12
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "myCategoryTypeProp"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "myCategoryTypeProp"
          }
        ],
        "title": "myCategoryTypeProp"
      },
      "pathComponents": [
        "Interface",
        "myCategoryTypeProp"
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
          "spelling": "myCategoryInstanceProp"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Interface(py)myCategoryInstanceProp"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "objective-c.property"
      },
      "location": {
        "position": {
          "character": 15,
          "line": 13
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "myCategoryInstanceProp"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "myCategoryInstanceProp"
          }
        ],
        "title": "myCategoryInstanceProp"
      },
      "pathComponents": [
        "Interface",
        "myCategoryInstanceProp"
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
          "spelling": "Protocol"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(pl)Protocol"
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
            "spelling": "Protocol"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Protocol"
          }
        ],
        "title": "Protocol"
      },
      "pathComponents": [
        "Protocol"
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
          "spelling": "class"
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
          "spelling": "myProtocolTypeProp"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(pl)Protocol(cpy)myProtocolTypeProp"
      },
      "kind": {
        "displayName": "Type Property",
        "identifier": "objective-c.type.property"
      },
      "location": {
        "position": {
          "character": 22,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "myProtocolTypeProp"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "myProtocolTypeProp"
          }
        ],
        "title": "myProtocolTypeProp"
      },
      "pathComponents": [
        "Protocol",
        "myProtocolTypeProp"
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
          "spelling": "myProtocolInstanceProp"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(pl)Protocol(py)myProtocolInstanceProp"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "objective-c.property"
      },
      "location": {
        "position": {
          "character": 15,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "myProtocolInstanceProp"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "myProtocolInstanceProp"
          }
        ],
        "title": "myProtocolInstanceProp"
      },
      "pathComponents": [
        "Protocol",
        "myProtocolInstanceProp"
      ]
    }
  ]
}
