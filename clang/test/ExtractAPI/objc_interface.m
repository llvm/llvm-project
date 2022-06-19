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

@interface Super <Protocol>
@property(readonly, getter=getProperty) unsigned Property;
+ (id)getWithProperty:(unsigned) Property;
- (void)setProperty:(unsigned) Property andOtherThing: (unsigned) Thing;
@end

@interface Derived : Super {
  char Ivar;
}
- (char)getIvar;
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
      "source": "c:objc(cs)Super(cm)getWithProperty:",
      "target": "c:objc(cs)Super"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Super(im)setProperty:andOtherThing:",
      "target": "c:objc(cs)Super"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Super(py)Property",
      "target": "c:objc(cs)Super"
    },
    {
      "kind": "conformsTo",
      "source": "c:objc(cs)Super",
      "target": "c:objc(pl)Protocol"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Derived@Ivar",
      "target": "c:objc(cs)Derived"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Derived(im)getIvar",
      "target": "c:objc(cs)Derived"
    },
    {
      "kind": "inheritsFrom",
      "source": "c:objc(cs)Derived",
      "target": "c:objc(cs)Super"
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
          "spelling": "Super"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Super"
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
            "spelling": "Super"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Super"
          }
        ],
        "title": "Super"
      },
      "pathComponents": [
        "Super"
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
          "kind": "keyword",
          "spelling": "id"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "getWithProperty:"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "internalParam",
          "spelling": "Property"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "text",
                "spelling": "("
              },
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:i",
                "spelling": "unsigned int"
              },
              {
                "kind": "text",
                "spelling": ") "
              },
              {
                "kind": "internalParam",
                "spelling": "Property"
              }
            ],
            "name": "Property"
          }
        ],
        "returns": [
          {
            "kind": "keyword",
            "spelling": "id"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Super(cm)getWithProperty:"
      },
      "kind": {
        "displayName": "Type Method",
        "identifier": "objective-c.type.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "getWithProperty:"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "+ "
          },
          {
            "kind": "identifier",
            "spelling": "getWithProperty:"
          }
        ],
        "title": "getWithProperty:"
      },
      "pathComponents": [
        "Super",
        "getWithProperty:"
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
          "spelling": "setProperty:"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "internalParam",
          "spelling": "Property"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "andOtherThing:"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "internalParam",
          "spelling": "Thing"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "text",
                "spelling": "("
              },
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:i",
                "spelling": "unsigned int"
              },
              {
                "kind": "text",
                "spelling": ") "
              },
              {
                "kind": "internalParam",
                "spelling": "Property"
              }
            ],
            "name": "Property"
          },
          {
            "declarationFragments": [
              {
                "kind": "text",
                "spelling": "("
              },
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:i",
                "spelling": "unsigned int"
              },
              {
                "kind": "text",
                "spelling": ") "
              },
              {
                "kind": "internalParam",
                "spelling": "Thing"
              }
            ],
            "name": "Thing"
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
        "precise": "c:objc(cs)Super(im)setProperty:andOtherThing:"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "setProperty:andOtherThing:"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "setProperty:andOtherThing:"
          }
        ],
        "title": "setProperty:andOtherThing:"
      },
      "pathComponents": [
        "Super",
        "setProperty:andOtherThing:"
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
          "spelling": "readonly"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "keyword",
          "spelling": "getter"
        },
        {
          "kind": "text",
          "spelling": "="
        },
        {
          "kind": "identifier",
          "spelling": "getProperty"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "identifier",
          "spelling": "Property"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Super(py)Property"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "objective-c.property"
      },
      "location": {
        "position": {
          "character": 50,
          "line": 4
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
        "Super",
        "Property"
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
          "kind": "identifier",
          "spelling": "Derived"
        },
        {
          "kind": "text",
          "spelling": " : "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:objc(cs)Super",
          "spelling": "Super"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Derived"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "objective-c.class"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Derived"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Derived"
          }
        ],
        "title": "Derived"
      },
      "pathComponents": [
        "Derived"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:C",
          "spelling": "char"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Ivar"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Derived@Ivar"
      },
      "kind": {
        "displayName": "Instance Variable",
        "identifier": "objective-c.ivar"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 10
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Ivar"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Ivar"
          }
        ],
        "title": "Ivar"
      },
      "pathComponents": [
        "Derived",
        "Ivar"
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
          "preciseIdentifier": "c:C",
          "spelling": "char"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "getIvar"
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
            "preciseIdentifier": "c:C",
            "spelling": "char"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Derived(im)getIvar"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 1,
          "line": 12
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "getIvar"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "getIvar"
          }
        ],
        "title": "getIvar"
      },
      "pathComponents": [
        "Derived",
        "getIvar"
      ]
    }
  ]
}
