// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api --product-name=TypedefChain -triple arm64-apple-macosx \
// RUN:   -x c-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
typedef struct { } MyStruct;
typedef MyStruct MyStructStruct;
typedef MyStructStruct MyStructStructStruct;
typedef enum { Case } MyEnum;
typedef MyEnum MyEnumEnum;
typedef MyEnumEnum MyEnumEnumEnum;
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
  "relationships": [
    {
      "kind": "memberOf",
      "source": "c:@EA@MyEnum@Case",
      "target": "c:@EA@MyEnum",
      "targetFallback": "MyEnum"
    }
  ],
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
          "spelling": "enum"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyEnum"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@EA@MyEnum"
      },
      "kind": {
        "displayName": "Enumeration",
        "identifier": "c.enum"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyEnum"
          }
        ],
        "title": "MyEnum"
      },
      "pathComponents": [
        "MyEnum"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "Case"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@EA@MyEnum@Case"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
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
            "spelling": "Case"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Case"
          }
        ],
        "title": "Case"
      },
      "pathComponents": [
        "MyEnum",
        "Case"
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
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@SA@MyStruct"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "c.struct"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 0
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyStruct"
          }
        ],
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
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:input.h@T@MyStructStruct"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "c.typealias"
      },
      "location": {
        "position": {
          "character": 17,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyStructStruct"
          }
        ],
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
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:input.h@T@MyStructStructStruct"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "c.typealias"
      },
      "location": {
        "position": {
          "character": 23,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyStructStructStruct"
          }
        ],
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
          "preciseIdentifier": "c:@EA@MyEnum",
          "spelling": "MyEnum"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyEnumEnum"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:input.h@T@MyEnumEnum"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "c.typealias"
      },
      "location": {
        "position": {
          "character": 15,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyEnumEnum"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyEnumEnum"
          }
        ],
        "title": "MyEnumEnum"
      },
      "pathComponents": [
        "MyEnumEnum"
      ],
      "type": "c:@EA@MyEnum"
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
          "preciseIdentifier": "c:input.h@T@MyEnumEnum",
          "spelling": "MyEnumEnum"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "MyEnumEnumEnum"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:input.h@T@MyEnumEnumEnum"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "c.typealias"
      },
      "location": {
        "position": {
          "character": 19,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MyEnumEnumEnum"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyEnumEnumEnum"
          }
        ],
        "title": "MyEnumEnumEnum"
      },
      "pathComponents": [
        "MyEnumEnumEnum"
      ],
      "type": "c:input.h@T@MyEnumEnum"
    }
  ]
}
