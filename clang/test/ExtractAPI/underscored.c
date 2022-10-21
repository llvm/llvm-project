// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api -triple arm64-apple-macosx \
// RUN:   -x c-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
// expected-no-diagnostics

// Global record
int _HiddenGlobal;
int exposed_global;

// Record type
struct _HiddenRecord {
  int a;
};

struct ExposedRecord {
  int a;
};

// Typedef
typedef struct {} _HiddenTypedef;
typedef int ExposedTypedef;
typedef _HiddenTypedef ExposedTypedefToHidden;

// Macros
#define _HIDDEN_MACRO 5
#define EXPOSED_MACRO 5

// Symbols that start with '_' should not appear in the reference output
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
      "source": "c:@S@ExposedRecord@FI@a",
      "target": "c:@S@ExposedRecord",
      "targetFallback": "ExposedRecord"
    }
  ],
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
          "spelling": "exposed_global"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@exposed_global"
      },
      "kind": {
        "displayName": "Global Variable",
        "identifier": "c.var"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "exposed_global"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "exposed_global"
          }
        ],
        "title": "exposed_global"
      },
      "pathComponents": [
        "exposed_global"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
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
          "spelling": "ExposedRecord"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@ExposedRecord"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "c.struct"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 12
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "ExposedRecord"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "ExposedRecord"
          }
        ],
        "title": "ExposedRecord"
      },
      "pathComponents": [
        "ExposedRecord"
      ]
    },
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
          "spelling": "a"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@ExposedRecord@FI@a"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
      },
      "location": {
        "position": {
          "character": 7,
          "line": 13
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "a"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "a"
          }
        ],
        "title": "a"
      },
      "pathComponents": [
        "ExposedRecord",
        "a"
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
          "spelling": "EXPOSED_MACRO"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:input.h@335@macro@EXPOSED_MACRO"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "c.macro"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 23
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "EXPOSED_MACRO"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "EXPOSED_MACRO"
          }
        ],
        "title": "EXPOSED_MACRO"
      },
      "pathComponents": [
        "EXPOSED_MACRO"
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
          "spelling": "ExposedTypedef"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:input.h@T@ExposedTypedef"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "c.typealias"
      },
      "location": {
        "position": {
          "character": 13,
          "line": 18
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "ExposedTypedef"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "ExposedTypedef"
          }
        ],
        "title": "ExposedTypedef"
      },
      "pathComponents": [
        "ExposedTypedef"
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
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:@SA@_HiddenTypedef",
          "spelling": "_HiddenTypedef"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "ExposedTypedefToHidden"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:input.h@T@ExposedTypedefToHidden"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "c.typealias"
      },
      "location": {
        "position": {
          "character": 24,
          "line": 19
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "ExposedTypedefToHidden"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "ExposedTypedefToHidden"
          }
        ],
        "title": "ExposedTypedefToHidden"
      },
      "pathComponents": [
        "ExposedTypedefToHidden"
      ],
      "type": "c:@SA@_HiddenTypedef"
    }
  ]
}
