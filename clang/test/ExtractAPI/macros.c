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
#define WORLD 2
#define MACRO_FUN(x) x x
#define FUN(x, y, z) x + y + z
#define FUNC99(x, ...)
#define FUNGNU(x...)

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
          "character": 8,
          "line": 0
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
          "spelling": "WORLD"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@24@macro@WORLD"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "objective-c.macro"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "WORLD"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "WORLD"
          }
        ],
        "title": "WORLD"
      },
      "pathComponents": [
        "WORLD"
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
          "spelling": "MACRO_FUN"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "internalParam",
          "spelling": "x"
        },
        {
          "kind": "text",
          "spelling": ")"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@40@macro@MACRO_FUN"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "objective-c.macro"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "MACRO_FUN"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MACRO_FUN"
          }
        ],
        "title": "MACRO_FUN"
      },
      "pathComponents": [
        "MACRO_FUN"
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
          "spelling": "FUN"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "internalParam",
          "spelling": "x"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "internalParam",
          "spelling": "y"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
        {
          "kind": "internalParam",
          "spelling": "z"
        },
        {
          "kind": "text",
          "spelling": ")"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@65@macro@FUN"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "objective-c.macro"
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
            "spelling": "FUN"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "FUN"
          }
        ],
        "title": "FUN"
      },
      "pathComponents": [
        "FUN"
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
          "spelling": "FUNC99"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "internalParam",
          "spelling": "x"
        },
        {
          "kind": "text",
          "spelling": ", ...)"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@96@macro@FUNC99"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "objective-c.macro"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "FUNC99"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "FUNC99"
          }
        ],
        "title": "FUNC99"
      },
      "pathComponents": [
        "FUNC99"
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
          "spelling": "FUNGNU"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "internalParam",
          "spelling": "x"
        },
        {
          "kind": "text",
          "spelling": "...)"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@119@macro@FUNGNU"
      },
      "kind": {
        "displayName": "Macro",
        "identifier": "objective-c.macro"
      },
      "location": {
        "position": {
          "character": 8,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "FUNGNU"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "FUNGNU"
          }
        ],
        "title": "FUNGNU"
      },
      "pathComponents": [
        "FUNGNU"
      ]
    }
  ]
}
