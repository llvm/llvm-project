// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api --pretty-sgf --product-name=Availability -triple arm64-apple-macosx -x c-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
void a(void);

void b(void) __attribute__((availability(macos, introduced=12.0)));

void c(void) __attribute__((availability(macos, introduced=11.0, deprecated=12.0, obsoleted=20.0)));

void d(void) __attribute__((availability(macos, introduced=11.0, deprecated=12.0, obsoleted=20.0))) __attribute__((availability(ios, introduced=13.0)));

void e(void) __attribute__((deprecated)) __attribute__((availability(macos, introduced=11.0)));

void f(void) __attribute__((unavailable)) __attribute__((availability(macos, introduced=11.0)));

void d(void) __attribute__((availability(tvos, introduced=15.0)));

void e(void) __attribute__((availability(tvos, unavailable)));

///expected-no-diagnostics

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
    "name": "Availability",
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
          "spelling": "a"
        },
        {
          "kind": "text",
          "spelling": "();"
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
        "interfaceLanguage": "c",
        "precise": "c:@F@a"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 0
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
        "a"
      ]
    },
    {
      "accessLevel": "public",
      "availability": [
        {
          "domain": "macos",
          "introduced": {
            "major": 12,
            "minor": 0,
            "patch": 0
          }
        }
      ],
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
          "spelling": "b"
        },
        {
          "kind": "text",
          "spelling": "();"
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
        "interfaceLanguage": "c",
        "precise": "c:@F@b"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "b"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "b"
          }
        ],
        "title": "b"
      },
      "pathComponents": [
        "b"
      ]
    },
    {
      "accessLevel": "public",
      "availability": [
        {
          "deprecated": {
            "major": 12,
            "minor": 0,
            "patch": 0
          },
          "domain": "macos",
          "introduced": {
            "major": 11,
            "minor": 0,
            "patch": 0
          },
          "obsoleted": {
            "major": 20,
            "minor": 0,
            "patch": 0
          }
        }
      ],
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
          "spelling": "c"
        },
        {
          "kind": "text",
          "spelling": "();"
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
        "interfaceLanguage": "c",
        "precise": "c:@F@c"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
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
            "spelling": "c"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "c"
          }
        ],
        "title": "c"
      },
      "pathComponents": [
        "c"
      ]
    },
    {
      "accessLevel": "public",
      "availability": [
        {
          "deprecated": {
            "major": 12,
            "minor": 0,
            "patch": 0
          },
          "domain": "macos",
          "introduced": {
            "major": 11,
            "minor": 0,
            "patch": 0
          },
          "obsoleted": {
            "major": 20,
            "minor": 0,
            "patch": 0
          }
        }
      ],
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
          "spelling": "d"
        },
        {
          "kind": "text",
          "spelling": "();"
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
        "interfaceLanguage": "c",
        "precise": "c:@F@d"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
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
            "spelling": "d"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "d"
          }
        ],
        "title": "d"
      },
      "pathComponents": [
        "d"
      ]
    },
    {
      "accessLevel": "public",
      "availability": [
        {
          "domain": "*",
          "isUnconditionallyDeprecated": true
        },
        {
          "domain": "macos",
          "introduced": {
            "major": 11,
            "minor": 0,
            "patch": 0
          }
        }
      ],
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
          "spelling": "e"
        },
        {
          "kind": "text",
          "spelling": "();"
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
        "interfaceLanguage": "c",
        "precise": "c:@F@e"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 8
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "e"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "e"
          }
        ],
        "title": "e"
      },
      "pathComponents": [
        "e"
      ]
    }
  ]
}
