// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api --pretty-sgf -triple arm64-apple-macosx \
// RUN:   -x c-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
/// Kinds of vehicles
enum Vehicle {
  Bicycle,
  Car,
  Train, ///< Move this to the top! -Sheldon
  Ship,
  Airplane,
};

enum Direction : unsigned char {
  North = 0,
  East,
  South,
  West
};

enum {
  Constant = 1
};

enum {
  OtherConstant = 2
};
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
      "source": "c:@E@Vehicle@Bicycle",
      "target": "c:@E@Vehicle",
      "targetFallback": "Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Vehicle@Car",
      "target": "c:@E@Vehicle",
      "targetFallback": "Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Vehicle@Train",
      "target": "c:@E@Vehicle",
      "targetFallback": "Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Vehicle@Ship",
      "target": "c:@E@Vehicle",
      "targetFallback": "Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Vehicle@Airplane",
      "target": "c:@E@Vehicle",
      "targetFallback": "Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Direction@North",
      "target": "c:@E@Direction",
      "targetFallback": "Direction"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Direction@East",
      "target": "c:@E@Direction",
      "targetFallback": "Direction"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Direction@South",
      "target": "c:@E@Direction",
      "targetFallback": "Direction"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Direction@West",
      "target": "c:@E@Direction",
      "targetFallback": "Direction"
    },
    {
      "kind": "memberOf",
      "source": "c:@Ea@Constant@Constant",
      "target": "c:@Ea@Constant",
      "targetFallback": "enum (unnamed)"
    },
    {
      "kind": "memberOf",
      "source": "c:@Ea@OtherConstant@OtherConstant",
      "target": "c:@Ea@OtherConstant",
      "targetFallback": "enum (unnamed)"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
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
          "spelling": "Vehicle"
        },
        {
          "kind": "text",
          "spelling": ": "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "docComment": {
        "lines": [
          {
            "range": {
              "end": {
                "character": 21,
                "line": 0
              },
              "start": {
                "character": 4,
                "line": 0
              }
            },
            "text": "Kinds of vehicles"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Vehicle"
      },
      "kind": {
        "displayName": "Enumeration",
        "identifier": "c.enum"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Vehicle"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Vehicle"
          }
        ],
        "title": "Vehicle"
      },
      "pathComponents": [
        "Vehicle"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "Bicycle"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Vehicle@Bicycle"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Bicycle"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Bicycle"
          }
        ],
        "title": "Bicycle"
      },
      "pathComponents": [
        "Vehicle",
        "Bicycle"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "Car"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Vehicle@Car"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Car"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Car"
          }
        ],
        "title": "Car"
      },
      "pathComponents": [
        "Vehicle",
        "Car"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "Train"
        }
      ],
      "docComment": {
        "lines": [
          {
            "range": {
              "end": {
                "character": 44,
                "line": 4
              },
              "start": {
                "character": 14,
                "line": 4
              }
            },
            "text": "Move this to the top! -Sheldon"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Vehicle@Train"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Train"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Train"
          }
        ],
        "title": "Train"
      },
      "pathComponents": [
        "Vehicle",
        "Train"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "Ship"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Vehicle@Ship"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Ship"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Ship"
          }
        ],
        "title": "Ship"
      },
      "pathComponents": [
        "Vehicle",
        "Ship"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "Airplane"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Vehicle@Airplane"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Airplane"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Airplane"
          }
        ],
        "title": "Airplane"
      },
      "pathComponents": [
        "Vehicle",
        "Airplane"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
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
          "spelling": "Direction"
        },
        {
          "kind": "text",
          "spelling": ": "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:c",
          "spelling": "unsigned char"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Direction"
      },
      "kind": {
        "displayName": "Enumeration",
        "identifier": "c.enum"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Direction"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Direction"
          }
        ],
        "title": "Direction"
      },
      "pathComponents": [
        "Direction"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "North"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Direction@North"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 10
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "North"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "North"
          }
        ],
        "title": "North"
      },
      "pathComponents": [
        "Direction",
        "North"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "East"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Direction@East"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 11
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "East"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "East"
          }
        ],
        "title": "East"
      },
      "pathComponents": [
        "Direction",
        "East"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "South"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Direction@South"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 12
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "South"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "South"
          }
        ],
        "title": "South"
      },
      "pathComponents": [
        "Direction",
        "South"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "West"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@E@Direction@West"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 13
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "West"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "West"
          }
        ],
        "title": "West"
      },
      "pathComponents": [
        "Direction",
        "West"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "enum"
        },
        {
          "kind": "text",
          "spelling": ": "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@Ea@Constant"
      },
      "kind": {
        "displayName": "Enumeration",
        "identifier": "c.enum"
      },
      "location": {
        "position": {
          "character": 0,
          "line": 16
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "enum (unnamed)"
          }
        ],
        "title": "enum (unnamed)"
      },
      "pathComponents": [
        "enum (unnamed)"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "Constant"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@Ea@Constant@Constant"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 17
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Constant"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Constant"
          }
        ],
        "title": "Constant"
      },
      "pathComponents": [
        "enum (unnamed)",
        "Constant"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "enum"
        },
        {
          "kind": "text",
          "spelling": ": "
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@Ea@OtherConstant"
      },
      "kind": {
        "displayName": "Enumeration",
        "identifier": "c.enum"
      },
      "location": {
        "position": {
          "character": 0,
          "line": 20
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "enum (unnamed)"
          }
        ],
        "title": "enum (unnamed)"
      },
      "pathComponents": [
        "enum (unnamed)"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "identifier",
          "spelling": "OtherConstant"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@Ea@OtherConstant@OtherConstant"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 2,
          "line": 21
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "OtherConstant"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "OtherConstant"
          }
        ],
        "title": "OtherConstant"
      },
      "pathComponents": [
        "enum (unnamed)",
        "OtherConstant"
      ]
    }
  ]
}
