// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%/t@g" %t/reference.output.json.in >> \
// RUN: %t/reference.output.json
// RUN: %clang -extract-api -target arm64-apple-macosx \
// RUN: %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

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
      "target": "c:@E@Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Vehicle@Car",
      "target": "c:@E@Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Vehicle@Train",
      "target": "c:@E@Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Vehicle@Ship",
      "target": "c:@E@Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Vehicle@Airplane",
      "target": "c:@E@Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Direction@North",
      "target": "c:@E@Direction"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Direction@East",
      "target": "c:@E@Direction"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Direction@South",
      "target": "c:@E@Direction"
    },
    {
      "kind": "memberOf",
      "source": "c:@E@Direction@West",
      "target": "c:@E@Direction"
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
        }
      ],
      "docComment": {
        "lines": [
          {
            "range": {
              "end": {
                "character": 22,
                "line": 1
              },
              "start": {
                "character": 5,
                "line": 1
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
          "character": 6,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 3,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 3,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
                "character": 45,
                "line": 5
              },
              "start": {
                "character": 15,
                "line": 5
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
          "character": 3,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 3,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 3,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 6,
          "line": 10
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 3,
          "line": 11
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 3,
          "line": 12
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 3,
          "line": 13
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
          "character": 3,
          "line": 14
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
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
    }
  ]
}
