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
/// A Vehicle
struct Vehicle {
    /// The type of vehicle.
    enum {
        Bicycle,
        Car
    } type;

    /// The information about the vehicle.
    struct {
        int wheels;
        char *name;
    } information;
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
      "source": "c:@S@Vehicle@E@input.h@64@Bicycle",
      "target": "c:@S@Vehicle@E@input.h@64",
      "targetFallback": "Vehicle::enum (unnamed)"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Vehicle@E@input.h@64@Car",
      "target": "c:@S@Vehicle@E@input.h@64",
      "targetFallback": "Vehicle::enum (unnamed)"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Vehicle@FI@type",
      "target": "c:@S@Vehicle",
      "targetFallback": "Vehicle"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Vehicle@FI@information",
      "target": "c:@S@Vehicle",
      "targetFallback": "Vehicle"
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
                "character": 29,
                "line": 3
              },
              "start": {
                "character": 9,
                "line": 3
              }
            },
            "text": "The type of vehicle."
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Vehicle@E@input.h@64"
      },
      "kind": {
        "displayName": "Enumeration",
        "identifier": "c.enum"
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
            "spelling": "Vehicle::enum (unnamed)"
          }
        ],
        "title": "Vehicle::enum (unnamed)"
      },
      "pathComponents": [
        "Vehicle::enum (unnamed)"
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
        "precise": "c:@S@Vehicle@E@input.h@64@Bicycle"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 5
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
        "Vehicle::enum (unnamed)",
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
        "precise": "c:@S@Vehicle@E@input.h@64@Car"
      },
      "kind": {
        "displayName": "Enumeration Case",
        "identifier": "c.enum.case"
      },
      "location": {
        "position": {
          "character": 9,
          "line": 6
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
        "Vehicle::enum (unnamed)",
        "Car"
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
          "spelling": "Vehicle"
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
                "character": 14,
                "line": 1
              },
              "start": {
                "character": 5,
                "line": 1
              }
            },
            "text": "A Vehicle"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Vehicle"
      },
      "kind": {
        "displayName": "Structure",
        "identifier": "c.struct"
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
          "kind": "keyword",
          "spelling": "enum"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "type"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Vehicle@FI@type"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
      },
      "location": {
        "position": {
          "character": 7,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "type"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "type"
          }
        ],
        "title": "type"
      },
      "pathComponents": [
        "Vehicle",
        "type"
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
          "spelling": "information"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Vehicle@FI@information"
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
            "spelling": "information"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "information"
          }
        ],
        "title": "information"
      },
      "pathComponents": [
        "Vehicle",
        "information"
      ]
    }
  ]
}
