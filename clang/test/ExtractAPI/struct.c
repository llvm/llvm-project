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
/// Color in RGBA
struct Color {
  unsigned Red;
  unsigned Green;
  unsigned Blue;
  /// Alpha channel for transparency
  unsigned Alpha;
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
      "source": "c:@S@Color@FI@Red",
      "target": "c:@S@Color"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Color@FI@Green",
      "target": "c:@S@Color"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Color@FI@Blue",
      "target": "c:@S@Color"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Color@FI@Alpha",
      "target": "c:@S@Color"
    }
  ],
  "symbols": [
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
          "spelling": "Color"
        }
      ],
      "docComment": {
        "lines": [
          {
            "range": {
              "end": {
                "character": 18,
                "line": 1
              },
              "start": {
                "character": 5,
                "line": 1
              }
            },
            "text": "Color in RGBA"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Color"
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
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Color"
          }
        ],
        "title": "Color"
      },
      "pathComponents": [
        "Color"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Red"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Color@FI@Red"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Red"
          }
        ],
        "title": "Red"
      },
      "pathComponents": [
        "Color",
        "Red"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Green"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Color@FI@Green"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Green"
          }
        ],
        "title": "Green"
      },
      "pathComponents": [
        "Color",
        "Green"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Blue"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Color@FI@Blue"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 5
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Blue"
          }
        ],
        "title": "Blue"
      },
      "pathComponents": [
        "Color",
        "Blue"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:i",
          "spelling": "unsigned int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Alpha"
        }
      ],
      "docComment": {
        "lines": [
          {
            "range": {
              "end": {
                "character": 37,
                "line": 6
              },
              "start": {
                "character": 7,
                "line": 6
              }
            },
            "text": "Alpha channel for transparency"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@S@Color@FI@Alpha"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
      },
      "location": {
        "position": {
          "character": 12,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Alpha"
          }
        ],
        "title": "Alpha"
      },
      "pathComponents": [
        "Color",
        "Alpha"
      ]
    }
  ]
}
