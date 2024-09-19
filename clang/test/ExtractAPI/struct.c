// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang -extract-api --pretty-sgf -target arm64-apple-macosx \
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
      "target": "c:@S@Color",
      "targetFallback": "Color"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Color@FI@Green",
      "target": "c:@S@Color",
      "targetFallback": "Color"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Color@FI@Blue",
      "target": "c:@S@Color",
      "targetFallback": "Color"
    },
    {
      "kind": "memberOf",
      "source": "c:@S@Color@FI@Alpha",
      "target": "c:@S@Color",
      "targetFallback": "Color"
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
                "character": 17,
                "line": 0
              },
              "start": {
                "character": 4,
                "line": 0
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
          "character": 7,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Color"
          }
        ],
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
        },
        {
          "kind": "text",
          "spelling": ";"
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
          "character": 11,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Red"
          }
        ],
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
        },
        {
          "kind": "text",
          "spelling": ";"
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
          "character": 11,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Green"
          }
        ],
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
        },
        {
          "kind": "text",
          "spelling": ";"
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
          "character": 11,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Blue"
          }
        ],
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
                "character": 36,
                "line": 5
              },
              "start": {
                "character": 6,
                "line": 5
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
          "character": 11,
          "line": 6
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Alpha"
          }
        ],
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
