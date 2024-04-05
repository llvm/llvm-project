// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api --pretty-sgf -triple arm64-apple-macosx -x c-header\
// RUN:   %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
/// My Union
union Union {
    /// the a option
    int a;
    /// the b option
    char b;
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
      "source": "c:@U@Union@FI@a",
      "target": "c:@U@Union",
      "targetFallback": "Union"
    },
    {
      "kind": "memberOf",
      "source": "c:@U@Union@FI@b",
      "target": "c:@U@Union",
      "targetFallback": "Union"
    }
  ],
  "symbols": [
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "union"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Union"
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
                "character": 12,
                "line": 0
              },
              "start": {
                "character": 4,
                "line": 0
              }
            },
            "text": "My Union"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@U@Union"
      },
      "kind": {
        "displayName": "Union",
        "identifier": "c.union"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Union"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Union"
          }
        ],
        "title": "Union"
      },
      "pathComponents": [
        "Union"
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
                "character": 20,
                "line": 2
              },
              "start": {
                "character": 8,
                "line": 2
              }
            },
            "text": "the a option"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@U@Union@FI@a"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
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
        "Union",
        "a"
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
          "spelling": "b"
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
                "character": 20,
                "line": 4
              },
              "start": {
                "character": 8,
                "line": 4
              }
            },
            "text": "the b option"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@U@Union@FI@b"
      },
      "kind": {
        "displayName": "Instance Property",
        "identifier": "c.property"
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
        "Union",
        "b"
      ]
    }
  ]
}
