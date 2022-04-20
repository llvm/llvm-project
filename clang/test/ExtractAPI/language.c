// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/c.reference.output.json.in >> %t/c.reference.output.json
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/objc.reference.output.json.in >> %t/objc.reference.output.json

// RUN: %clang -extract-api -x c-header -target arm64-apple-macosx \
// RUN: %t/c.h -o %t/c.output.json | FileCheck -allow-empty %s
// RUN: %clang -extract-api -x objective-c-header -target arm64-apple-macosx \
// RUN: %t/objc.h -o %t/objc.output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/c.output.json >> %t/c.output-normalized.json
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/objc.output.json >> %t/objc.output-normalized.json

// RUN: diff %t/c.reference.output.json %t/c.output-normalized.json
// RUN: diff %t/objc.reference.output.json %t/objc.output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- c.h
char c;

//--- objc.h
char objc;

//--- c.reference.output.json.in
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
  "relationships": [],
  "symbols": [
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
          "spelling": "c"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@c"
      },
      "kind": {
        "displayName": "Global Variable",
        "identifier": "c.var"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 1
        },
        "uri": "file://INPUT_DIR/c.h"
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
    }
  ]
}
//--- objc.reference.output.json.in
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
  "relationships": [],
  "symbols": [
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
          "spelling": "objc"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:@objc"
      },
      "kind": {
        "displayName": "Global Variable",
        "identifier": "objective-c.var"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 1
        },
        "uri": "file://INPUT_DIR/objc.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "objc"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "objc"
          }
        ],
        "title": "objc"
      },
      "pathComponents": [
        "objc"
      ]
    }
  ]
}
