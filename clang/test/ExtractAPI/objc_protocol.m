// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%/t@g" %t/reference.output.json.in >> \
// RUN: %t/reference.output.json
// RUN: %clang -extract-api -x objective-c-header -target arm64-apple-macosx \
// RUN: %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
@protocol Protocol
@end

@protocol AnotherProtocol <Protocol>
@end

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
  "relationhips": [
    {
      "kind": "conformsTo",
      "source": "c:objc(pl)AnotherProtocol",
      "target": "c:objc(pl)Protocol"
    }
  ],
  "symbols": [
    {
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@protocol"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Protocol"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(pl)Protocol"
      },
      "kind": {
        "displayName": "Protocol",
        "identifier": "objective-c.protocol"
      },
      "location": {
        "character": 11,
        "line": 1,
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Protocol"
          }
        ],
        "title": "Protocol"
      }
    },
    {
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@protocol"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "AnotherProtocol"
        },
        {
          "kind": "text",
          "spelling": " <"
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:objc(pl)Protocol",
          "spelling": "Protocol"
        },
        {
          "kind": "text",
          "spelling": ">"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(pl)AnotherProtocol"
      },
      "kind": {
        "displayName": "Protocol",
        "identifier": "objective-c.protocol"
      },
      "location": {
        "character": 11,
        "line": 4,
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "AnotherProtocol"
          }
        ],
        "title": "AnotherProtocol"
      }
    }
  ]
}
