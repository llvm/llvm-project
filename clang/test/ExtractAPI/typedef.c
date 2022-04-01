// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%/t@g" %t/reference.output.json.in >> \
// RUN: %t/reference.output.json
// RUN: %clang -extract-api --product-name=Typedef -target arm64-apple-macosx \
// RUN: -x objective-c-header %t/input.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- input.h
typedef int MyInt;

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
    "name": "Typedef",
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
          "spelling": "typedef"
        },
        {
          "kind": "text",
          "spelling": " "
        },
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
          "spelling": "MyInt"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:input.h@T@MyInt"
      },
      "kind": {
        "displayName": "Type Alias",
        "identifier": "objective-c.typealias"
      },
      "location": {
        "position": {
          "character": 13,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "MyInt"
          }
        ],
        "title": "MyInt"
      },
      "pathComponents": [
        "MyInt"
      ],
      "type": "c:I"
    }
  ]
}
