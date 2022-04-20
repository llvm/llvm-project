// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/known_files_only.hmap.json.in >> %t/known_files_only.hmap.json
// RUN: %hmaptool write %t/known_files_only.hmap.json %t/known_files_only.hmap
// RUN: %clang -extract-api --product-name=KnownFilesOnlyHmap -target arm64-apple-macosx \
// RUN: -I%t/known_files_only.hmap -I%t/subdir %t/subdir/subdir1/input.h \
// RUN: %t/subdir/subdir2/known_file.h -o %t/output.json | FileCheck -allow-empty %s

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:
//--- known_files_only.hmap.json.in
{
  "mappings" :
    {
     "subdir2/known_file.h" : "INPUT_DIR/subdir/subdir3/unknown.h"
    }
}

//--- subdir/subdir1/input.h
int num;
#include "subdir2/known_file.h"

//--- subdir/subdir2/known_file.h
int known_num;

//--- subdir/subdir3/unknown.h
// Ensure that these symbols are not emitted in the Symbol Graph.
#ifndef INPUT4_H
#define INPUT4_H

#define HELLO 1
char not_emitted;
void foo(int);
struct Foo { int a; };

#endif

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
    "name": "KnownFilesOnlyHmap",
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
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "num"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@num"
      },
      "kind": {
        "displayName": "Global Variable",
        "identifier": "c.var"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 1
        },
        "uri": "file://INPUT_DIR/subdir/subdir1/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "num"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "num"
          }
        ],
        "title": "num"
      },
      "pathComponents": [
        "num"
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
          "spelling": "known_num"
        }
      ],
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@known_num"
      },
      "kind": {
        "displayName": "Global Variable",
        "identifier": "c.var"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 1
        },
        "uri": "file://INPUT_DIR/subdir/subdir2/known_file.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "known_num"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "known_num"
          }
        ],
        "title": "known_num"
      },
      "pathComponents": [
        "known_num"
      ]
    }
  ]
}
