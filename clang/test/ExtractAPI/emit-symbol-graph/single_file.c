// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 %t/main.c -emit-symbol-graph --pretty-sgf  \
// RUN:   --symbol-graph-dir=%t/SymbolGraphs --product-name=basicfile -triple=x86_64-apple-macosx12.0.0

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/SymbolGraphs/main.c.symbols.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

// CHECK-NOT: error:
// CHECK-NOT: warning:

//--- main.c
int main ()
{
  return 0;
}


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
    "name": "basicfile",
    "platform": {
      "architecture": "x86_64",
      "operatingSystem": {
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
          "spelling": "main"
        },
        {
          "kind": "text",
          "spelling": "();"
        }
      ],
      "functionSignature": {
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:I",
            "spelling": "int"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "c",
        "precise": "c:@F@main"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "c.func"
      },
      "location": {
        "position": {
          "character": 4,
          "line": 0
        },
        "uri": "file://INPUT_DIR/main.c"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "main"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "main"
          }
        ],
        "title": "main"
      },
      "pathComponents": [
        "main"
      ]
    }
  ]
}
