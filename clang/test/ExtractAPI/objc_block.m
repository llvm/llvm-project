// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s@INPUT_DIR@%{/t:regex_replacement}@g" \
// RUN: %t/reference.output.json.in >> %t/reference.output.json
// RUN: %clang_cc1 -extract-api -fblocks -triple arm64-apple-macosx \
// RUN: -x objective-c-header %t/input.h -o %t/output.json -verify

// Generator version is not consistent across test runs, normalize it.
// RUN: sed -e "s@\"generator\": \".*\"@\"generator\": \"?\"@g" \
// RUN: %t/output.json >> %t/output-normalized.json
// RUN: diff %t/reference.output.json %t/output-normalized.json

//--- input.h
@interface Foo
-(void)methodBlockNoParam:(void (^)())block;
-(void)methodBlockWithParam:(int (^)(int foo))block;
-(void)methodBlockWithMultipleParam:(int (^)(int foo, unsigned baz))block;
-(void)methodBlockVariadic:(int (^)(int foo, ...))block;
@end

void func(int (^arg)(int foo));

int (^global)(int foo);

///expected-no-diagnostics

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
      "source": "c:objc(cs)Foo(im)methodBlockNoParam:",
      "target": "c:objc(cs)Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Foo(im)methodBlockWithParam:",
      "target": "c:objc(cs)Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Foo(im)methodBlockWithMultipleParam:",
      "target": "c:objc(cs)Foo",
      "targetFallback": "Foo"
    },
    {
      "kind": "memberOf",
      "source": "c:objc(cs)Foo(im)methodBlockVariadic:",
      "target": "c:objc(cs)Foo",
      "targetFallback": "Foo"
    }
  ],
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
          "spelling": " (^"
        },
        {
          "kind": "identifier",
          "spelling": "global"
        },
        {
          "kind": "text",
          "spelling": ")("
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
          "kind": "internalParam",
          "spelling": "foo"
        },
        {
          "kind": "text",
          "spelling": ");"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:@global"
      },
      "kind": {
        "displayName": "Global Variable",
        "identifier": "objective-c.var"
      },
      "location": {
        "position": {
          "character": 6,
          "line": 9
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "global"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "global"
          }
        ],
        "title": "global"
      },
      "pathComponents": [
        "global"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "func"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " (^"
        },
        {
          "kind": "internalParam",
          "spelling": "arg"
        },
        {
          "kind": "text",
          "spelling": ")("
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
          "kind": "internalParam",
          "spelling": "foo"
        },
        {
          "kind": "text",
          "spelling": "));"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:I",
                "spelling": "int"
              },
              {
                "kind": "text",
                "spelling": " (^"
              },
              {
                "kind": "internalParam",
                "spelling": "arg"
              },
              {
                "kind": "text",
                "spelling": ")("
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
                "kind": "internalParam",
                "spelling": "foo"
              },
              {
                "kind": "text",
                "spelling": ")"
              }
            ],
            "name": "arg"
          }
        ],
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:@F@func"
      },
      "kind": {
        "displayName": "Function",
        "identifier": "objective-c.func"
      },
      "location": {
        "position": {
          "character": 5,
          "line": 7
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "func"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "func"
          }
        ],
        "title": "func"
      },
      "pathComponents": [
        "func"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "keyword",
          "spelling": "@interface"
        },
        {
          "kind": "text",
          "spelling": " "
        },
        {
          "kind": "identifier",
          "spelling": "Foo"
        }
      ],
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Foo"
      },
      "kind": {
        "displayName": "Class",
        "identifier": "objective-c.class"
      },
      "location": {
        "position": {
          "character": 11,
          "line": 0
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "Foo"
          }
        ],
        "subHeading": [
          {
            "kind": "identifier",
            "spelling": "Foo"
          }
        ],
        "title": "Foo"
      },
      "pathComponents": [
        "Foo"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "methodBlockNoParam:"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": " (^"
        },
        {
          "kind": "text",
          "spelling": ")()) "
        },
        {
          "kind": "internalParam",
          "spelling": "block"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "text",
                "spelling": "("
              },
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:v",
                "spelling": "void"
              },
              {
                "kind": "text",
                "spelling": " (^"
              },
              {
                "kind": "text",
                "spelling": ")()) "
              },
              {
                "kind": "internalParam",
                "spelling": "block"
              }
            ],
            "name": "block"
          }
        ],
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Foo(im)methodBlockNoParam:"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 0,
          "line": 1
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "methodBlockNoParam:"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "methodBlockNoParam:"
          }
        ],
        "title": "methodBlockNoParam:"
      },
      "pathComponents": [
        "Foo",
        "methodBlockNoParam:"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "methodBlockWithParam:"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " (^"
        },
        {
          "kind": "text",
          "spelling": ")("
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
          "kind": "internalParam",
          "spelling": "foo"
        },
        {
          "kind": "text",
          "spelling": ")) "
        },
        {
          "kind": "internalParam",
          "spelling": "block"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "text",
                "spelling": "("
              },
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:I",
                "spelling": "int"
              },
              {
                "kind": "text",
                "spelling": " (^"
              },
              {
                "kind": "text",
                "spelling": ")("
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
                "kind": "internalParam",
                "spelling": "foo"
              },
              {
                "kind": "text",
                "spelling": ")) "
              },
              {
                "kind": "internalParam",
                "spelling": "block"
              }
            ],
            "name": "block"
          }
        ],
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Foo(im)methodBlockWithParam:"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 0,
          "line": 2
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "methodBlockWithParam:"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "methodBlockWithParam:"
          }
        ],
        "title": "methodBlockWithParam:"
      },
      "pathComponents": [
        "Foo",
        "methodBlockWithParam:"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "methodBlockWithMultipleParam:"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " (^"
        },
        {
          "kind": "text",
          "spelling": ")("
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
          "kind": "internalParam",
          "spelling": "foo"
        },
        {
          "kind": "text",
          "spelling": ", "
        },
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
          "kind": "internalParam",
          "spelling": "baz"
        },
        {
          "kind": "text",
          "spelling": ")) "
        },
        {
          "kind": "internalParam",
          "spelling": "block"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "text",
                "spelling": "("
              },
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:I",
                "spelling": "int"
              },
              {
                "kind": "text",
                "spelling": " (^"
              },
              {
                "kind": "text",
                "spelling": ")("
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
                "kind": "internalParam",
                "spelling": "foo"
              },
              {
                "kind": "text",
                "spelling": ", "
              },
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
                "kind": "internalParam",
                "spelling": "baz"
              },
              {
                "kind": "text",
                "spelling": ")) "
              },
              {
                "kind": "internalParam",
                "spelling": "block"
              }
            ],
            "name": "block"
          }
        ],
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Foo(im)methodBlockWithMultipleParam:"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 0,
          "line": 3
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "methodBlockWithMultipleParam:"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "methodBlockWithMultipleParam:"
          }
        ],
        "title": "methodBlockWithMultipleParam:"
      },
      "pathComponents": [
        "Foo",
        "methodBlockWithMultipleParam:"
      ]
    },
    {
      "accessLevel": "public",
      "declarationFragments": [
        {
          "kind": "text",
          "spelling": "- ("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:v",
          "spelling": "void"
        },
        {
          "kind": "text",
          "spelling": ") "
        },
        {
          "kind": "identifier",
          "spelling": "methodBlockVariadic:"
        },
        {
          "kind": "text",
          "spelling": "("
        },
        {
          "kind": "typeIdentifier",
          "preciseIdentifier": "c:I",
          "spelling": "int"
        },
        {
          "kind": "text",
          "spelling": " (^"
        },
        {
          "kind": "text",
          "spelling": ")("
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
          "kind": "internalParam",
          "spelling": "foo"
        },
        {
          "kind": "text",
          "spelling": ", ...)) "
        },
        {
          "kind": "internalParam",
          "spelling": "block"
        },
        {
          "kind": "text",
          "spelling": ";"
        }
      ],
      "functionSignature": {
        "parameters": [
          {
            "declarationFragments": [
              {
                "kind": "text",
                "spelling": "("
              },
              {
                "kind": "typeIdentifier",
                "preciseIdentifier": "c:I",
                "spelling": "int"
              },
              {
                "kind": "text",
                "spelling": " (^"
              },
              {
                "kind": "text",
                "spelling": ")("
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
                "kind": "internalParam",
                "spelling": "foo"
              },
              {
                "kind": "text",
                "spelling": ", ...)) "
              },
              {
                "kind": "internalParam",
                "spelling": "block"
              }
            ],
            "name": "block"
          }
        ],
        "returns": [
          {
            "kind": "typeIdentifier",
            "preciseIdentifier": "c:v",
            "spelling": "void"
          }
        ]
      },
      "identifier": {
        "interfaceLanguage": "objective-c",
        "precise": "c:objc(cs)Foo(im)methodBlockVariadic:"
      },
      "kind": {
        "displayName": "Instance Method",
        "identifier": "objective-c.method"
      },
      "location": {
        "position": {
          "character": 0,
          "line": 4
        },
        "uri": "file://INPUT_DIR/input.h"
      },
      "names": {
        "navigator": [
          {
            "kind": "identifier",
            "spelling": "methodBlockVariadic:"
          }
        ],
        "subHeading": [
          {
            "kind": "text",
            "spelling": "- "
          },
          {
            "kind": "identifier",
            "spelling": "methodBlockVariadic:"
          }
        ],
        "title": "methodBlockVariadic:"
      },
      "pathComponents": [
        "Foo",
        "methodBlockVariadic:"
      ]
    }
  ]
}
