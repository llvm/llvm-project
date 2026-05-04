// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -fblocks -triple arm64-apple-macosx -x objective-c-header %s -o %t/output.symbols.json -verify

@interface Foo
// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix NOPARAM
-(void)methodBlockNoParam:(void (^)())block;
// NOPARAM-LABEL: "!testLabel": "c:objc(cs)Foo(im)methodBlockNoParam:"
// NOPARAM:      "declarationFragments": [
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "text",
// NOPARAM-NEXT:     "spelling": "- ("
// NOPARAM-NEXT:   },
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "typeIdentifier",
// NOPARAM-NEXT:     "preciseIdentifier": "c:v",
// NOPARAM-NEXT:     "spelling": "void"
// NOPARAM-NEXT:   },
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "text",
// NOPARAM-NEXT:     "spelling": ") "
// NOPARAM-NEXT:   },
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "identifier",
// NOPARAM-NEXT:     "spelling": "methodBlockNoParam:"
// NOPARAM-NEXT:   },
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "text",
// NOPARAM-NEXT:     "spelling": "("
// NOPARAM-NEXT:   },
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "typeIdentifier",
// NOPARAM-NEXT:     "preciseIdentifier": "c:v",
// NOPARAM-NEXT:     "spelling": "void"
// NOPARAM-NEXT:   },
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "text",
// NOPARAM-NEXT:     "spelling": " (^)()) "
// NOPARAM-NEXT:   },
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "internalParam",
// NOPARAM-NEXT:     "spelling": "block"
// NOPARAM-NEXT:   },
// NOPARAM-NEXT:   {
// NOPARAM-NEXT:     "kind": "text",
// NOPARAM-NEXT:     "spelling": ";"
// NOPARAM-NEXT:   }
// NOPARAM-NEXT: ],
// NOPARAM:      "functionSignature": {
// NOPARAM-NEXT:   "parameters": [
// NOPARAM-NEXT:     {
// NOPARAM-NEXT:       "declarationFragments": [
// NOPARAM-NEXT:         {
// NOPARAM-NEXT:           "kind": "text",
// NOPARAM-NEXT:           "spelling": "("
// NOPARAM-NEXT:         },
// NOPARAM-NEXT:         {
// NOPARAM-NEXT:           "kind": "typeIdentifier",
// NOPARAM-NEXT:           "preciseIdentifier": "c:v",
// NOPARAM-NEXT:           "spelling": "void"
// NOPARAM-NEXT:         },
// NOPARAM-NEXT:         {
// NOPARAM-NEXT:           "kind": "text",
// NOPARAM-NEXT:           "spelling": " (^)()) "
// NOPARAM-NEXT:         },
// NOPARAM-NEXT:         {
// NOPARAM-NEXT:           "kind": "internalParam",
// NOPARAM-NEXT:           "spelling": "block"
// NOPARAM-NEXT:         }
// NOPARAM-NEXT:       ],
// NOPARAM-NEXT:       "name": "block"
// NOPARAM-NEXT:     }
// NOPARAM-NEXT:   ],
// NOPARAM-NEXT:   "returns": [
// NOPARAM-NEXT:     {
// NOPARAM-NEXT:       "kind": "typeIdentifier",
// NOPARAM-NEXT:       "preciseIdentifier": "c:v",
// NOPARAM-NEXT:       "spelling": "void"
// NOPARAM-NEXT:     }
// NOPARAM-NEXT:   ]
// NOPARAM-NEXT: }

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix PARAM
-(void)methodBlockWithParam:(int (^)(int foo))block;
// PARAM-LABEL: "!testLabel": "c:objc(cs)Foo(im)methodBlockWithParam:"
// PARAM:      "declarationFragments": [
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "text",
// PARAM-NEXT:     "spelling": "- ("
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "typeIdentifier",
// PARAM-NEXT:     "preciseIdentifier": "c:v",
// PARAM-NEXT:     "spelling": "void"
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "text",
// PARAM-NEXT:     "spelling": ") "
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "identifier",
// PARAM-NEXT:     "spelling": "methodBlockWithParam:"
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "text",
// PARAM-NEXT:     "spelling": "("
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "typeIdentifier",
// PARAM-NEXT:     "preciseIdentifier": "c:I",
// PARAM-NEXT:     "spelling": "int"
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "text",
// PARAM-NEXT:     "spelling": " (^)("
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "typeIdentifier",
// PARAM-NEXT:     "preciseIdentifier": "c:I",
// PARAM-NEXT:     "spelling": "int"
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "text",
// PARAM-NEXT:     "spelling": " "
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "internalParam",
// PARAM-NEXT:     "spelling": "foo"
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "text",
// PARAM-NEXT:     "spelling": ")) "
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "internalParam",
// PARAM-NEXT:     "spelling": "block"
// PARAM-NEXT:   },
// PARAM-NEXT:   {
// PARAM-NEXT:     "kind": "text",
// PARAM-NEXT:     "spelling": ";"
// PARAM-NEXT:   }
// PARAM-NEXT: ],
// PARAM:      "functionSignature": {
// PARAM-NEXT:   "parameters": [
// PARAM-NEXT:     {
// PARAM-NEXT:       "declarationFragments": [
// PARAM-NEXT:         {
// PARAM-NEXT:           "kind": "text",
// PARAM-NEXT:           "spelling": "("
// PARAM-NEXT:         },
// PARAM-NEXT:         {
// PARAM-NEXT:           "kind": "typeIdentifier",
// PARAM-NEXT:           "preciseIdentifier": "c:I",
// PARAM-NEXT:           "spelling": "int"
// PARAM-NEXT:         },
// PARAM-NEXT:         {
// PARAM-NEXT:           "kind": "text",
// PARAM-NEXT:           "spelling": " (^)("
// PARAM-NEXT:         },
// PARAM-NEXT:         {
// PARAM-NEXT:           "kind": "typeIdentifier",
// PARAM-NEXT:           "preciseIdentifier": "c:I",
// PARAM-NEXT:           "spelling": "int"
// PARAM-NEXT:         },
// PARAM-NEXT:         {
// PARAM-NEXT:           "kind": "text",
// PARAM-NEXT:           "spelling": " "
// PARAM-NEXT:         },
// PARAM-NEXT:         {
// PARAM-NEXT:           "kind": "internalParam",
// PARAM-NEXT:           "spelling": "foo"
// PARAM-NEXT:         },
// PARAM-NEXT:         {
// PARAM-NEXT:           "kind": "text",
// PARAM-NEXT:           "spelling": ")) "
// PARAM-NEXT:         },
// PARAM-NEXT:         {
// PARAM-NEXT:           "kind": "internalParam",
// PARAM-NEXT:           "spelling": "block"
// PARAM-NEXT:         }
// PARAM-NEXT:       ],
// PARAM-NEXT:       "name": "block"
// PARAM-NEXT:     }
// PARAM-NEXT:   ],
// PARAM-NEXT:   "returns": [
// PARAM-NEXT:     {
// PARAM-NEXT:       "kind": "typeIdentifier",
// PARAM-NEXT:       "preciseIdentifier": "c:v",
// PARAM-NEXT:       "spelling": "void"
// PARAM-NEXT:     }
// PARAM-NEXT:   ]
// PARAM-NEXT: }

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix MULTIPARAM
-(void)methodBlockWithMultipleParam:(int (^)(int foo, unsigned baz))block;
// MULTIPARAM-LABEL: "!testLabel": "c:objc(cs)Foo(im)methodBlockWithMultipleParam:"
// MULTIPARAM:      "declarationFragments": [
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": "- ("
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "typeIdentifier",
// MULTIPARAM-NEXT:     "preciseIdentifier": "c:v",
// MULTIPARAM-NEXT:     "spelling": "void"
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": ") "
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "identifier",
// MULTIPARAM-NEXT:     "spelling": "methodBlockWithMultipleParam:"
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": "("
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "typeIdentifier",
// MULTIPARAM-NEXT:     "preciseIdentifier": "c:I",
// MULTIPARAM-NEXT:     "spelling": "int"
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": " (^)("
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "typeIdentifier",
// MULTIPARAM-NEXT:     "preciseIdentifier": "c:I",
// MULTIPARAM-NEXT:     "spelling": "int"
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": " "
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "internalParam",
// MULTIPARAM-NEXT:     "spelling": "foo"
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": ", "
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "typeIdentifier",
// MULTIPARAM-NEXT:     "preciseIdentifier": "c:i",
// MULTIPARAM-NEXT:     "spelling": "unsigned int"
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": " "
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "internalParam",
// MULTIPARAM-NEXT:     "spelling": "baz"
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": ")) "
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "internalParam",
// MULTIPARAM-NEXT:     "spelling": "block"
// MULTIPARAM-NEXT:   },
// MULTIPARAM-NEXT:   {
// MULTIPARAM-NEXT:     "kind": "text",
// MULTIPARAM-NEXT:     "spelling": ";"
// MULTIPARAM-NEXT:   }
// MULTIPARAM-NEXT: ],
// MULTIPARAM:      "functionSignature": {
// MULTIPARAM-NEXT:   "parameters": [
// MULTIPARAM-NEXT:     {
// MULTIPARAM-NEXT:       "declarationFragments": [
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "text",
// MULTIPARAM-NEXT:           "spelling": "("
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "typeIdentifier",
// MULTIPARAM-NEXT:           "preciseIdentifier": "c:I",
// MULTIPARAM-NEXT:           "spelling": "int"
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "text",
// MULTIPARAM-NEXT:           "spelling": " (^)("
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "typeIdentifier",
// MULTIPARAM-NEXT:           "preciseIdentifier": "c:I",
// MULTIPARAM-NEXT:           "spelling": "int"
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "text",
// MULTIPARAM-NEXT:           "spelling": " "
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "internalParam",
// MULTIPARAM-NEXT:           "spelling": "foo"
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "text",
// MULTIPARAM-NEXT:           "spelling": ", "
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "typeIdentifier",
// MULTIPARAM-NEXT:           "preciseIdentifier": "c:i",
// MULTIPARAM-NEXT:           "spelling": "unsigned int"
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "text",
// MULTIPARAM-NEXT:           "spelling": " "
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "internalParam",
// MULTIPARAM-NEXT:           "spelling": "baz"
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "text",
// MULTIPARAM-NEXT:           "spelling": ")) "
// MULTIPARAM-NEXT:         },
// MULTIPARAM-NEXT:         {
// MULTIPARAM-NEXT:           "kind": "internalParam",
// MULTIPARAM-NEXT:           "spelling": "block"
// MULTIPARAM-NEXT:         }
// MULTIPARAM-NEXT:       ],
// MULTIPARAM-NEXT:       "name": "block"
// MULTIPARAM-NEXT:     }
// MULTIPARAM-NEXT:   ],
// MULTIPARAM-NEXT:   "returns": [
// MULTIPARAM-NEXT:     {
// MULTIPARAM-NEXT:       "kind": "typeIdentifier",
// MULTIPARAM-NEXT:       "preciseIdentifier": "c:v",
// MULTIPARAM-NEXT:       "spelling": "void"
// MULTIPARAM-NEXT:     }
// MULTIPARAM-NEXT:   ]
// MULTIPARAM-NEXT: },

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix VARIADIC
-(void)methodBlockVariadic:(int (^)(int foo, ...))block;
// VARIADIC-LABEL: "!testLabel": "c:objc(cs)Foo(im)methodBlockVariadic:"
// VARIADIC:      "declarationFragments": [
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "text",
// VARIADIC-NEXT:     "spelling": "- ("
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "typeIdentifier",
// VARIADIC-NEXT:     "preciseIdentifier": "c:v",
// VARIADIC-NEXT:     "spelling": "void"
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "text",
// VARIADIC-NEXT:     "spelling": ") "
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "identifier",
// VARIADIC-NEXT:     "spelling": "methodBlockVariadic:"
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "text",
// VARIADIC-NEXT:     "spelling": "("
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "typeIdentifier",
// VARIADIC-NEXT:     "preciseIdentifier": "c:I",
// VARIADIC-NEXT:     "spelling": "int"
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "text",
// VARIADIC-NEXT:     "spelling": " (^)("
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "typeIdentifier",
// VARIADIC-NEXT:     "preciseIdentifier": "c:I",
// VARIADIC-NEXT:     "spelling": "int"
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "text",
// VARIADIC-NEXT:     "spelling": " "
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "internalParam",
// VARIADIC-NEXT:     "spelling": "foo"
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "text",
// VARIADIC-NEXT:     "spelling": ", ...)) "
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "internalParam",
// VARIADIC-NEXT:     "spelling": "block"
// VARIADIC-NEXT:   },
// VARIADIC-NEXT:   {
// VARIADIC-NEXT:     "kind": "text",
// VARIADIC-NEXT:     "spelling": ";"
// VARIADIC-NEXT:   }
// VARIADIC-NEXT: ],
// VARIADIC:      "functionSignature": {
// VARIADIC-NEXT:   "parameters": [
// VARIADIC-NEXT:     {
// VARIADIC-NEXT:       "declarationFragments": [
// VARIADIC-NEXT:         {
// VARIADIC-NEXT:           "kind": "text",
// VARIADIC-NEXT:           "spelling": "("
// VARIADIC-NEXT:         },
// VARIADIC-NEXT:         {
// VARIADIC-NEXT:           "kind": "typeIdentifier",
// VARIADIC-NEXT:           "preciseIdentifier": "c:I",
// VARIADIC-NEXT:           "spelling": "int"
// VARIADIC-NEXT:         },
// VARIADIC-NEXT:         {
// VARIADIC-NEXT:           "kind": "text",
// VARIADIC-NEXT:           "spelling": " (^)("
// VARIADIC-NEXT:         },
// VARIADIC-NEXT:         {
// VARIADIC-NEXT:           "kind": "typeIdentifier",
// VARIADIC-NEXT:           "preciseIdentifier": "c:I",
// VARIADIC-NEXT:           "spelling": "int"
// VARIADIC-NEXT:         },
// VARIADIC-NEXT:         {
// VARIADIC-NEXT:           "kind": "text",
// VARIADIC-NEXT:           "spelling": " "
// VARIADIC-NEXT:         },
// VARIADIC-NEXT:         {
// VARIADIC-NEXT:           "kind": "internalParam",
// VARIADIC-NEXT:           "spelling": "foo"
// VARIADIC-NEXT:         },
// VARIADIC-NEXT:         {
// VARIADIC-NEXT:           "kind": "text",
// VARIADIC-NEXT:           "spelling": ", ...)) "
// VARIADIC-NEXT:         },
// VARIADIC-NEXT:         {
// VARIADIC-NEXT:           "kind": "internalParam",
// VARIADIC-NEXT:           "spelling": "block"
// VARIADIC-NEXT:         }
// VARIADIC-NEXT:       ],
// VARIADIC-NEXT:       "name": "block"
// VARIADIC-NEXT:     }
// VARIADIC-NEXT:   ],
// VARIADIC-NEXT:   "returns": [
// VARIADIC-NEXT:     {
// VARIADIC-NEXT:       "kind": "typeIdentifier",
// VARIADIC-NEXT:       "preciseIdentifier": "c:v",
// VARIADIC-NEXT:       "spelling": "void"
// VARIADIC-NEXT:     }
// VARIADIC-NEXT:   ]
// VARIADIC-NEXT: },
@end

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix FUNC
void func(int (^arg)(int foo));
// FUNC-LABEL: "!testLabel": "c:@F@func"
// FUNC:      "declarationFragments": [
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "typeIdentifier",
// FUNC-NEXT:     "preciseIdentifier": "c:v",
// FUNC-NEXT:     "spelling": "void"
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "text",
// FUNC-NEXT:     "spelling": " "
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "identifier",
// FUNC-NEXT:     "spelling": "func"
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "text",
// FUNC-NEXT:     "spelling": "("
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "typeIdentifier",
// FUNC-NEXT:     "preciseIdentifier": "c:I",
// FUNC-NEXT:     "spelling": "int"
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "text",
// FUNC-NEXT:     "spelling": " (^"
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "internalParam",
// FUNC-NEXT:     "spelling": "arg"
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "text",
// FUNC-NEXT:     "spelling": ")("
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "typeIdentifier",
// FUNC-NEXT:     "preciseIdentifier": "c:I",
// FUNC-NEXT:     "spelling": "int"
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "text",
// FUNC-NEXT:     "spelling": " "
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "internalParam",
// FUNC-NEXT:     "spelling": "foo"
// FUNC-NEXT:   },
// FUNC-NEXT:   {
// FUNC-NEXT:     "kind": "text",
// FUNC-NEXT:     "spelling": "));"
// FUNC-NEXT:   }
// FUNC-NEXT: ],
// FUNC:      "functionSignature": {
// FUNC-NEXT:   "parameters": [
// FUNC-NEXT:     {
// FUNC-NEXT:       "declarationFragments": [
// FUNC-NEXT:         {
// FUNC-NEXT:           "kind": "typeIdentifier",
// FUNC-NEXT:           "preciseIdentifier": "c:I",
// FUNC-NEXT:           "spelling": "int"
// FUNC-NEXT:         },
// FUNC-NEXT:         {
// FUNC-NEXT:           "kind": "text",
// FUNC-NEXT:           "spelling": " (^"
// FUNC-NEXT:         },
// FUNC-NEXT:         {
// FUNC-NEXT:           "kind": "internalParam",
// FUNC-NEXT:           "spelling": "arg"
// FUNC-NEXT:         },
// FUNC-NEXT:         {
// FUNC-NEXT:           "kind": "text",
// FUNC-NEXT:           "spelling": ")("
// FUNC-NEXT:         },
// FUNC-NEXT:         {
// FUNC-NEXT:           "kind": "typeIdentifier",
// FUNC-NEXT:           "preciseIdentifier": "c:I",
// FUNC-NEXT:           "spelling": "int"
// FUNC-NEXT:         },
// FUNC-NEXT:         {
// FUNC-NEXT:           "kind": "text",
// FUNC-NEXT:           "spelling": " "
// FUNC-NEXT:         },
// FUNC-NEXT:         {
// FUNC-NEXT:           "kind": "internalParam",
// FUNC-NEXT:           "spelling": "foo"
// FUNC-NEXT:         },
// FUNC-NEXT:         {
// FUNC-NEXT:           "kind": "text",
// FUNC-NEXT:           "spelling": ")"
// FUNC-NEXT:         }
// FUNC-NEXT:       ],
// FUNC-NEXT:       "name": "arg"
// FUNC-NEXT:     }
// FUNC-NEXT:   ],
// FUNC-NEXT:   "returns": [
// FUNC-NEXT:     {
// FUNC-NEXT:       "kind": "typeIdentifier",
// FUNC-NEXT:       "preciseIdentifier": "c:v",
// FUNC-NEXT:       "spelling": "void"
// FUNC-NEXT:     }
// FUNC-NEXT:   ]
// FUNC-NEXT: },

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix GLOBAL
int (^global)(int foo);
// GLOBAL-LABEL: "!testLabel": "c:@global"
// GLOBAL:      "declarationFragments": [
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "typeIdentifier",
// GLOBAL-NEXT:     "preciseIdentifier": "c:I",
// GLOBAL-NEXT:     "spelling": "int"
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "text",
// GLOBAL-NEXT:     "spelling": " (^"
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "identifier",
// GLOBAL-NEXT:     "spelling": "global"
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "text",
// GLOBAL-NEXT:     "spelling": ")("
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "typeIdentifier",
// GLOBAL-NEXT:     "preciseIdentifier": "c:I",
// GLOBAL-NEXT:     "spelling": "int"
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "text",
// GLOBAL-NEXT:     "spelling": " "
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "internalParam",
// GLOBAL-NEXT:     "spelling": "foo"
// GLOBAL-NEXT:   },
// GLOBAL-NEXT:   {
// GLOBAL-NEXT:     "kind": "text",
// GLOBAL-NEXT:     "spelling": ");"
// GLOBAL-NEXT:   }
// GLOBAL-NEXT: ],

///expected-no-diagnostics
