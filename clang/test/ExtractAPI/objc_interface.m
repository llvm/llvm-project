// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -x objective-c-header -triple arm64-apple-macosx %s -o %t/output.symbols.json -verify

@protocol Protocol
@end

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix SUPER
@interface Super <Protocol>
// SUPER: "!testRelLabel": "conformsTo $ c:objc(cs)Super $ c:objc(pl)Protocol"
// SUPER-LABEL: "!testLabel": "c:objc(cs)Super"
// SUPER: "accessLevel": "public",
// SUPER:      "declarationFragments": [
// SUPER-NEXT:   {
// SUPER-NEXT:     "kind": "keyword",
// SUPER-NEXT:     "spelling": "@interface"
// SUPER-NEXT:   },
// SUPER-NEXT:   {
// SUPER-NEXT:     "kind": "text",
// SUPER-NEXT:     "spelling": " "
// SUPER-NEXT:   },
// SUPER-NEXT:   {
// SUPER-NEXT:     "kind": "identifier",
// SUPER-NEXT:     "spelling": "Super"
// SUPER-NEXT:   }
// SUPER-NEXT: ],
// SUPER:      "kind": {
// SUPER-NEXT:   "displayName": "Class",
// SUPER-NEXT:   "identifier": "objective-c.class"
// SUPER-NEXT: },
// SUPER:   "title": "Super"
// SUPER:      "pathComponents": [
// SUPER-NEXT:   "Super"
// SUPER-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix PROP
@property(readonly, getter=getProperty) unsigned Property;
// PROP: "!testRelLabel": "memberOf $ c:objc(cs)Super(py)Property $ c:objc(cs)Super"
// PROP: "!testLabel": "c:objc(cs)Super(py)Property"
// PROP: "accessLevel": "public",
// PROP:      "declarationFragments": [
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "keyword",
// PROP-NEXT:     "spelling": "@property"
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "text",
// PROP-NEXT:     "spelling": " ("
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "keyword",
// PROP-NEXT:     "spelling": "readonly"
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "text",
// PROP-NEXT:     "spelling": ", "
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "keyword",
// PROP-NEXT:     "spelling": "getter"
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "text",
// PROP-NEXT:     "spelling": "="
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "identifier",
// PROP-NEXT:     "spelling": "getProperty"
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "text",
// PROP-NEXT:     "spelling": ") "
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "typeIdentifier",
// PROP-NEXT:     "preciseIdentifier": "c:i",
// PROP-NEXT:     "spelling": "unsigned int"
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "text",
// PROP-NEXT:     "spelling": " "
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "identifier",
// PROP-NEXT:     "spelling": "Property"
// PROP-NEXT:   },
// PROP-NEXT:   {
// PROP-NEXT:     "kind": "text",
// PROP-NEXT:     "spelling": ";"
// PROP-NEXT:   }
// PROP-NEXT: ],
// PROP:      "kind": {
// PROP-NEXT:   "displayName": "Instance Property",
// PROP-NEXT:   "identifier": "objective-c.property"
// PROP-NEXT: },
// PROP:   "title": "Property"
// PROP:      "pathComponents": [
// PROP-NEXT:   "Super",
// PROP-NEXT:   "Property"
// PROP-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix GET
+ (id)getWithProperty:(unsigned) Property;
// GET: "!testRelLabel": "memberOf $ c:objc(cs)Super(cm)getWithProperty: $ c:objc(cs)Super"
// GET-LABEL: "!testLabel": "c:objc(cs)Super(cm)getWithProperty:"
// GET: "accessLevel": "public",
// GET:      "declarationFragments": [
// GET-NEXT:   {
// GET-NEXT:     "kind": "text",
// GET-NEXT:     "spelling": "+ ("
// GET-NEXT:   },
// GET-NEXT:   {
// GET-NEXT:     "kind": "keyword",
// GET-NEXT:     "spelling": "id"
// GET-NEXT:   },
// GET-NEXT:   {
// GET-NEXT:     "kind": "text",
// GET-NEXT:     "spelling": ") "
// GET-NEXT:   },
// GET-NEXT:   {
// GET-NEXT:     "kind": "identifier",
// GET-NEXT:     "spelling": "getWithProperty:"
// GET-NEXT:   },
// GET-NEXT:   {
// GET-NEXT:     "kind": "text",
// GET-NEXT:     "spelling": "("
// GET-NEXT:   },
// GET-NEXT:   {
// GET-NEXT:     "kind": "typeIdentifier",
// GET-NEXT:     "preciseIdentifier": "c:i",
// GET-NEXT:     "spelling": "unsigned int"
// GET-NEXT:   },
// GET-NEXT:   {
// GET-NEXT:     "kind": "text",
// GET-NEXT:     "spelling": ") "
// GET-NEXT:   },
// GET-NEXT:   {
// GET-NEXT:     "kind": "internalParam",
// GET-NEXT:     "spelling": "Property"
// GET-NEXT:   },
// GET-NEXT:   {
// GET-NEXT:     "kind": "text",
// GET-NEXT:     "spelling": ";"
// GET-NEXT:   }
// GET-NEXT: ],
// GET:      "functionSignature": {
// GET-NEXT:   "parameters": [
// GET-NEXT:     {
// GET-NEXT:       "declarationFragments": [
// GET-NEXT:         {
// GET-NEXT:           "kind": "text",
// GET-NEXT:           "spelling": "("
// GET-NEXT:         },
// GET-NEXT:         {
// GET-NEXT:           "kind": "typeIdentifier",
// GET-NEXT:           "preciseIdentifier": "c:i",
// GET-NEXT:           "spelling": "unsigned int"
// GET-NEXT:         },
// GET-NEXT:         {
// GET-NEXT:           "kind": "text",
// GET-NEXT:           "spelling": ") "
// GET-NEXT:         },
// GET-NEXT:         {
// GET-NEXT:           "kind": "internalParam",
// GET-NEXT:           "spelling": "Property"
// GET-NEXT:         }
// GET-NEXT:       ],
// GET-NEXT:       "name": "Property"
// GET-NEXT:     }
// GET-NEXT:   ],
// GET-NEXT:   "returns": [
// GET-NEXT:     {
// GET-NEXT:       "kind": "keyword",
// GET-NEXT:       "spelling": "id"
// GET-NEXT:     }
// GET-NEXT:   ]
// GET-NEXT: },
// GET:      "kind": {
// GET-NEXT:   "displayName": "Type Method",
// GET-NEXT:   "identifier": "objective-c.type.method"
// GET-NEXT: },
// GET:   "title": "getWithProperty:"
// GET:      "pathComponents": [
// GET-NEXT:   "Super",
// GET-NEXT:   "getWithProperty:"
// GET-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix SET
- (void)setProperty:(unsigned) Property andOtherThing: (unsigned) Thing;
// SET: "!testRelLabel": "memberOf $ c:objc(cs)Super(im)setProperty:andOtherThing: $ c:objc(cs)Super"
// SET-LABEL: "!testLabel": "c:objc(cs)Super(im)setProperty:andOtherThing:"
// SET: "accessLevel": "public",
// SET:      "declarationFragments": [
// SET-NEXT:   {
// SET-NEXT:     "kind": "text",
// SET-NEXT:     "spelling": "- ("
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "typeIdentifier",
// SET-NEXT:     "preciseIdentifier": "c:v",
// SET-NEXT:     "spelling": "void"
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "text",
// SET-NEXT:     "spelling": ") "
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "identifier",
// SET-NEXT:     "spelling": "setProperty:"
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "text",
// SET-NEXT:     "spelling": "("
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "typeIdentifier",
// SET-NEXT:     "preciseIdentifier": "c:i",
// SET-NEXT:     "spelling": "unsigned int"
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "text",
// SET-NEXT:     "spelling": ") "
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "internalParam",
// SET-NEXT:     "spelling": "Property"
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "text",
// SET-NEXT:     "spelling": " "
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "identifier",
// SET-NEXT:     "spelling": "andOtherThing:"
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "text",
// SET-NEXT:     "spelling": "("
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "typeIdentifier",
// SET-NEXT:     "preciseIdentifier": "c:i",
// SET-NEXT:     "spelling": "unsigned int"
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "text",
// SET-NEXT:     "spelling": ") "
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "internalParam",
// SET-NEXT:     "spelling": "Thing"
// SET-NEXT:   },
// SET-NEXT:   {
// SET-NEXT:     "kind": "text",
// SET-NEXT:     "spelling": ";"
// SET-NEXT:   }
// SET-NEXT: ],
// SET:      "functionSignature": {
// SET-NEXT:   "parameters": [
// SET-NEXT:     {
// SET-NEXT:       "declarationFragments": [
// SET-NEXT:         {
// SET-NEXT:           "kind": "text",
// SET-NEXT:           "spelling": "("
// SET-NEXT:         },
// SET-NEXT:         {
// SET-NEXT:           "kind": "typeIdentifier",
// SET-NEXT:           "preciseIdentifier": "c:i",
// SET-NEXT:           "spelling": "unsigned int"
// SET-NEXT:         },
// SET-NEXT:         {
// SET-NEXT:           "kind": "text",
// SET-NEXT:           "spelling": ") "
// SET-NEXT:         },
// SET-NEXT:         {
// SET-NEXT:           "kind": "internalParam",
// SET-NEXT:           "spelling": "Property"
// SET-NEXT:         }
// SET-NEXT:       ],
// SET-NEXT:       "name": "Property"
// SET-NEXT:     },
// SET-NEXT:     {
// SET-NEXT:       "declarationFragments": [
// SET-NEXT:         {
// SET-NEXT:           "kind": "text",
// SET-NEXT:           "spelling": "("
// SET-NEXT:         },
// SET-NEXT:         {
// SET-NEXT:           "kind": "typeIdentifier",
// SET-NEXT:           "preciseIdentifier": "c:i",
// SET-NEXT:           "spelling": "unsigned int"
// SET-NEXT:         },
// SET-NEXT:         {
// SET-NEXT:           "kind": "text",
// SET-NEXT:           "spelling": ") "
// SET-NEXT:         },
// SET-NEXT:         {
// SET-NEXT:           "kind": "internalParam",
// SET-NEXT:           "spelling": "Thing"
// SET-NEXT:         }
// SET-NEXT:       ],
// SET-NEXT:       "name": "Thing"
// SET-NEXT:     }
// SET-NEXT:   ],
// SET-NEXT:   "returns": [
// SET-NEXT:     {
// SET-NEXT:       "kind": "typeIdentifier",
// SET-NEXT:       "preciseIdentifier": "c:v",
// SET-NEXT:       "spelling": "void"
// SET-NEXT:     }
// SET-NEXT:   ]
// SET-NEXT: },
// SET:      "kind": {
// SET-NEXT:   "displayName": "Instance Method",
// SET-NEXT:   "identifier": "objective-c.method"
// SET-NEXT: },
// SET:   "title": "setProperty:andOtherThing:"
@end

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix DERIVED
@interface Derived : Super {
// DERIVED: "!testRelLabel": "inheritsFrom $ c:objc(cs)Derived $ c:objc(cs)Super"

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix IVAR
  char Ivar;
// IVAR: "!testRelLabel": "memberOf $ c:objc(cs)Derived@Ivar $ c:objc(cs)Derived"
// IVAR-LABEL: "!testLabel": "c:objc(cs)Derived@Ivar"
// IVAR: "accessLevel": "public",
// IVAR:      "declarationFragments": [
// IVAR-NEXT:   {
// IVAR-NEXT:     "kind": "typeIdentifier",
// IVAR-NEXT:     "preciseIdentifier": "c:C",
// IVAR-NEXT:     "spelling": "char"
// IVAR-NEXT:   },
// IVAR-NEXT:   {
// IVAR-NEXT:     "kind": "text",
// IVAR-NEXT:     "spelling": " "
// IVAR-NEXT:   },
// IVAR-NEXT:   {
// IVAR-NEXT:     "kind": "identifier",
// IVAR-NEXT:     "spelling": "Ivar"
// IVAR-NEXT:   },
// IVAR-NEXT:   {
// IVAR-NEXT:     "kind": "text",
// IVAR-NEXT:     "spelling": ";"
// IVAR-NEXT:   }
// IVAR-NEXT: ],
// IVAR:      "kind": {
// IVAR-NEXT:   "displayName": "Instance Variable",
// IVAR-NEXT:   "identifier": "objective-c.ivar"
// IVAR-NEXT: },
// IVAR: "title": "Ivar"
// IVAR:      "pathComponents": [
// IVAR-NEXT:   "Derived",
// IVAR-NEXT:   "Ivar"
// IVAR-NEXT: ]
}
@end

// expected-no-diagnostics
