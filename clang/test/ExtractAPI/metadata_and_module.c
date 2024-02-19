// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --product-name=module -triple arm64-apple-macosx -x c-header %s -o %t/module.symbols.json -verify

// RUN: Filecheck %s --input-file %t/module.symbols.json --check-prefix METADATA
// RUN: Filecheck %s --input-file %t/module.symbols.json --check-prefix MOD

// expected-no-diagnostics

// METADATA:      "metadata": {
// METADATA-NEXT:   "formatVersion": {
// METADATA-NEXT:     "major":
// METADATA-NEXT:     "minor":
// METADATA-NEXT:     "patch":
// METADATA-NEXT:   },
// METADATA-NEXT:   "generator":
// METADATA-NEXT: }

// MOD: "module": {
// MOD-NEXT:   "name": "module",
// MOD-NEXT:   "platform": {
// MOD-NEXT:     "architecture": "arm64",
// MOD-NEXT:     "operatingSystem": {
// MOD-NEXT:       "minimumVersion": {
// MOD-NEXT:         "major":
// MOD-NEXT:         "minor":
// MOD-NEXT:         "patch":
// MOD-NEXT:       },
// MOD-NEXT:       "name": "macosx"
// MOD-NEXT:     },
// MOD-NEXT:     "vendor": "apple"
// MOD-NEXT:   }
// MOD-NEXT: }
