// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump=json %s | FileCheck %s

#define FOO frobble
#define BAR FOO

void FOO(void);
void BAR(void);

#define BING(x)	x

void BING(quux)(void);

#define BLIP(x, y) x ## y
#define BLAP(x, y) BLIP(x, y)

void BLAP(foo, __COUNTER__)(void);
void BLAP(foo, __COUNTER__)(void);


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "expansionLoc": {
// CHECK-NEXT:    "col": 6,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 6
// CHECK-NEXT:   },
// CHECK-NEXT:   "spellingLoc": {
// CHECK-NEXT:    "col": 13,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 3
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 6
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 14,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 6
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "frobble",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void)"
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "expansionLoc": {
// CHECK-NEXT:    "col": 6,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 7
// CHECK-NEXT:   },
// CHECK-NEXT:   "spellingLoc": {
// CHECK-NEXT:    "col": 13,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 3
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 7
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 14,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 7
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "previousDecl": "0x{{.*}}",
// CHECK-NEXT:  "name": "frobble",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void)"
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "expansionLoc": {
// CHECK-NEXT:    "col": 6,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "isMacroArgExpansion": true,
// CHECK-NEXT:    "line": 11
// CHECK-NEXT:   },
// CHECK-NEXT:   "spellingLoc": {
// CHECK-NEXT:    "col": 11,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 11
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 11
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 21,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 11
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "quux",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void)"
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "expansionLoc": {
// CHECK-NEXT:    "col": 6,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 16
// CHECK-NEXT:   },
// CHECK-NEXT:   "spellingLoc": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "<scratch space>",
// CHECK-NEXT:    "line": 3
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 16
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 33,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 16
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "foo0",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void)"
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "expansionLoc": {
// CHECK-NEXT:    "col": 6,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 17
// CHECK-NEXT:   },
// CHECK-NEXT:   "spellingLoc": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "<scratch space>",
// CHECK-NEXT:    "line": 5
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 17
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 33,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 17
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "foo1",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (void)"
// CHECK-NEXT:  }
// CHECK-NEXT: }
