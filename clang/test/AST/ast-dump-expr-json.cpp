// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -ast-dump=json -ast-dump-filter Test %s | FileCheck %s

namespace std {
using size_t = decltype(sizeof(0));

class type_info {
public:
  virtual ~type_info();
  bool operator==(const type_info& rhs) const noexcept;
  bool operator!=(const type_info& rhs) const noexcept;
  type_info(const type_info& rhs) = delete; // cannot be copied
  type_info& operator=(const type_info& rhs) = delete; // cannot be copied
};

class bad_typeid {
public:
  bad_typeid() noexcept;
  bad_typeid(const bad_typeid&) noexcept;
  virtual ~bad_typeid();
  bad_typeid& operator=(const bad_typeid&) noexcept;
  const char* what() const noexcept;
};
} // namespace std
void *operator new(std::size_t, void *ptr);

struct S {
  virtual ~S() = default;

  void func(int);
  template <typename Ty>
  Ty foo();

  int i;
};

struct T : S {};

template <typename>
struct U {};

void TestThrow() {
  throw 12;
  throw;
}

void TestPointerToMember(S obj1, S *obj2, int S::* data, void (S::*call)(int)) {
  obj1.*data;
  obj2->*data;
  (obj1.*call)(12);
  (obj2->*call)(12);
}

void TestCasting(const S *s) {
  const_cast<S *>(s);
  static_cast<const T *>(s);
  dynamic_cast<const T *>(s);
  reinterpret_cast<const int *>(s);
}

template <typename... Ts>
void TestUnaryExpressions(int *p) {
  sizeof...(Ts);
  noexcept(p - p);

  ::new int;
  new (int);
  new int{12};
  new int[2];
  new int[2]{1, 2};
  new (p) int;
  new (p) int{12};

  ::delete p;
  delete [] p;
}

void TestPostfixExpressions(S a, S *p, U<int> *r) {
  a.func(0);
  p->func(0);
  p->template foo<int>();
  a.template foo<float>();
  p->~S();
  a.~S();
  a.~decltype(a)();
  p->::S::~S();
  r->template U<int>::~U();
  typeid(a);
  typeid(S);
  typeid(const volatile S);
}

template <typename... Ts>
void TestPrimaryExpressions(Ts... a) {
  struct V {
    void f() {
      this;

      [this]{};
      [*this]{};
    }
  };

  int b, c;

  [](){};
  [](int a, ...){};
  [a...]{};
  [=]{};
  [=] { return b; };
  [&]{};
  [&] { return c; };
  [b, &c]{ return b + c; };
  [a..., x = 12]{};
  []() constexpr {};
  []() mutable {};
  []() noexcept {};
  []() -> int { return 0; };

  (a + ...);
  (... + a);
  (a + ... + b);
}

namespace NS {
struct X {};
void f(X);
void y(...);
} // namespace NS

void TestADLCall() {
  NS::X x;
  f(x);
  y(x);
}

void TestNonADLCall() {
  NS::X x;
  NS::f(x);
}

void TestNonADLCall2() {
  NS::X x;
  using NS::f;
  f(x);
  y(x);
}

namespace test_adl_call_three {
using namespace NS;
void TestNonADLCall3() {
  X x;
  f(x);
}
} // namespace test_adl_call_three


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 41
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 41
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 44
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestThrow",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 18,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 41
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 44
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXThrowExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 42
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 42
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "IntegerLiteral",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 42
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 42
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "value": "12"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXThrowExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 43
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 43
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue"
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 46
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 46
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 51
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestPointerToMember",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (S, S *, int S::*, void (S::*)(int))"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 28,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 46
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 26,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 28,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "obj1",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "S"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 37,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 46
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 34,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 37,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "obj2",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "S *"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 52,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 46
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 43,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 52,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "data",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int S::*"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 68,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 46
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 58,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 77,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "call",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void (S::*)(int)"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 80,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 46
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 51
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "BinaryOperator",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 47
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 47
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "opcode": ".*",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclRefExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 47
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 47
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "S"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "referencedDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "ParmVarDecl",
// CHECK-NEXT:         "name": "obj1",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "S"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 47
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 47
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int S::*"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 47
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 47
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int S::*"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "data",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int S::*"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "BinaryOperator",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 48
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 10,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 48
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "opcode": "->*",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 48
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 48
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "S *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 48
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 48
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "obj2",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "S *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 48
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 48
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int S::*"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 48
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 48
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int S::*"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "data",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int S::*"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 49
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 18,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 49
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ParenExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 49
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 49
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "BinaryOperator",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 49
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 49
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "<bound member function type>"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "opcode": ".*",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 49
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 49
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "S"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "obj1",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "S"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 10,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 49
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 10,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 49
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "void (S::*)(int)"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 10,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 49
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 10,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 49
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "void (S::*)(int)"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "ParmVarDecl",
// CHECK-NEXT:               "name": "call",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "void (S::*)(int)"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "IntegerLiteral",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 49
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 49
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "value": "12"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 50
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 19,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 50
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ParenExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 50
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 50
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "BinaryOperator",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 50
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 11,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 50
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "<bound member function type>"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "opcode": "->*",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 50
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 50
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "S *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 4,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 50
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 4,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 50
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "S *"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "ParmVarDecl",
// CHECK-NEXT:               "name": "obj2",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "S *"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 50
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 50
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "void (S::*)(int)"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 11,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 50
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 11,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 50
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "void (S::*)(int)"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "ParmVarDecl",
// CHECK-NEXT:               "name": "call",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "void (S::*)(int)"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "IntegerLiteral",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 50
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 50
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "value": "12"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 53
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 53
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 58
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestCasting",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (const S *)"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 27,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 53
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 18,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 53
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 27,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 53
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "s",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "const S *"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 30,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 53
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 58
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXConstCastExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 54
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 20,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 54
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "S *"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "castKind": "NoOp",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 19,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 54
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 19,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 54
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "const S *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "isPartOfExplicitCast": true,
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 19,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 54
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 19,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 54
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "const S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "s",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "const S *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXStaticCastExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 55
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 27,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 55
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const T *"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "castKind": "BaseToDerived",
// CHECK-NEXT:      "path": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "name": "S"
// CHECK-NEXT:       }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 26,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 55
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 26,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 55
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "const S *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "isPartOfExplicitCast": true,
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 26,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 55
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 26,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 55
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "const S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "s",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "const S *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXDynamicCastExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 56
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 28,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 56
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const T *"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "castKind": "Dynamic",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 27,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 56
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 27,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 56
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "const S *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "isPartOfExplicitCast": true,
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 27,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 56
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 27,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 56
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "const S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "s",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "const S *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXReinterpretCastExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 57
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 34,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 57
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const int *"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "castKind": "BitCast",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 33,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 57
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 33,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 57
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "const S *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "LValueToRValue",
// CHECK-NEXT:        "isPartOfExplicitCast": true,
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 33,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 57
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 33,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 57
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "const S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "s",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "const S *"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionTemplateDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 61
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 60
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 75
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestUnaryExpressions",
// CHECK-NEXT:  "templateParams": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "TemplateTypeParmDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 23,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 60
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 11,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 60
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 23,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 60
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isReferenced": true,
// CHECK-NEXT:    "name": "Ts",
// CHECK-NEXT:    "tagUsed": "typename",
// CHECK-NEXT:    "depth": 0,
// CHECK-NEXT:    "index": 0,
// CHECK-NEXT:    "isParameterPack": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ],
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "FunctionDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 61
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 61
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 75
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "name": "TestUnaryExpressions",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void (int *)"
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParmVarDecl",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 32,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 61
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 27,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 61
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 32,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 61
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "isReferenced": true,
// CHECK-NEXT:      "name": "p",
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int *"
// CHECK-NEXT:      }
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CompoundStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 35,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 61
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 1,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 75
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "SizeOfPackExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 62
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 62
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "unsigned long"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "Ts"
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNoexceptExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 63
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 63
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "bool"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "BinaryOperator",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 12,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 63
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 63
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "long"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "opcode": "-",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 12,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 63
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 12,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 63
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 12,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 63
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 12,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 63
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int *"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "ParmVarDecl",
// CHECK-NEXT:               "name": "p",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int *"
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "nonOdrUseReason": "unevaluated"
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 63
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 63
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 63
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 63
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int *"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "ParmVarDecl",
// CHECK-NEXT:               "name": "p",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int *"
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "nonOdrUseReason": "unevaluated"
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 65
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 65
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isGlobal": true,
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(unsigned long)"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 66
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 66
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(unsigned long)"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 67
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 13,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 67
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "initStyle": "list",
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(unsigned long)"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "InitListExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 67
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 13,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 67
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 67
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 67
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "12"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 68
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 12,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 68
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isArray": true,
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new[]",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(unsigned long)"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 11,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 68
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 11,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 68
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "unsigned long"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "IntegralCast",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 68
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 68
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "2"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 69
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 69
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isArray": true,
// CHECK-NEXT:        "initStyle": "list",
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new[]",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(unsigned long)"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 11,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 69
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 11,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 69
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "unsigned long"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "IntegralCast",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 69
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 69
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "2"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "InitListExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 13,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 69
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 69
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int [2]"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 69
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 14,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 69
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "1"
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 17,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 69
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 17,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 69
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "2"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 70
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 70
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isPlacement": true,
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(std::size_t, void *)"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 70
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 70
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "BitCast",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 70
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 70
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 8,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 70
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 8,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 70
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int *"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "ParmVarDecl",
// CHECK-NEXT:               "name": "p",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int *"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXNewExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 71
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 71
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int *"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isPlacement": true,
// CHECK-NEXT:        "initStyle": "list",
// CHECK-NEXT:        "operatorNewDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator new",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void *(std::size_t, void *)"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "InitListExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 71
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 71
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "IntegerLiteral",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 15,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 71
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 15,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 71
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "value": "12"
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 71
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 71
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "BitCast",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ImplicitCastExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 71
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 71
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "rvalue",
// CHECK-NEXT:            "castKind": "LValueToRValue",
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 8,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 71
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 8,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 71
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int *"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "ParmVarDecl",
// CHECK-NEXT:               "name": "p",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int *"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXDeleteExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 73
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 12,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 73
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isGlobal": true,
// CHECK-NEXT:        "operatorDeleteDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator delete",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void (void *) noexcept"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 12,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 73
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 12,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 73
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 12,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 73
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 12,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 73
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "p",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "int *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXDeleteExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 74
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 13,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 74
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "isArray": true,
// CHECK-NEXT:        "isArrayAsWritten": true,
// CHECK-NEXT:        "operatorDeleteDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "FunctionDecl",
// CHECK-NEXT:         "name": "operator delete[]",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "void (void *) noexcept"
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 13,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 74
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 13,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 74
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 13,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 74
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 13,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 74
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "p",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "int *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 77
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 77
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 90
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestPostfixExpressions",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void (S, S *, U<int> *)"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 31,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 77
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 29,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 31,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "a",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "S"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 37,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 77
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 34,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 37,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "p",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "S *"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParmVarDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 48,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 77
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 40,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 48,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isUsed": true,
// CHECK-NEXT:    "name": "r",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "U<int> *"
// CHECK-NEXT:    }
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 51,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 77
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 90
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 78
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 78
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 78
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 78
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "func",
// CHECK-NEXT:        "isArrow": false,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 78
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 78
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "a",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "S"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "IntegerLiteral",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 78
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 78
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "value": "0"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 79
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 12,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 79
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 79
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 6,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 79
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "func",
// CHECK-NEXT:        "isArrow": true,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 79
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 79
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 79
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 79
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "S *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "p",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "S *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "IntegerLiteral",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 79
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 79
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "int"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "value": "0"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 80
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 24,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 80
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "desugaredQualType": "int",
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 80
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 22,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 80
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "foo",
// CHECK-NEXT:        "isArrow": true,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 80
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 80
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 80
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 80
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "S *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "p",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "S *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 81
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 25,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 81
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "desugaredQualType": "float",
// CHECK-NEXT:       "qualType": "float"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 81
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 23,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 81
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "foo",
// CHECK-NEXT:        "isArrow": false,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 81
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 81
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "a",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "S"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 82
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 9,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 82
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 82
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 82
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "~S",
// CHECK-NEXT:        "isArrow": true,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 82
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 82
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 82
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 82
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "S *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "p",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "S *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 83
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 83
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 83
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 6,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 83
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "~S",
// CHECK-NEXT:        "isArrow": false,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 83
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 83
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "a",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "S"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 84
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 18,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 84
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 84
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 84
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "~S",
// CHECK-NEXT:        "isArrow": false,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 84
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 84
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "a",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "S"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 85
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 14,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 85
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 85
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 12,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 85
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "~S",
// CHECK-NEXT:        "isArrow": true,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 85
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 85
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "S *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 85
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 85
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "S *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "p",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "S *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMemberCallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 86
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 26,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 86
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "MemberExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 86
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 24,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 86
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<bound member function type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "name": "~U",
// CHECK-NEXT:        "isArrow": true,
// CHECK-NEXT:        "referencedMemberDecl": "0x{{.*}}",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 86
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 86
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "U<int> *"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 86
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 86
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "U<int> *"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "r",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "U<int> *"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXTypeidExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 87
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 87
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const std::type_info"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclRefExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 87
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 87
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "S"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "referencedDecl": {
// CHECK-NEXT:         "id": "0x{{.*}}",
// CHECK-NEXT:         "kind": "ParmVarDecl",
// CHECK-NEXT:         "name": "a",
// CHECK-NEXT:         "type": {
// CHECK-NEXT:          "qualType": "S"
// CHECK-NEXT:         }
// CHECK-NEXT:        }
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXTypeidExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 88
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 11,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 88
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const std::type_info"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "typeArg": {
// CHECK-NEXT:       "qualType": "S"
// CHECK-NEXT:      }
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXTypeidExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 89
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 26,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 89
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "const std::type_info"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "lvalue",
// CHECK-NEXT:      "typeArg": {
// CHECK-NEXT:       "qualType": "const volatile S"
// CHECK-NEXT:      },
// CHECK-NEXT:      "adjustedTypeArg": {
// CHECK-NEXT:       "qualType": "S"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionTemplateDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 93
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 92
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 122
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestPrimaryExpressions",
// CHECK-NEXT:  "templateParams": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "TemplateTypeParmDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 23,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 92
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 11,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 92
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 23,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 92
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "isReferenced": true,
// CHECK-NEXT:    "name": "Ts",
// CHECK-NEXT:    "tagUsed": "typename",
// CHECK-NEXT:    "depth": 0,
// CHECK-NEXT:    "index": 0,
// CHECK-NEXT:    "isParameterPack": true
// CHECK-NEXT:   }
// CHECK-NEXT:  ],
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "FunctionDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 93
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 93
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 122
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "name": "TestPrimaryExpressions",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void (Ts...)"
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParmVarDecl",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 35,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 93
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 29,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 93
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 35,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 93
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "isReferenced": true,
// CHECK-NEXT:      "name": "a",
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "Ts..."
// CHECK-NEXT:      },
// CHECK-NEXT:      "isParameterPack": true
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CompoundStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 38,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 93
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 1,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 122
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 94
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 4,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 101
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 10,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 94
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 94
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 101
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "name": "V",
// CHECK-NEXT:          "tagUsed": "struct",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true,
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "isConstexpr": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:           "isAggregate": true,
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isPOD": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTrivial": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXRecordDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 10,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 94
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 94
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 10,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 94
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "V",
// CHECK-NEXT:            "tagUsed": "struct"
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 10,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 95
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 95
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 100
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "f",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "void ()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 14,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 95
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 5,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 100
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "CXXThisExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 7,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 96
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 7,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 96
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "V *"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "rvalue"
// CHECK-NEXT:               },
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "LambdaExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 7,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 98
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 14,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 98
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "(lambda at {{.*}}:98:7)"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "rvalue",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "CXXRecordDecl",
// CHECK-NEXT:                  "loc": {
// CHECK-NEXT:                   "col": 7,
// CHECK-NEXT:                   "file": "{{.*}}",
// CHECK-NEXT:                   "line": 98
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 7,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 98
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 7,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 98
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "isImplicit": true,
// CHECK-NEXT:                  "tagUsed": "class",
// CHECK-NEXT:                  "completeDefinition": true,
// CHECK-NEXT:                  "definitionData": {
// CHECK-NEXT:                   "canConstDefaultInit": true,
// CHECK-NEXT:                   "copyAssign": {
// CHECK-NEXT:                    "hasConstParam": true,
// CHECK-NEXT:                    "implicitHasConstParam": true,
// CHECK-NEXT:                    "needsImplicit": true,
// CHECK-NEXT:                    "trivial": true
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "copyCtor": {
// CHECK-NEXT:                    "hasConstParam": true,
// CHECK-NEXT:                    "implicitHasConstParam": true,
// CHECK-NEXT:                    "needsImplicit": true,
// CHECK-NEXT:                    "simple": true,
// CHECK-NEXT:                    "trivial": true
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "defaultCtor": {},
// CHECK-NEXT:                   "dtor": {
// CHECK-NEXT:                    "irrelevant": true,
// CHECK-NEXT:                    "needsImplicit": true,
// CHECK-NEXT:                    "simple": true,
// CHECK-NEXT:                    "trivial": true
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "isLambda": true,
// CHECK-NEXT:                   "isStandardLayout": true,
// CHECK-NEXT:                   "isTriviallyCopyable": true,
// CHECK-NEXT:                   "moveAssign": {},
// CHECK-NEXT:                   "moveCtor": {
// CHECK-NEXT:                    "exists": true,
// CHECK-NEXT:                    "needsImplicit": true,
// CHECK-NEXT:                    "simple": true,
// CHECK-NEXT:                    "trivial": true
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "inner": [
// CHECK-NEXT:                   {
// CHECK-NEXT:                    "id": "0x{{.*}}",
// CHECK-NEXT:                    "kind": "CXXMethodDecl",
// CHECK-NEXT:                    "loc": {
// CHECK-NEXT:                     "col": 7,
// CHECK-NEXT:                     "file": "{{.*}}",
// CHECK-NEXT:                     "line": 98
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                      "col": 12,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 98
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                      "col": 14,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 98
// CHECK-NEXT:                     }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "name": "operator()",
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                     "qualType": "auto () const -> auto"
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "inline": true,
// CHECK-NEXT:                    "inner": [
// CHECK-NEXT:                     {
// CHECK-NEXT:                      "id": "0x{{.*}}",
// CHECK-NEXT:                      "kind": "CompoundStmt",
// CHECK-NEXT:                      "range": {
// CHECK-NEXT:                       "begin": {
// CHECK-NEXT:                        "col": 13,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 98
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "end": {
// CHECK-NEXT:                        "col": 14,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 98
// CHECK-NEXT:                       }
// CHECK-NEXT:                      }
// CHECK-NEXT:                     }
// CHECK-NEXT:                    ]
// CHECK-NEXT:                   },
// CHECK-NEXT:                   {
// CHECK-NEXT:                    "id": "0x{{.*}}",
// CHECK-NEXT:                    "kind": "FieldDecl",
// CHECK-NEXT:                    "loc": {
// CHECK-NEXT:                     "col": 8,
// CHECK-NEXT:                     "file": "{{.*}}",
// CHECK-NEXT:                     "line": 98
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                      "col": 8,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 98
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                      "col": 8,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 98
// CHECK-NEXT:                     }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "isImplicit": true,
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                     "qualType": "V *"
// CHECK-NEXT:                    }
// CHECK-NEXT:                   }
// CHECK-NEXT:                  ]
// CHECK-NEXT:                 },
// CHECK-NEXT:                 {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "ParenListExpr",
// CHECK-NEXT:                   "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                       "col": 8,
// CHECK-NEXT:                       "file": "{{.*}}",
// CHECK-NEXT:                       "line": 98
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                       "col": 8,
// CHECK-NEXT:                       "file": "{{.*}}",
// CHECK-NEXT:                       "line": 98
// CHECK-NEXT:                     }
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                     "qualType": "NULL TYPE"
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "valueCategory": "rvalue",
// CHECK-NEXT:                   "inner": [
// CHECK-NEXT:                     {
// CHECK-NEXT:                       "id": "0x{{.*}}",
// CHECK-NEXT:                       "kind": "CXXThisExpr",
// CHECK-NEXT:                       "range": {
// CHECK-NEXT:                         "begin": {
// CHECK-NEXT:                           "col": 8,
// CHECK-NEXT:                           "file": "{{.*}}",
// CHECK-NEXT:                           "line": 98
// CHECK-NEXT:                         },
// CHECK-NEXT:                         "end": {
// CHECK-NEXT:                           "col": 8,
// CHECK-NEXT:                           "file": "{{.*}}",
// CHECK-NEXT:                           "line": 98
// CHECK-NEXT:                         }
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "type": {
// CHECK-NEXT:                         "qualType": "V *"
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "valueCategory": "rvalue"
// CHECK-NEXT:                     }
// CHECK-NEXT:                   ]
// CHECK-NEXT:                 },
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "CompoundStmt",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 13,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 98
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 98
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               },
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "LambdaExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 7,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 99
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 15,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 99
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "(lambda at {{.*}}:99:7)"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "rvalue",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "CXXRecordDecl",
// CHECK-NEXT:                  "loc": {
// CHECK-NEXT:                   "col": 7,
// CHECK-NEXT:                   "file": "{{.*}}",
// CHECK-NEXT:                   "line": 99
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 7,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 7,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "isImplicit": true,
// CHECK-NEXT:                  "tagUsed": "class",
// CHECK-NEXT:                  "completeDefinition": true,
// CHECK-NEXT:                  "definitionData": {
// CHECK-NEXT:                   "canConstDefaultInit": true,
// CHECK-NEXT:                   "copyAssign": {
// CHECK-NEXT:                    "hasConstParam": true,
// CHECK-NEXT:                    "implicitHasConstParam": true,
// CHECK-NEXT:                    "needsImplicit": true,
// CHECK-NEXT:                    "trivial": true
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "copyCtor": {
// CHECK-NEXT:                    "hasConstParam": true,
// CHECK-NEXT:                    "implicitHasConstParam": true,
// CHECK-NEXT:                    "needsImplicit": true,
// CHECK-NEXT:                    "simple": true,
// CHECK-NEXT:                    "trivial": true
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "defaultCtor": {
// CHECK-NEXT:                    "defaultedIsConstexpr": true
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "dtor": {
// CHECK-NEXT:                    "irrelevant": true,
// CHECK-NEXT:                    "needsImplicit": true,
// CHECK-NEXT:                    "simple": true,
// CHECK-NEXT:                    "trivial": true
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "isLambda": true,
// CHECK-NEXT:                   "isStandardLayout": true,
// CHECK-NEXT:                   "isTriviallyCopyable": true,
// CHECK-NEXT:                   "moveAssign": {},
// CHECK-NEXT:                   "moveCtor": {
// CHECK-NEXT:                    "exists": true,
// CHECK-NEXT:                    "needsImplicit": true,
// CHECK-NEXT:                    "simple": true,
// CHECK-NEXT:                    "trivial": true
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "inner": [
// CHECK-NEXT:                   {
// CHECK-NEXT:                    "id": "0x{{.*}}",
// CHECK-NEXT:                    "kind": "CXXMethodDecl",
// CHECK-NEXT:                    "loc": {
// CHECK-NEXT:                     "col": 7,
// CHECK-NEXT:                     "file": "{{.*}}",
// CHECK-NEXT:                     "line": 99
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                      "col": 13,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 99
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                      "col": 15,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 99
// CHECK-NEXT:                     }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "name": "operator()",
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                     "qualType": "auto () const -> auto"
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "inline": true,
// CHECK-NEXT:                    "inner": [
// CHECK-NEXT:                     {
// CHECK-NEXT:                      "id": "0x{{.*}}",
// CHECK-NEXT:                      "kind": "CompoundStmt",
// CHECK-NEXT:                      "range": {
// CHECK-NEXT:                       "begin": {
// CHECK-NEXT:                        "col": 14,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 99
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "end": {
// CHECK-NEXT:                        "col": 15,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 99
// CHECK-NEXT:                       }
// CHECK-NEXT:                      }
// CHECK-NEXT:                     }
// CHECK-NEXT:                    ]
// CHECK-NEXT:                   },
// CHECK-NEXT:                   {
// CHECK-NEXT:                    "id": "0x{{.*}}",
// CHECK-NEXT:                    "kind": "FieldDecl",
// CHECK-NEXT:                    "loc": {
// CHECK-NEXT:                     "col": 8,
// CHECK-NEXT:                     "file": "{{.*}}",
// CHECK-NEXT:                     "line": 99
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                      "col": 8,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 99
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                      "col": 8,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 99
// CHECK-NEXT:                     }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "isImplicit": true,
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                     "qualType": "V"
// CHECK-NEXT:                    }
// CHECK-NEXT:                   }
// CHECK-NEXT:                  ]
// CHECK-NEXT:                 },
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "ParenListExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 8,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 8,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "NULL TYPE"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "rvalue",
// CHECK-NEXT:                  "inner": [
// CHECK-NEXT:                   {
// CHECK-NEXT:                    "id": "0x{{.*}}",
// CHECK-NEXT:                    "kind": "UnaryOperator",
// CHECK-NEXT:                    "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                      "col": 8,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 99
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                      "col": 8,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 99
// CHECK-NEXT:                     }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                     "qualType": "<dependent type>"
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "valueCategory": "rvalue",
// CHECK-NEXT:                    "isPostfix": false,
// CHECK-NEXT:                    "opcode": "*",
// CHECK-NEXT:                    "canOverflow": false,
// CHECK-NEXT:                    "inner": [
// CHECK-NEXT:                     {
// CHECK-NEXT:                      "id": "0x{{.*}}",
// CHECK-NEXT:                      "kind": "CXXThisExpr",
// CHECK-NEXT:                      "range": {
// CHECK-NEXT:                       "begin": {
// CHECK-NEXT:                        "col": 8,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 99
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "end": {
// CHECK-NEXT:                        "col": 8,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 99
// CHECK-NEXT:                       }
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                       "qualType": "V *"
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "valueCategory": "rvalue"
// CHECK-NEXT:                     }
// CHECK-NEXT:                    ]
// CHECK-NEXT:                   }
// CHECK-NEXT:                  ]
// CHECK-NEXT:                 },
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "CompoundStmt",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 14,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 15,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 99
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "DeclStmt",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 103
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 103
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 7,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 103
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 103
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 103
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isReferenced": true,
// CHECK-NEXT:          "name": "b",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          }
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "VarDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 10,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 103
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 103
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 103
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isReferenced": true,
// CHECK-NEXT:          "name": "c",
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 105
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 8,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 105
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:105:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 105
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 105
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 105
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 105
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 6,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 105
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 105
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 7,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 105
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 8,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 105
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXConversionDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 105
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 105
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 105
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "operator auto (*)()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto (*() const noexcept)()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "constexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 105
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 105
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 105
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "__invoke",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto ()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "storageClass": "static",
// CHECK-NEXT:            "inline": true
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 105
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 105
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 106
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 106
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:106:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 106
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 106
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 106
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 106
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 106
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 106
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto (int, ...) const"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "variadic": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "ParmVarDecl",
// CHECK-NEXT:              "loc": {
// CHECK-NEXT:               "col": 10,
// CHECK-NEXT:               "file": "{{.*}}",
// CHECK-NEXT:               "line": 106
// CHECK-NEXT:              },
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 6,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 106
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 10,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 106
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "name": "a",
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              }
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 17,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 106
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 18,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 106
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXConversionDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 106
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 106
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 106
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "operator auto (*)(int, ...)",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto (*() const noexcept)(int, ...)"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "constexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 106
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 106
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 106
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "__invoke",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto (int, ...)"
// CHECK-NEXT:            },
// CHECK-NEXT:            "storageClass": "static",
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "variadic": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "ParmVarDecl",
// CHECK-NEXT:              "loc": {
// CHECK-NEXT:               "col": 10,
// CHECK-NEXT:               "file": "{{.*}}",
// CHECK-NEXT:               "line": 106
// CHECK-NEXT:              },
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 6,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 106
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 10,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 106
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "name": "a",
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 106
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 106
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 107
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 10,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 107
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:107:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 107
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 107
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 107
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {},
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 107
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 107
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 10,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 107
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const -> auto"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 9,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 107
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 10,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 107
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "FieldDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 4,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 107
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 107
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 107
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "Ts..."
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ParenListExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 107
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 107
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "NULL TYPE"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 107
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 107
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "Ts..."
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "a",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "Ts..."
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 107
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 107
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 108
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 108
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:108:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 108
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 108
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 108
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 108
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 108
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 7,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 108
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const -> auto"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 6,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 108
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 7,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 108
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 6,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 108
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 108
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 109
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 19,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 109
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:109:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 109
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 109
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 109
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 109
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 109
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 19,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 109
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const -> auto"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 7,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 109
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 19,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 109
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ReturnStmt",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 9,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 109
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 16,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 109
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 16,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 109
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 16,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 109
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "const int"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "b",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "qualType": "int"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 109
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 19,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 109
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ReturnStmt",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 9,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 109
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 109
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 109
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 109
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "const int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "VarDecl",
// CHECK-NEXT:               "name": "b",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 110
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 110
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:110:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 110
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 110
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 110
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 110
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 110
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 7,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 110
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const -> auto"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 6,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 110
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 7,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 110
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 6,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 110
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 110
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 111
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 19,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 111
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:111:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 111
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 111
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 111
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 111
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 111
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 19,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 111
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const -> auto"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 7,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 111
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 19,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 111
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ReturnStmt",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 9,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 111
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 16,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 111
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 16,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 111
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 16,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 111
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "int"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "c",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "qualType": "int"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 111
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 19,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 111
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ReturnStmt",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 9,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 111
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 111
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "DeclRefExpr",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 111
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 111
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "lvalue",
// CHECK-NEXT:              "referencedDecl": {
// CHECK-NEXT:               "id": "0x{{.*}}",
// CHECK-NEXT:               "kind": "VarDecl",
// CHECK-NEXT:               "name": "c",
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                "qualType": "int"
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 112
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 26,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 112
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:112:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 112
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 112
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 112
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {},
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 112
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 9,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 26,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const -> auto"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 10,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 112
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 26,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 112
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ReturnStmt",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 12,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 112
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 23,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 112
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "BinaryOperator",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 19,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 112
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 23,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 112
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "int"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "rvalue",
// CHECK-NEXT:                  "opcode": "+",
// CHECK-NEXT:                  "inner": [
// CHECK-NEXT:                   {
// CHECK-NEXT:                    "id": "0x{{.*}}",
// CHECK-NEXT:                    "kind": "ImplicitCastExpr",
// CHECK-NEXT:                    "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                      "col": 19,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 112
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                      "col": 19,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 112
// CHECK-NEXT:                     }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                     "qualType": "int"
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "valueCategory": "rvalue",
// CHECK-NEXT:                    "castKind": "LValueToRValue",
// CHECK-NEXT:                    "inner": [
// CHECK-NEXT:                     {
// CHECK-NEXT:                      "id": "0x{{.*}}",
// CHECK-NEXT:                      "kind": "DeclRefExpr",
// CHECK-NEXT:                      "range": {
// CHECK-NEXT:                       "begin": {
// CHECK-NEXT:                        "col": 19,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 112
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "end": {
// CHECK-NEXT:                        "col": 19,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 112
// CHECK-NEXT:                       }
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                       "qualType": "const int"
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "valueCategory": "lvalue",
// CHECK-NEXT:                      "referencedDecl": {
// CHECK-NEXT:                       "id": "0x{{.*}}",
// CHECK-NEXT:                       "kind": "VarDecl",
// CHECK-NEXT:                       "name": "b",
// CHECK-NEXT:                       "type": {
// CHECK-NEXT:                        "qualType": "int"
// CHECK-NEXT:                       }
// CHECK-NEXT:                      }
// CHECK-NEXT:                     }
// CHECK-NEXT:                    ]
// CHECK-NEXT:                   },
// CHECK-NEXT:                   {
// CHECK-NEXT:                    "id": "0x{{.*}}",
// CHECK-NEXT:                    "kind": "ImplicitCastExpr",
// CHECK-NEXT:                    "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                      "col": 23,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 112
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                      "col": 23,
// CHECK-NEXT:                      "file": "{{.*}}",
// CHECK-NEXT:                      "line": 112
// CHECK-NEXT:                     }
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "type": {
// CHECK-NEXT:                     "qualType": "int"
// CHECK-NEXT:                    },
// CHECK-NEXT:                    "valueCategory": "rvalue",
// CHECK-NEXT:                    "castKind": "LValueToRValue",
// CHECK-NEXT:                    "inner": [
// CHECK-NEXT:                     {
// CHECK-NEXT:                      "id": "0x{{.*}}",
// CHECK-NEXT:                      "kind": "DeclRefExpr",
// CHECK-NEXT:                      "range": {
// CHECK-NEXT:                       "begin": {
// CHECK-NEXT:                        "col": 23,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 112
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "end": {
// CHECK-NEXT:                        "col": 23,
// CHECK-NEXT:                        "file": "{{.*}}",
// CHECK-NEXT:                        "line": 112
// CHECK-NEXT:                       }
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "type": {
// CHECK-NEXT:                       "qualType": "int"
// CHECK-NEXT:                      },
// CHECK-NEXT:                      "valueCategory": "lvalue",
// CHECK-NEXT:                      "referencedDecl": {
// CHECK-NEXT:                       "id": "0x{{.*}}",
// CHECK-NEXT:                       "kind": "VarDecl",
// CHECK-NEXT:                       "name": "c",
// CHECK-NEXT:                       "type": {
// CHECK-NEXT:                        "qualType": "int"
// CHECK-NEXT:                       }
// CHECK-NEXT:                      }
// CHECK-NEXT:                     }
// CHECK-NEXT:                    ]
// CHECK-NEXT:                   }
// CHECK-NEXT:                  ]
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "FieldDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 4,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 112
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            }
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "FieldDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 8,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 112
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int &"
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 112
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 112
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "castKind": "LValueToRValue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "b",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "int"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 112
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 8,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 112
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "c",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 112
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 26,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 112
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ReturnStmt",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 12,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 23,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 112
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "BinaryOperator",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 19,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 112
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 23,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 112
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "opcode": "+",
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ImplicitCastExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 19,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 112
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 19,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 112
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "int"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "rvalue",
// CHECK-NEXT:                "castKind": "LValueToRValue",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 19,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 112
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 19,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 112
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "const int"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "b",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "qualType": "int"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               },
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ImplicitCastExpr",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 23,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 112
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 23,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 112
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "type": {
// CHECK-NEXT:                 "qualType": "int"
// CHECK-NEXT:                },
// CHECK-NEXT:                "valueCategory": "rvalue",
// CHECK-NEXT:                "castKind": "LValueToRValue",
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "DeclRefExpr",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 23,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 112
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 23,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 112
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "int"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "lvalue",
// CHECK-NEXT:                  "referencedDecl": {
// CHECK-NEXT:                   "id": "0x{{.*}}",
// CHECK-NEXT:                   "kind": "VarDecl",
// CHECK-NEXT:                   "name": "c",
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                    "qualType": "int"
// CHECK-NEXT:                   }
// CHECK-NEXT:                  }
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 113
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 113
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:113:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 113
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 113
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 113
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {},
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 113
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 16,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 113
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 113
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const -> auto"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 17,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 113
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 18,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 113
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "FieldDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 4,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 113
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 113
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 113
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "Ts..."
// CHECK-NEXT:            }
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "FieldDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 10,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 113
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 10,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 113
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 10,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 113
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "int",
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ParenListExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 113
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 113
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "NULL TYPE"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 113
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 4,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 113
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "Ts..."
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "ParmVarDecl",
// CHECK-NEXT:             "name": "a",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "Ts..."
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "IntegerLiteral",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 113
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 113
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue",
// CHECK-NEXT:          "value": "12"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 113
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 113
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 114
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 19,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 114
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:114:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 114
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 114
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 114
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 114
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 114
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 19,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 114
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "constexpr": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 18,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 114
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 19,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 114
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXConversionDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 114
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 114
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 19,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 114
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "operator auto (*)()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto (*() const noexcept)()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "constexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 114
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 114
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 19,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 114
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "__invoke",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto ()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "storageClass": "static",
// CHECK-NEXT:            "inline": true
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 114
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 19,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 114
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 115
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 17,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 115
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:115:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 115
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 115
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 115
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 115
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 115
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 17,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 115
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto ()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 16,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 115
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 17,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 115
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXConversionDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 115
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 115
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 17,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 115
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "operator auto (*)()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto (*() const noexcept)()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "constexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 115
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 115
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 17,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 115
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "__invoke",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto ()"
// CHECK-NEXT:            },
// CHECK-NEXT:            "storageClass": "static",
// CHECK-NEXT:            "inline": true
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 16,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 115
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 115
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 116
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 116
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:116:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 116
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 116
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 116
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 116
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 8,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 116
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 116
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const noexcept"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 17,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 116
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 18,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 116
// CHECK-NEXT:               }
// CHECK-NEXT:              }
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXConversionDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 116
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 116
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 116
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "operator auto (*)() noexcept",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto (*() const noexcept)() noexcept"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "constexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 116
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 116
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 18,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 116
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "__invoke",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () noexcept"
// CHECK-NEXT:            },
// CHECK-NEXT:            "storageClass": "static",
// CHECK-NEXT:            "inline": true
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 17,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 116
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 18,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 116
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "LambdaExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 117
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 27,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 117
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "(lambda at {{.*}}:117:3)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXRecordDecl",
// CHECK-NEXT:          "loc": {
// CHECK-NEXT:           "col": 3,
// CHECK-NEXT:           "file": "{{.*}}",
// CHECK-NEXT:           "line": 117
// CHECK-NEXT:          },
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 117
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 117
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "isImplicit": true,
// CHECK-NEXT:          "tagUsed": "class",
// CHECK-NEXT:          "completeDefinition": true,
// CHECK-NEXT:          "definitionData": {
// CHECK-NEXT:           "canConstDefaultInit": true,
// CHECK-NEXT:           "copyAssign": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "copyCtor": {
// CHECK-NEXT:            "hasConstParam": true,
// CHECK-NEXT:            "implicitHasConstParam": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "defaultCtor": {
// CHECK-NEXT:            "defaultedIsConstexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "dtor": {
// CHECK-NEXT:            "irrelevant": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           },
// CHECK-NEXT:           "isEmpty": true,
// CHECK-NEXT:           "isLambda": true,
// CHECK-NEXT:           "isLiteral": true,
// CHECK-NEXT:           "isStandardLayout": true,
// CHECK-NEXT:           "isTriviallyCopyable": true,
// CHECK-NEXT:           "moveAssign": {},
// CHECK-NEXT:           "moveCtor": {
// CHECK-NEXT:            "exists": true,
// CHECK-NEXT:            "needsImplicit": true,
// CHECK-NEXT:            "simple": true,
// CHECK-NEXT:            "trivial": true
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 117
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 11,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 117
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 27,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 117
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "name": "operator()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () const -> int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "CompoundStmt",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 15,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 117
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 27,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 117
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "inner": [
// CHECK-NEXT:               {
// CHECK-NEXT:                "id": "0x{{.*}}",
// CHECK-NEXT:                "kind": "ReturnStmt",
// CHECK-NEXT:                "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                  "col": 17,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 117
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                  "col": 24,
// CHECK-NEXT:                  "file": "{{.*}}",
// CHECK-NEXT:                  "line": 117
// CHECK-NEXT:                 }
// CHECK-NEXT:                },
// CHECK-NEXT:                "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                  "id": "0x{{.*}}",
// CHECK-NEXT:                  "kind": "IntegerLiteral",
// CHECK-NEXT:                  "range": {
// CHECK-NEXT:                   "begin": {
// CHECK-NEXT:                    "col": 24,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 117
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "end": {
// CHECK-NEXT:                    "col": 24,
// CHECK-NEXT:                    "file": "{{.*}}",
// CHECK-NEXT:                    "line": 117
// CHECK-NEXT:                   }
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                   "qualType": "int"
// CHECK-NEXT:                  },
// CHECK-NEXT:                  "valueCategory": "rvalue",
// CHECK-NEXT:                  "value": "0"
// CHECK-NEXT:                 }
// CHECK-NEXT:                ]
// CHECK-NEXT:               }
// CHECK-NEXT:              ]
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXConversionDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 117
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 117
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 27,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 117
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "operator int (*)()",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto (*() const noexcept)() -> int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "inline": true,
// CHECK-NEXT:            "constexpr": true
// CHECK-NEXT:           },
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "CXXMethodDecl",
// CHECK-NEXT:            "loc": {
// CHECK-NEXT:             "col": 3,
// CHECK-NEXT:             "file": "{{.*}}",
// CHECK-NEXT:             "line": 117
// CHECK-NEXT:            },
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 3,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 117
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 27,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 117
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "isImplicit": true,
// CHECK-NEXT:            "name": "__invoke",
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "auto () -> int"
// CHECK-NEXT:            },
// CHECK-NEXT:            "storageClass": "static",
// CHECK-NEXT:            "inline": true
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CompoundStmt",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 15,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 117
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 27,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 117
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "ReturnStmt",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 17,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 117
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 24,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 117
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:              "id": "0x{{.*}}",
// CHECK-NEXT:              "kind": "IntegerLiteral",
// CHECK-NEXT:              "range": {
// CHECK-NEXT:               "begin": {
// CHECK-NEXT:                "col": 24,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 117
// CHECK-NEXT:               },
// CHECK-NEXT:               "end": {
// CHECK-NEXT:                "col": 24,
// CHECK-NEXT:                "file": "{{.*}}",
// CHECK-NEXT:                "line": 117
// CHECK-NEXT:               }
// CHECK-NEXT:              },
// CHECK-NEXT:              "type": {
// CHECK-NEXT:               "qualType": "int"
// CHECK-NEXT:              },
// CHECK-NEXT:              "valueCategory": "rvalue",
// CHECK-NEXT:              "value": "0"
// CHECK-NEXT:             }
// CHECK-NEXT:            ]
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXFoldExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 119
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 119
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<dependent type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 119
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 119
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "Ts..."
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "a",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "Ts..."
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         },
// CHECK-NEXT:         {}
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXFoldExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 120
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 120
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<dependent type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {},
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 120
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 10,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 120
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "Ts..."
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "a",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "Ts..."
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXFoldExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 121
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 121
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<dependent type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 121
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 4,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 121
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "Ts..."
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "ParmVarDecl",
// CHECK-NEXT:           "name": "a",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "Ts..."
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 121
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 14,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 121
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "int"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "VarDecl",
// CHECK-NEXT:           "name": "b",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "int"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 130
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 130
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 134
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestADLCall",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 20,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 130
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 134
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 131
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 10,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 131
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 9,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 131
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 131
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 131
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "isUsed": true,
// CHECK-NEXT:        "name": "x",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "desugaredQualType": "NS::X",
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "init": "call",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXConstructExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 131
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 131
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "NS::X",
// CHECK-NEXT:           "qualType": "NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue"
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 132
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 6,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 132
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "adl": true,
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 132
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 132
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (*)(NS::X)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "FunctionToPointerDecay",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 132
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 132
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void (NS::X)"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "FunctionDecl",
// CHECK-NEXT:           "name": "f",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "void (NS::X)"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXConstructExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 132
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 132
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 132
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 132
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "const NS::X",
// CHECK-NEXT:           "qualType": "const NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "castKind": "NoOp",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 132
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 132
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "NS::X",
// CHECK-NEXT:             "qualType": "NS::X"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "x",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "NS::X",
// CHECK-NEXT:              "qualType": "NS::X"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 133
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 6,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 133
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "adl": true,
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 133
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 133
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (*)(...)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "FunctionToPointerDecay",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 133
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 133
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void (...)"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "FunctionDecl",
// CHECK-NEXT:           "name": "y",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "void (...)"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXConstructExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 133
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 133
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "desugaredQualType": "NS::X",
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 133
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 133
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "const NS::X",
// CHECK-NEXT:           "qualType": "const NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "castKind": "NoOp",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 133
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 133
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "NS::X",
// CHECK-NEXT:             "qualType": "NS::X"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "x",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "NS::X",
// CHECK-NEXT:              "qualType": "NS::X"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 136
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 136
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 139
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestNonADLCall",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 23,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 136
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 139
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 137
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 10,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 137
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 9,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 137
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 137
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 137
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "isUsed": true,
// CHECK-NEXT:        "name": "x",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "desugaredQualType": "NS::X",
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "init": "call",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXConstructExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 137
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 137
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "NS::X",
// CHECK-NEXT:           "qualType": "NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue"
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 138
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 10,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 138
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 138
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 7,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 138
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (*)(NS::X)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "FunctionToPointerDecay",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 138
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 7,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 138
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void (NS::X)"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "FunctionDecl",
// CHECK-NEXT:           "name": "f",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "void (NS::X)"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXConstructExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 138
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 138
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 138
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 138
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "const NS::X",
// CHECK-NEXT:           "qualType": "const NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "castKind": "NoOp",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 9,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 138
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 9,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 138
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "NS::X",
// CHECK-NEXT:             "qualType": "NS::X"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "x",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "NS::X",
// CHECK-NEXT:              "qualType": "NS::X"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 141
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 141
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 146
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestNonADLCall2",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 24,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 141
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 146
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 142
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 10,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 142
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 9,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 142
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 142
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 9,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 142
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "isUsed": true,
// CHECK-NEXT:        "name": "x",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "desugaredQualType": "NS::X",
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "init": "call",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXConstructExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 142
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 9,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 142
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "NS::X",
// CHECK-NEXT:           "qualType": "NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue"
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 143
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 14,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 143
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "UsingDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 13,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 143
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 143
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 13,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 143
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "name": "NS::f"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 144
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 6,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 144
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 144
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 144
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (*)(NS::X)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "FunctionToPointerDecay",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 144
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 144
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void (NS::X)"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "FunctionDecl",
// CHECK-NEXT:           "name": "f",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "void (NS::X)"
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "foundReferencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "UsingShadowDecl",
// CHECK-NEXT:           "name": "f"
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXConstructExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 144
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 144
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 144
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 144
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "const NS::X",
// CHECK-NEXT:           "qualType": "const NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "castKind": "NoOp",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 144
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 144
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "NS::X",
// CHECK-NEXT:             "qualType": "NS::X"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "x",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "NS::X",
// CHECK-NEXT:              "qualType": "NS::X"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 145
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 6,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 145
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "adl": true,
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 145
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 145
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (*)(...)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "FunctionToPointerDecay",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 145
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 145
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void (...)"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "FunctionDecl",
// CHECK-NEXT:           "name": "y",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "void (...)"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXConstructExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 145
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 145
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "desugaredQualType": "NS::X",
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 145
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 145
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "desugaredQualType": "const NS::X",
// CHECK-NEXT:           "qualType": "const NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "castKind": "NoOp",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 145
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 145
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "desugaredQualType": "NS::X",
// CHECK-NEXT:             "qualType": "NS::X"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "x",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "desugaredQualType": "NS::X",
// CHECK-NEXT:              "qualType": "NS::X"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FunctionDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 6,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 150
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 150
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 153
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestNonADLCall3",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "void ()"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundStmt",
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 24,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 150
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 153
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "DeclStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 151
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 6,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 151
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "VarDecl",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 5,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 151
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 151
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 151
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "isUsed": true,
// CHECK-NEXT:        "name": "x",
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "init": "call",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "CXXConstructExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 151
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 151
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "rvalue"
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CallExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 152
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 6,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 152
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "rvalue",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ImplicitCastExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 152
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 3,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 152
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "void (*)(NS::X)"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "castKind": "FunctionToPointerDecay",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "DeclRefExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 152
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 3,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 152
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "void (NS::X)"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "referencedDecl": {
// CHECK-NEXT:           "id": "0x{{.*}}",
// CHECK-NEXT:           "kind": "FunctionDecl",
// CHECK-NEXT:           "name": "f",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:            "qualType": "void (NS::X)"
// CHECK-NEXT:           }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "CXXConstructExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 152
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 5,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 152
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "NS::X"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "rvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "ImplicitCastExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 152
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "col": 5,
// CHECK-NEXT:            "file": "{{.*}}",
// CHECK-NEXT:            "line": 152
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "const NS::X"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "castKind": "NoOp",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "DeclRefExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 152
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "col": 5,
// CHECK-NEXT:              "file": "{{.*}}",
// CHECK-NEXT:              "line": 152
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "NS::X"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "referencedDecl": {
// CHECK-NEXT:             "id": "0x{{.*}}",
// CHECK-NEXT:             "kind": "VarDecl",
// CHECK-NEXT:             "name": "x",
// CHECK-NEXT:             "type": {
// CHECK-NEXT:              "qualType": "NS::X"
// CHECK-NEXT:             }
// CHECK-NEXT:            }
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }
