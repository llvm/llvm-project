// RUN: %clang_cc1 -Wdocumentation -ast-dump=json %s | FileCheck %s

/// Aaa
int TestLocation;

///
int TestIndent;

/// Aaa
int Test_TextComment;

/// \brief Aaa
int Test_BlockCommandComment;

/// \param Aaa xxx
/// \param [in,out] Bbb yyy
void Test_ParamCommandComment(int Aaa, int Bbb);

/// \tparam Aaa xxx
template <typename Aaa> class Test_TParamCommandComment;

/// \c Aaa
int Test_InlineCommandComment;

/// <a>Aaa</a>
/// <br/>
int Test_HTMLTagComment;

/// \verbatim
/// Aaa
/// \endverbatim
int Test_VerbatimBlockComment;

/// \param ... More arguments
template<typename T>
void Test_TemplatedFunctionVariadic(int arg, ...);


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 3
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 3
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 7,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 3
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 3
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 3
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 3
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 3
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 3
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 7,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 3
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " Aaa"
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {},
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {},
// CHECK-NEXT:   "end": {}
// CHECK-NEXT:  }
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 9
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 9
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 7,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 9
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 9
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 9
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 7,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 9
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 9
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 9
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 7,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 9
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " Aaa"
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 12
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 12
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 14,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 12
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 12
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 12
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 12
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 12
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 12
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 12
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "BlockCommandComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 12
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 5,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 12
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 14,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 12
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "name": "brief",
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParagraphComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 11,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 12
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 11,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 12
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 14,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 12
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "TextComment",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 11,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 12
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 11,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 12
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 14,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 12
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "text": " Aaa"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 15
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 15
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 27,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 16
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 15
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 15
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 15
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 15
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 15
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 15
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParamCommandComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 15
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 5,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 15
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 16
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "direction": "in",
// CHECK-NEXT:    "param": "Aaa",
// CHECK-NEXT:    "paramIdx": 0,
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParagraphComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 15,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 15
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 15,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 15
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 16
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "TextComment",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 15,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 15
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 15
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 18,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 15
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "text": " xxx"
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "TextComment",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 4,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 16
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 4,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 16
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 4,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 16
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "text": " "
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParamCommandComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 16
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 5,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 16
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 27,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 16
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "direction": "in,out",
// CHECK-NEXT:    "explicit": true,
// CHECK-NEXT:    "param": "Bbb",
// CHECK-NEXT:    "paramIdx": 1,
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParagraphComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 24,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 16
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 24,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 16
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 27,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 16
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "TextComment",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 24,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 16
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 24,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 16
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 27,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 16
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "text": " yyy"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 19
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 19
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 19,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 19
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 19
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 19
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "TParamCommandComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 19
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 5,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 19,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "param": "Aaa",
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParagraphComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 16,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 19
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 16,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 19,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "TextComment",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 16,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 19
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 19
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 19,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 19
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "text": " xxx"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 19
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 19
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 19,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 19
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 19
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 19
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "TParamCommandComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 19
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 5,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 19,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 19
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "param": "Aaa",
// CHECK-NEXT:    "positions": [
// CHECK-NEXT:     0
// CHECK-NEXT:    ],
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParagraphComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 16,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 19
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 16,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 19,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 19
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "TextComment",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 16,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 19
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 16,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 19
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 19,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 19
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "text": " xxx"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 22
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 22
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 6,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 22
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 22
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 22
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 6,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 22
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 22
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 22
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 22
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "InlineCommandComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 5,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 22
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 22
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 6,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 22
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "name": "c",
// CHECK-NEXT:      "renderKind": "monospaced",
// CHECK-NEXT:      "args": [
// CHECK-NEXT:       "Aaa"
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 25
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 25
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 8,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 26
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 25
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 25
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 8,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 26
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 25
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "HTMLStartTagComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 6,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 25
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 7,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "name": "a"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 8,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 25
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 8,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 10,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": "Aaa"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "HTMLEndTagComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 13,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 25
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 11,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 14,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 25
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "name": "a"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 26
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 26
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 26
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "HTMLStartTagComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 6,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 26
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 5,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 26
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 26
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "name": "br",
// CHECK-NEXT:      "selfClosing": true
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 29
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 29
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 14,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 29
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 29
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 29
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 29
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 29
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 29
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 29
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "VerbatimBlockComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 29
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 5,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 29
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 14,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 29
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "name": "verbatim",
// CHECK-NEXT:    "closeName": "endverbatim",
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "VerbatimBlockLineComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 30
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 30
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 8,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 30
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " Aaa"
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 34
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 34
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 29,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 34
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 34
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 34
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 34
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 34
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParamCommandComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 34
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 5,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 34
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 29,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 34
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "direction": "in",
// CHECK-NEXT:    "param": "...",
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParagraphComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 15,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 34
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 15,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 29,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "TextComment",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 15,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 34
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 34
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 29,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 34
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "text": " More arguments"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }


// CHECK:  "kind": "FullComment",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "col": 4,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 34
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "col": 4,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 34
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "col": 29,
// CHECK-NEXT:    "file": "{{.*}}",
// CHECK-NEXT:    "line": 34
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParagraphComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 4,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 34
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 34
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 4,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 34
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "TextComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 4,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 34
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 4,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "text": " "
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "ParamCommandComment",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "col": 6,
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 34
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "col": 5,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 34
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "col": 29,
// CHECK-NEXT:      "file": "{{.*}}",
// CHECK-NEXT:      "line": 34
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "direction": "in",
// CHECK-NEXT:    "param": "...",
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "ParagraphComment",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "col": 15,
// CHECK-NEXT:       "file": "{{.*}}",
// CHECK-NEXT:       "line": 34
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "col": 15,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "col": 29,
// CHECK-NEXT:        "file": "{{.*}}",
// CHECK-NEXT:        "line": 34
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "TextComment",
// CHECK-NEXT:        "loc": {
// CHECK-NEXT:         "col": 15,
// CHECK-NEXT:         "file": "{{.*}}",
// CHECK-NEXT:         "line": 34
// CHECK-NEXT:        },
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "col": 15,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 34
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "col": 29,
// CHECK-NEXT:          "file": "{{.*}}",
// CHECK-NEXT:          "line": 34
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "text": " More arguments"
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }
