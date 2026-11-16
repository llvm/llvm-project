// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -verify %s

template <typename T> void clang_analyzer_dump(T);
template <typename T> void clang_analyzer_value(T);
void clang_analyzer_warnIfReached();

struct Node { int* ptr; };

void copy_on_stack(Node* n1) {
  Node tmp = *n1;
  Node* n2 = &tmp;
  clang_analyzer_dump(n1); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<Node * n1>}}}
  clang_analyzer_dump(n2); // expected-warning {{&tmp}}

  clang_analyzer_dump(n1->ptr); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<int * Element{SymRegion{reg_${{[0-9]+}}<Node * n1>},0 S64b,struct Node}.ptr>}}}
  clang_analyzer_dump(n2->ptr); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<int * Element{SymRegion{reg_${{[0-9]+}}<Node * n1>},0 S64b,struct Node}.ptr>}}}

  if (n1->ptr != n2->ptr)
    clang_analyzer_warnIfReached(); // unreachable
  (void)(n1->ptr);
  (void)(n2->ptr);
}

void copy_on_heap(Node* n1) {
  Node* n2 = new Node(*n1);

  clang_analyzer_dump(n1); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<Node * n1>}}}
  clang_analyzer_dump(n2); // expected-warning-re {{&HeapSymRegion{conj_${{[0-9]+}}{Node *, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}}}}}

  clang_analyzer_dump(n1->ptr); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<int * Element{SymRegion{reg_${{[0-9]+}}<Node * n1>},0 S64b,struct Node}.ptr>}}}
  clang_analyzer_dump(n2->ptr); // expected-warning-re {{&SymRegion{reg_${{[0-9]+}}<int * Element{SymRegion{reg_${{[0-9]+}}<Node * n1>},0 S64b,struct Node}.ptr>}}}

  if (n1->ptr != n2->ptr)
    clang_analyzer_warnIfReached(); // unreachable
  (void)(n1->ptr);
  (void)(n2->ptr);
}

struct List {
  List* next;
  int value;
  int padding;
};

void deadCode(List orig) {
  List c = orig;
  clang_analyzer_dump(c.value);
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<int orig.value>}}
  if (c.value == 42)
    return;
  clang_analyzer_value(c.value);
  // expected-warning@-1 {{32s:{ [-2147483648, 41], [43, 2147483647] }}}
  // Before symbol was garbage collected too early, and we lost the constraints.
  if (c.value != 42)
    return;

  clang_analyzer_warnIfReached(); // no-warning: Dead code.
};

void ptr1(List* n) {
  List* n2 = new List(*n); // ctor
  if (!n->next) {
    if (n2->next) {
      clang_analyzer_warnIfReached(); // unreachable
    }
  }
  delete n2;
}

void ptr2(List* n) {
  List* n2 = new List(); // ctor
  *n2 = *n; // assignment
  if (!n->next) {
    if (n2->next) {
      clang_analyzer_warnIfReached(); // unreachable
    }
  }
  delete n2;
}

struct Wrapper {
  List head;
  int count;
};

void nestedLazyCompoundVal(List* n) {
  Wrapper* w = 0;
  {
     Wrapper lw;
     lw.head = *n;
     w = new Wrapper(lw);
  }
  if (!n->next) {
    if (w->head.next) {
      // Unreachable, w->head is a copy of *n, therefore
      // w->head.next and n->next are equal
      clang_analyzer_warnIfReached(); // no-warning: unreachable
    }
  }
  delete w;
}

// In some cases SymbolRegions were wrapped by ElementRegions in
// some they weren't. That is why the analyzer did not recognize
// if two regions are referring to the same entity which could
// lead to unreachable paths discovered and also false reports.
// D132142 resolved this issue while D132143 made
// LazyCompoundVals to only appear as default bindings in the
// RegionStore. This applies to both GH60694 and GH26595.
struct GH60694_Node { GH60694_Node *parent; };

void GH60694(GH60694_Node *a, GH60694_Node *b) {
  if (a->parent)
    return;
  *a = *b;
  a->parent->parent = 0;
}

void free(void *);

struct GH26595_List { GH26595_List *next; };

GH26595_List *GH26595(GH26595_List *item) {
  GH26595_List *next = item->next;
  if (next) {
    *item = *next;
    free(next);
    return item->next; // no-warning
  }

  return item;
}
