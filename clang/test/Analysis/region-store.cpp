// DEFINE: %{analyzer} = %clang_analyze_cc1 -Wno-array-bounds %s \
// DEFINE:   -analyzer-checker=core,cplusplus,unix,debug.ExprInspection

// RUN: %{analyzer} -verify=default
// RUN: %{analyzer} -analyzer-config region-store-max-binding-fanout=10 -verify=limit10
// RUN: %{analyzer} -analyzer-config region-store-max-binding-fanout=0  -verify=unlimited

template <class T> void clang_analyzer_dump(T);
void clang_analyzer_eval(bool);

template <class... Ts> void escape(Ts...);
bool coin();

class Loc {
  int x;
};
class P1 {
public:
  Loc l;
  void setLoc(Loc L) {
    l = L;
  }
  
};
class P2 {
public:
  int m;
  int accessBase() {
    return m;
  }
};
class Derived: public P1, public P2 {
};
int radar13445834(Derived *Builder, Loc l) {
  Builder->setLoc(l);
  return Builder->accessBase();
  
}

void boundedNumberOfBindings() {
  int array[] {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22};
  clang_analyzer_dump(array[0]);  // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{0 S32b}}
  clang_analyzer_dump(array[1]);  // default-warning  {{1 S32b}}  unlimited-warning  {{1 S32b}}  limit10-warning  {{1 S32b}}
  clang_analyzer_dump(array[2]);  // default-warning  {{2 S32b}}  unlimited-warning  {{2 S32b}}  limit10-warning  {{2 S32b}}
  clang_analyzer_dump(array[3]);  // default-warning  {{3 S32b}}  unlimited-warning  {{3 S32b}}  limit10-warning  {{3 S32b}}
  clang_analyzer_dump(array[4]);  // default-warning  {{4 S32b}}  unlimited-warning  {{4 S32b}}  limit10-warning  {{4 S32b}}
  clang_analyzer_dump(array[5]);  // default-warning  {{5 S32b}}  unlimited-warning  {{5 S32b}}  limit10-warning  {{5 S32b}}
  clang_analyzer_dump(array[6]);  // default-warning  {{6 S32b}}  unlimited-warning  {{6 S32b}}  limit10-warning  {{6 S32b}}
  clang_analyzer_dump(array[7]);  // default-warning  {{7 S32b}}  unlimited-warning  {{7 S32b}}  limit10-warning  {{7 S32b}}
  clang_analyzer_dump(array[8]);  // default-warning  {{8 S32b}}  unlimited-warning  {{8 S32b}}  limit10-warning  {{8 S32b}}
  clang_analyzer_dump(array[9]);  // default-warning  {{9 S32b}}  unlimited-warning  {{9 S32b}}  limit10-warning  {{9 S32b}}
  clang_analyzer_dump(array[10]); // default-warning {{10 S32b}}  unlimited-warning {{10 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[11]); // default-warning {{11 S32b}}  unlimited-warning {{11 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[12]); // default-warning {{12 S32b}}  unlimited-warning {{12 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[13]); // default-warning {{13 S32b}}  unlimited-warning {{13 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[14]); // default-warning {{14 S32b}}  unlimited-warning {{14 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[15]); // default-warning {{15 S32b}}  unlimited-warning {{15 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[16]); // default-warning {{16 S32b}}  unlimited-warning {{16 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[17]); // default-warning {{17 S32b}}  unlimited-warning {{17 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[18]); // default-warning {{18 S32b}}  unlimited-warning {{18 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[19]); // default-warning {{19 S32b}}  unlimited-warning {{19 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[20]); // default-warning {{20 S32b}}  unlimited-warning {{20 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[21]); // default-warning {{21 S32b}}  unlimited-warning {{21 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[22]); // default-warning {{22 S32b}}  unlimited-warning {{22 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(array[23]); //          see below                     see below            limit10-warning {{Unknown}}
  // default-warning@-1   {{1st function call argument is an uninitialized value}}
  // unlimited-warning@-2 {{1st function call argument is an uninitialized value}}
  // FIXME: The last dump at index 23 should be Undefined due to out of bounds access.
}

void incompleteInitList() {
  int array[23] {0,1,2,3,4,5,6,7,8,9,10,11 /*rest are zeroes*/ };
  clang_analyzer_dump(array[0]);  // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{0 S32b}}
  clang_analyzer_dump(array[1]);  // default-warning  {{1 S32b}}  unlimited-warning  {{1 S32b}}  limit10-warning  {{1 S32b}}
  clang_analyzer_dump(array[2]);  // default-warning  {{2 S32b}}  unlimited-warning  {{2 S32b}}  limit10-warning  {{2 S32b}}
  clang_analyzer_dump(array[3]);  // default-warning  {{3 S32b}}  unlimited-warning  {{3 S32b}}  limit10-warning  {{3 S32b}}
  clang_analyzer_dump(array[4]);  // default-warning  {{4 S32b}}  unlimited-warning  {{4 S32b}}  limit10-warning  {{4 S32b}}
  clang_analyzer_dump(array[5]);  // default-warning  {{5 S32b}}  unlimited-warning  {{5 S32b}}  limit10-warning  {{5 S32b}}
  clang_analyzer_dump(array[6]);  // default-warning  {{6 S32b}}  unlimited-warning  {{6 S32b}}  limit10-warning  {{6 S32b}}
  clang_analyzer_dump(array[7]);  // default-warning  {{7 S32b}}  unlimited-warning  {{7 S32b}}  limit10-warning  {{7 S32b}}
  clang_analyzer_dump(array[8]);  // default-warning  {{8 S32b}}  unlimited-warning  {{8 S32b}}  limit10-warning  {{8 S32b}}
  clang_analyzer_dump(array[9]);  // default-warning  {{9 S32b}}  unlimited-warning  {{9 S32b}}  limit10-warning  {{9 S32b}}
  clang_analyzer_dump(array[10]); // default-warning {{10 S32b}}  unlimited-warning {{10 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[11]); // default-warning {{11 S32b}}  unlimited-warning {{11 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[12]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[13]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[14]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[15]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[16]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[17]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[18]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[19]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[20]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[21]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[22]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  clang_analyzer_dump(array[23]); // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{Unknown}}
  // FIXME: The last dump at index 23 should be Undefined due to out of bounds access.
}

struct Inner {
  int first;
  int second;
};
struct Outer {
  Inner upper;
  Inner lower;
};

void nestedStructInitLists() {
  Outer array[]{ // 7*4: 28 values
    {{00, 01}, {02, 03}},
    {{10, 11}, {12, 13}},
    {{20, 21}, {22, 23}},
    {{30, 31}, {32, 33}},
    {{40, 41}, {42, 43}},
    {{50, 51}, {52, 53}},
    {{60, 61}, {62, 63}},
  };

  int *p = (int*)array;
  clang_analyzer_dump(p[0]);  // default-warning  {{0 S32b}}  unlimited-warning  {{0 S32b}}  limit10-warning  {{0 S32b}}
  clang_analyzer_dump(p[1]);  // default-warning  {{1 S32b}}  unlimited-warning  {{1 S32b}}  limit10-warning  {{1 S32b}}
  clang_analyzer_dump(p[2]);  // default-warning  {{2 S32b}}  unlimited-warning  {{2 S32b}}  limit10-warning  {{2 S32b}}
  clang_analyzer_dump(p[3]);  // default-warning  {{3 S32b}}  unlimited-warning  {{3 S32b}}  limit10-warning  {{3 S32b}}
  clang_analyzer_dump(p[4]);  // default-warning {{10 S32b}}  unlimited-warning {{10 S32b}}  limit10-warning {{10 S32b}}
  clang_analyzer_dump(p[5]);  // default-warning {{11 S32b}}  unlimited-warning {{11 S32b}}  limit10-warning {{11 S32b}}
  clang_analyzer_dump(p[6]);  // default-warning {{12 S32b}}  unlimited-warning {{12 S32b}}  limit10-warning {{12 S32b}}
  clang_analyzer_dump(p[7]);  // default-warning {{13 S32b}}  unlimited-warning {{13 S32b}}  limit10-warning {{13 S32b}}
  clang_analyzer_dump(p[8]);  // default-warning {{20 S32b}}  unlimited-warning {{20 S32b}}  limit10-warning {{20 S32b}}
  clang_analyzer_dump(p[9]);  // default-warning {{21 S32b}}  unlimited-warning {{21 S32b}}  limit10-warning {{21 S32b}}
  clang_analyzer_dump(p[10]); // default-warning {{22 S32b}}  unlimited-warning {{22 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[11]); // default-warning {{23 S32b}}  unlimited-warning {{23 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[12]); // default-warning {{30 S32b}}  unlimited-warning {{30 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[13]); // default-warning {{31 S32b}}  unlimited-warning {{31 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[14]); // default-warning {{32 S32b}}  unlimited-warning {{32 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[15]); // default-warning {{33 S32b}}  unlimited-warning {{33 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[16]); // default-warning {{40 S32b}}  unlimited-warning {{40 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[17]); // default-warning {{41 S32b}}  unlimited-warning {{41 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[18]); // default-warning {{42 S32b}}  unlimited-warning {{42 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[19]); // default-warning {{43 S32b}}  unlimited-warning {{43 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[20]); // default-warning {{50 S32b}}  unlimited-warning {{50 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[21]); // default-warning {{51 S32b}}  unlimited-warning {{51 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[22]); // default-warning {{52 S32b}}  unlimited-warning {{52 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[23]); // default-warning {{53 S32b}}  unlimited-warning {{53 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[24]); // default-warning {{60 S32b}}  unlimited-warning {{60 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[25]); // default-warning {{61 S32b}}  unlimited-warning {{61 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[26]); // default-warning {{62 S32b}}  unlimited-warning {{62 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[27]); // default-warning {{63 S32b}}  unlimited-warning {{63 S32b}}  limit10-warning {{Unknown}}
  clang_analyzer_dump(p[28]); //          see below                     see below            limit10-warning {{Unknown}}
  // default-warning@-1   {{1st function call argument is an uninitialized value}}
  // unlimited-warning@-2 {{1st function call argument is an uninitialized value}}
  // FIXME: The last dump at index 28 should be Undefined due to out of bounds access.
}

void expectNoLeaksInWidenedInitLists() {
  int *p[] {
    new int(0),
    new int(1),
    new int(2),
    new int(3),
    new int(4),
    new int(5),
    new int(6),
    new int(7),
    new int(8),
    new int(9),
    new int(10),
    new int(11),
    new int(12),
    new int(13),
    new int(14),
    new int(15),
    new int(16),
    new int(17),
    new int(18),
    new int(19),
    new int(20),
    new int(21),
    new int(22),
    new int(23),
    new int(24),
  };
  clang_analyzer_dump(*p[0]);  // default-warning  {{0 S32b}} unlimited-warning  {{0 S32b}} limit10-warning {{0 S32b}}
  clang_analyzer_dump(*p[12]); // default-warning {{12 S32b}} unlimited-warning {{12 S32b}} limit10-warning {{Unknown}}
  escape(p); // no-leaks
}

void rawArrayWithSelfReference() {
  // If a pointer to some object escapes, that pointed object should escape too.
  // Consequently, if the 22th initializer would escape, then the p[6] should also escape - clobbering any loads from that location later.
  int *p[25] = {
    new int(0),
    new int(1),
    new int(2),
    new int(3),
    new int(4),
    new int(5),
    new int(6),
    new int(7),
    p[5], // Should be a pointer to the 6th array element, but we get Undefined as the analyzer thinks that "p" is not yet initialized, thus loading from index 5 is UB. This is wrong.
    new int(9),
    new int(10),
    new int(11),
    new int(12),
    new int(13),
    new int(14),
    new int(15),
    new int(16),
    new int(17),
    new int(18),
    new int(19),
    new int(20),
    new int(21),
    p[6], // Should be a pointer to the 6th array element, but we get Undefined as the analyzer thinks that "p" is not yet initialized, thus loading from index 5 is UB. This is wrong.
    new int(23),
    new int(24),
  };
  clang_analyzer_dump(*p[5]);  // default-warning {{5 S32b}} unlimited-warning {{5 S32b}} limit10-warning {{5 S32b}}
  clang_analyzer_dump(*p[6]);  // default-warning {{6 S32b}} unlimited-warning {{6 S32b}} limit10-warning {{6 S32b}}

  if (coin()) {
    clang_analyzer_dump(*p[8]);
    // default-warning@-1   {{Dereference of undefined pointer value}}
    // unlimited-warning@-2 {{Dereference of undefined pointer value}}
    // limit10-warning@-3   {{Dereference of undefined pointer value}}
  }

  clang_analyzer_dump(*p[12]); // default-warning {{12 S32b}} unlimited-warning {{12 S32b}} limit10-warning {{Unknown}}

  if (coin()) {
    clang_analyzer_dump(*p[22]);
    // default-warning@-1   {{Dereference of undefined pointer value}}
    // unlimited-warning@-2 {{Dereference of undefined pointer value}}
    // limit10-warning@-3   {{Unknown}}
  }

  clang_analyzer_dump(*p[23]); // default-warning {{23 S32b}} unlimited-warning {{23 S32b}} limit10-warning {{Unknown}}

  escape(p); // no-leaks
}

template <class T, unsigned Size> struct BigArray {
  T array[Size];
};

void fieldArrayWithSelfReference() {
  // Similar to "rawArrayWithSelfReference", but using an aggregate object and assignment operator to achieve the element-wise binds.
  BigArray<int *, 25> p;
  p = {
    new int(0),
    new int(1),
    new int(2),
    new int(3),
    new int(4),
    new int(5),
    new int(6),
    new int(7),
    p.array[5], // Pointer to the 6th array element.
    new int(9),
    new int(10),
    new int(11),
    new int(12),
    new int(13),
    new int(14),
    new int(15),
    new int(16),
    new int(17),
    new int(18),
    new int(19),
    new int(20),
    new int(21),
    p.array[6], // Pointer to the 7th array element.
    new int(23),
    new int(24),
  };
  clang_analyzer_dump(*p.array[5]);  // default-warning {{5 S32b}} unlimited-warning {{5 S32b}} limit10-warning {{5 S32b}}
  clang_analyzer_dump(*p.array[6]);  // default-warning {{6 S32b}} unlimited-warning {{6 S32b}} limit10-warning {{6 S32b}}

  if (coin()) {
    clang_analyzer_dump(*p.array[8]);
    // default-warning@-1   {{Unknown}}
    // unlimited-warning@-2 {{Unknown}}
    // limit10-warning@-3   {{Unknown}}
  }

  clang_analyzer_dump(*p.array[12]); // default-warning {{12 S32b}} unlimited-warning {{12 S32b}} limit10-warning {{Unknown}}

  if (coin()) {
    clang_analyzer_dump(*p.array[22]);
    // default-warning@-1   {{Unknown}}
    // unlimited-warning@-2 {{Unknown}}
    // limit10-warning@-3   {{Unknown}}
  }

  clang_analyzer_dump(*p.array[23]); // default-warning {{23 S32b}} unlimited-warning {{23 S32b}} limit10-warning {{Unknown}}

  escape(p); // no-leaks
}

struct PtrHolderBase {
  int *ptr;
};
struct BigStruct : BigArray<int, 1000>, PtrHolderBase {};
void largeBaseClasses() {
  BigStruct D{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, {new int(25)}};
  (void)D; // no-leak here, the PtrHolderBase subobject is properly escaped.

  clang_analyzer_dump(*D.ptr);  // default-warning {{25 S32b}} unlimited-warning {{25 S32b}} limit10-warning {{Unknown}}
  escape(D);
}

struct List {
  int* ptr;
  BigArray<int, 30> head;
  List *tail;
};
void tempObjectMayEscapeArgumentsInAssignment() {
  // This will be leaked after the assignment. However, we should not diagnose
  // this because in the RHS of the assignment the temporary couldn't be really
  // materialized due to the number of bindings, thus the address of `l` will
  // escape there.
  List l{new int(404)};

  // ExprWithCleanups wraps the assignment operator call, which assigns a MaterializeTemporaryExpr.
  l = List{new int(42), {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, &l};
  (void)l;
  // default-warning@-1   {{Potential memory leak}} We detect the leak with the default settings.
  // unlimited-warning@-2 {{Potential memory leak}} We detect the leak with the default settings.
  // limit10 is missing! It's because in that case we escape `&l`, thus we assume freed. This is good.

  clang_analyzer_dump(*l.ptr); // default-warning {{42 S32b}} unlimited-warning {{42 S32b}} limit10-warning {{42 S32b}}
  escape(l);
}

void tempObjNotMaterializedThusDoesntEscapeAnything() {
  List l{new int(404)};
  // We have no ExprWithCleanups or MaterializeTemporaryExpr here, so `&l` is never escaped. This is good.
  (void)List{new int(42), {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, &l};
  (void)l;
  // default-warning@-1   {{Potential memory leak}} We detect the leak with the default settings.
  // limit10-warning@-2   {{Potential memory leak}} We detect the leak with the default settings.
  // unlimited-warning@-3 {{Potential memory leak}} We detect the leak with the default settings.

  clang_analyzer_dump(*l.ptr); // default-warning {{404 S32b}} unlimited-warning {{404 S32b}} limit10-warning {{404 S32b}}
  escape(l);
}

void theValueOfTheEscapedRegionRemainsTheSame() {
  int *p = new int(404);
  List l{p};
  List l2{new int(42), {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, &l};

  // The value of l.ptr shouldn't be clobbered after a failing to copy `&l`.
  clang_analyzer_eval(p == l.ptr); // default-warning {{TRUE}} unlimited-warning {{TRUE}} limit10-warning {{TRUE}}

  // If the bindings fit, we will know that it aliases with `p`. Otherwise, it's unknown. This is good.
  clang_analyzer_dump(*l2.tail->ptr); // default-warning {{404 S32b}} unlimited-warning {{404 S32b}} limit10-warning {{Unknown}}

  escape(l, l2);
}

void calleeWithManyParms(BigArray<int, 7> arr7, BigArray<int, 100> arr100) {
  clang_analyzer_dump(arr7.array[0]);    // default-warning {{0 S32b}} unlimited-warning {{0 S32b}} limit10-warning {{0 S32b}}
  clang_analyzer_dump(arr7.array[6]);    // default-warning {{6 S32b}} unlimited-warning {{6 S32b}} limit10-warning {{6 S32b}}

  clang_analyzer_dump(arr100.array[0]);  // default-warning {{10 S32b}} unlimited-warning {{10 S32b}} limit10-warning {{10 S32b}}
  clang_analyzer_dump(arr100.array[6]);  // default-warning {{16 S32b}} unlimited-warning {{16 S32b}} limit10-warning {{16 S32b}}

  clang_analyzer_dump(arr100.array[8]);  // default-warning {{18 S32b}} unlimited-warning {{18 S32b}} limit10-warning {{18 S32b}}
  clang_analyzer_dump(arr100.array[9]);  // default-warning {{19 S32b}} unlimited-warning {{19 S32b}} limit10-warning {{19 S32b}}
  clang_analyzer_dump(arr100.array[10]); // default-warning {{20 S32b}} unlimited-warning {{20 S32b}} limit10-warning {{Unknown}}
  clang_analyzer_dump(arr100.array[99]); // default-warning {{19 S32b}} unlimited-warning {{19 S32b}} limit10-warning {{Unknown}}
}

void tooManyFnArgumentsWhenInlining() {
  calleeWithManyParms({0,1,2,3,4,5,6}, {
    10,11,12,13,14,15,16,17,18,19,
    20,21,22,23,24,25,26,27,28,29,
    30,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,45,46,47,48,49,
    50,51,52,53,54,55,56,57,58,59,
    60,61,62,63,64,65,66,67,68,69,
    70,71,72,73,74,75,76,77,78,79,
    80,81,82,83,84,85,86,87,88,89,
    90,91,92,93,94,95,96,97,98,99,
    10,11,12,13,14,15,16,17,18,19,
  });
}

void gh129211_assertion() {
  struct Clazz {
    int b;
    int : 0;
  };

  Clazz d[][5][5] = {
    {
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}}
    },
    {
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}},
    },
    {
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}, {}},
      {{}, {}, {}, {}},
    }
  }; // no-crash
}
