// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection\
// RUN:                    -Wno-dangling -Wno-c++1z-extensions\
// RUN:                    -verify=expected,cpp14\
// RUN:                    -x c++ -std=c++14 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection\
// RUN:                    -analyzer-config elide-constructors=false\
// RUN:                    -Wno-dangling -Wno-c++1z-extensions\
// RUN:                    -verify=expected,cpp14\
// RUN:                    -x c++ -std=c++14 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core\
// RUN:                    -analyzer-checker=debug.ExprInspection\
// RUN:                    -Wno-dangling -verify=expected,cpp17\
// RUN:                    -x c++ -std=c++17 %s

template<typename T>
void clang_analyzer_dump(T&&) {}

template<typename T>
T create() { return T{}; }

template<typename T>
T const& select(bool cond, T const& t, T const& u) { return cond ? t : u; }

struct Composite {
  int x;
  int y;
};

struct Derived : Composite {
  int z;
};

template<typename T>
struct Array {
  T array[20];

  T&& front() && { return static_cast<T&&>(array[0]); }
};

void whole_object() {
  int const& i = 10; // extends `int`
  clang_analyzer_dump(i); // expected-warning-re {{&lifetime_extended_object{int, i, S{{[0-9]+}}} }}
  Composite&& c = Composite{}; // extends `Composite`
  clang_analyzer_dump(c); // expected-warning-re {{&lifetime_extended_object{Composite, c, S{{[0-9]+}}} }}
  auto&& a = Array<int>{}; // extends `Array<int>`
  clang_analyzer_dump(a); // expected-warning-re {{&lifetime_extended_object{Array<int>, a, S{{[0-9]+}}} }}
  Composite&& d = Derived{}; // extends `Derived`
  clang_analyzer_dump(d); // expected-warning-re {{&Base{lifetime_extended_object{Derived, d, S{{[0-9]+}}},Composite} }}
}

void member_access() {
  int&& x = Composite{}.x;  // extends `Composite`
  clang_analyzer_dump(x); // expected-warning-re {{&lifetime_extended_object{Composite, x, S{{[0-9]+}}}.x }}
  int&& y = create<Composite>().y; // extends `Composite`
  clang_analyzer_dump(y); // expected-warning-re {{&lifetime_extended_object{struct Composite, y, S{{[0-9]+}}}.y }}
  int&& d = Array<int>{}.front(); // dangles `Array<int>`
  clang_analyzer_dump(d); // expected-warning-re {{&Element{temp_object{Array<int>, S{{[0-9]+}}}.array,0 S64b,int} }}
}

void array_subscript() {
  int&& i = Array<int>{}.array[0]; // extends `Array<int>`
  clang_analyzer_dump(i); // expected-warning-re {{&Element{lifetime_extended_object{Array<int>, i, S{{[0-9]+}}}.array,0 S64b,int} }}
  auto&& c = Array<Composite>{}.array[0]; // extends `Array<int>`
  clang_analyzer_dump(c); // expected-warning-re {{&Element{lifetime_extended_object{Array<Composite>, c, S{{[0-9]+}}}.array,0 S64b,struct Composite} }}
  auto&& x = Array<Composite>{}.array[0].x; // extends `Array<Composite>`
  clang_analyzer_dump(x); // expected-warning-re {{&Element{lifetime_extended_object{Array<Composite>, x, S{{[0-9]+}}}.array,0 S64b,struct Composite}.x }}
}

void ternary(bool cond) {
  Composite cc;
  // Value category mismatch of the operands (lvalue and xvalue), ternary produces prvalue
  auto&& ternaryProducesPRvalue = cond ? Composite{}.x : cc.x; // extends prvalue of 'int', `Composite` in true branch is destroyed
  clang_analyzer_dump(ternaryProducesPRvalue); // expected-warning-re {{&lifetime_extended_object{int, ternaryProducesPRvalue, S{{[0-9]+}}} }}

  // Value category agrees (xvalues), lifetime extension is triggered
  auto&& branchesExtended = cond ? Composite{}.x : static_cast<Composite&&>(cc).x; // extends `Composite` in true branch
  clang_analyzer_dump(branchesExtended);
  // expected-warning-re@-1 {{&lifetime_extended_object{Composite, branchesExtended, S{{[0-9]+}}}.x }}
  // expected-warning@-2 {{&cc.x }}

  // Object of different types in branches are lifetime extended
  auto&& extendingDifferentTypes = cond ? Composite{}.x : Array<int>{}.array[0]; // extends `Composite` or `Array<int>`
  clang_analyzer_dump(extendingDifferentTypes);
  // expected-warning-re@-1 {{&lifetime_extended_object{Composite, extendingDifferentTypes, S{{[0-9]+}}}.x }}
  // expected-warning-re@-2 {{&Element{lifetime_extended_object{Array<int>, extendingDifferentTypes, S{{[0-9]+}}}.array,0 S64b,int} }}

  Composite const& variableAndExtended = cond ? static_cast<Composite&&>(cc) : Array<Composite>{}.array[0]; // extends `Array<Composite>` in false branch
  clang_analyzer_dump(variableAndExtended);
  // expected-warning@-1 {{&cc }}
  // expected-warning-re@-2 {{&Element{lifetime_extended_object{Array<Composite>, variableAndExtended, S{{[0-9]+}}}.array,0 S64b,struct Composite} }}

  int const& extendAndDangling = cond ? Array<int>{}.array[0] : Array<int>{}.front(); // extends `Array<int>` only in true branch, false branch dangles
  clang_analyzer_dump(extendAndDangling);
  // expected-warning-re@-1 {{&Element{lifetime_extended_object{Array<int>, extendAndDangling, S{{[0-9]+}}}.array,0 S64b,int} }}
  // expected-warning-re@-2 {{&Element{temp_object{Array<int>, S{{[0-9]+}}}.array,0 S64b,int} }}
}

struct RefAggregate {
  int const& rx;
  Composite&& ry = Composite{};
};

void aggregateWithReferences() {
  RefAggregate multipleExtensions = {10, Composite{}}; // extends `int` and `Composite`
  clang_analyzer_dump(multipleExtensions.rx); // expected-warning-re {{&lifetime_extended_object{int, multipleExtensions, S{{[0-9]+}}} }}
  clang_analyzer_dump(multipleExtensions.ry); // expected-warning-re {{&lifetime_extended_object{Composite, multipleExtensions, S{{[0-9]+}}} }}

  RefAggregate danglingAndExtended{Array<int>{}.front(), Composite{}}; // extends only `Composite`, `Array<int>` dangles
  clang_analyzer_dump(danglingAndExtended.rx); // expected-warning-re {{&Element{temp_object{Array<int>, S{{[0-9]+}}}.array,0 S64b,int} }}
  clang_analyzer_dump(danglingAndExtended.ry); // expected-warning-re {{&lifetime_extended_object{Composite, danglingAndExtended, S{{[0-9]+}}} }}

  int i = 10;
  RefAggregate varAndExtended{i, Composite{}};  // extends `Composite`
  clang_analyzer_dump(varAndExtended.rx); // expected-warning {{&i }}
  clang_analyzer_dump(varAndExtended.ry); // expected-warning-re {{&lifetime_extended_object{Composite, varAndExtended, S{{[0-9]+}}} }}

  auto const& viaReference = RefAggregate{10, Composite{}}; // extends `int`, `Composite`, and `RefAggregate`
  clang_analyzer_dump(viaReference);    // expected-warning-re {{&lifetime_extended_object{RefAggregate, viaReference, S{{[0-9]+}}} }}
  clang_analyzer_dump(viaReference.rx); // expected-warning-re {{&lifetime_extended_object{int, viaReference, S{{[0-9]+}}} }}
  clang_analyzer_dump(viaReference.ry); // expected-warning-re {{&lifetime_extended_object{Composite, viaReference, S{{[0-9]+}}} }}
  
  // FIXME: clang currently support extending lifetime of object bound to reference members of aggregates,
  // that are created from default member initializer. But CFG and ExprEngine need to be updated to address this change.
  // The following expect warning: {{&lifetime_extended_object{Composite, defaultInitExtended, S{{[0-9]+}}} }}
  RefAggregate defaultInitExtended{i};
  clang_analyzer_dump(defaultInitExtended.ry); // expected-warning {{Unknown }}
}

void lambda() {
  auto const& lambdaRef = [capture = create<Composite>()] {};
  clang_analyzer_dump(lambdaRef); // expected-warning-re {{lifetime_extended_object{class (lambda at {{[^)]+}}), lambdaRef, S{{[0-9]+}}} }}

  // The capture [&refCapture = create<Composite const>()] { ... } per [expr.prim.lambda.capture] p6 equivalent to:
  //   auto& refCapture = create<Composite const>(); // Well-formed, deduces auto = Composite const, and performs lifetime extension
  //   [&refCapture] { ... }
  // Where 'refCapture' has the same lifetime as the lambda itself.
  // However, compilers differ: Clang lifetime-extends from C++17, GCC rejects the code, and MSVC dangles
  // See also CWG2737 (https://cplusplus.github.io/CWG/issues/2737.html)
  auto const refExtendingCapture = [&refCapture = create<Composite const>()] {
     clang_analyzer_dump(refCapture);
     // cpp14-warning-re@-1 {{&temp_object{const struct Composite, S{{[0-9]+}}} }}
     // cpp17-warning-re@-2 {{&lifetime_extended_object{const struct Composite, refExtendingCapture, S{{[0-9]+}}} }}
  };
  refExtendingCapture();
}

void viaStructuredBinding() {
  auto&& [x, y] = Composite{}; // extends `Composite` and binds it to unnamed decomposed object
  clang_analyzer_dump(x); // expected-warning-re {{&lifetime_extended_object{Composite, D{{[0-9]+}}, S{{[0-9]+}}}.x }}
  clang_analyzer_dump(y); // expected-warning-re {{&lifetime_extended_object{Composite, D{{[0-9]+}}, S{{[0-9]+}}}.y }}

  auto&& [rx, ry] = RefAggregate{10, Composite{}}; // extends `int`, `Composite`, and `RefAggregate`, and binds them to unnamed decomposed object
  clang_analyzer_dump(rx); // expected-warning-re {{&lifetime_extended_object{int, D{{[0-9]+}}, S{{[0-9]+}}} }}
  clang_analyzer_dump(ry); // expected-warning-re {{&lifetime_extended_object{Composite, D{{[0-9]+}}, S{{[0-9]+}}} }}
}

void propagation(bool cond) {
  int const& le = Composite{}.x;
  // May return lifetime-extended region or dangling temporary
  auto&& oneDangling = select(cond, le, 10); // does not extend lifetime of arguments
  clang_analyzer_dump(oneDangling);
  // expected-warning-re@-1 {{&lifetime_extended_object{Composite, le, S{{[0-9]+}}}.x }}
  // expected-warning-re@-2 {{&temp_object{int, S{{[0-9]+}}} }}

  // Always dangles
  auto&& bothDangling = select(cond, 10, 20); // does not extend lifetime of arguments
  clang_analyzer_dump(bothDangling);
  // expected-warning-re@-1 {{&temp_object{int, S{{[0-9]+}}} }}
  // expected-warning-re@-2 {{&temp_object{int, S{{[0-9]+}}} }}
}
