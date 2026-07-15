// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 -Wno-uninitialized %s
// The ERROR run adds a leading unrelated error so every later function is
// analyzed through the post-error path; the same constructor-body diagnostics
// must still fire there.
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DLEADING_ERROR %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

#ifdef LEADING_ERROR
int leading_unrelated_error = undeclared_identifier;
// expected-error@-1 {{use of undeclared identifier 'undeclared_identifier'}}
#endif

namespace std { enum class byte : unsigned char {}; }

// A [[uninit]] member that is never read needs no assignment: the constructor
// is not required to initialize it (paper §5.1/§5.3).
struct NeverReadEmpty {
  int m [[uninit]];
  NeverReadEmpty() {}
};

struct NeverReadActive {
  int m [[uninit]];
  int other = 0;
  NeverReadActive() { other = 1; }
};

struct ReadAfterAssign {
  int m [[uninit]];
  ReadAfterAssign() { m = 1; int y = m; (void)y; }
};

struct ReadBeforeAssign {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  ReadBeforeAssign() { int y = m; (void)y; m = 1; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct SelfReadOnRHS {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  SelfReadOnRHS() { m = m + 1; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct CompoundAssignReads {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  CompoundAssignReads() { m += 1; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

// A built-in increment or decrement reads the member's old (uninitialized)
// value before writing, exactly like a compound assignment.
struct PreIncReads {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  PreIncReads() { ++m; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct PostIncReads {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  PostIncReads() { m++; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct PreDecReads {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  PreDecReads() { --m; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct PostDecReads {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  PostDecReads() { m--; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct IncDecAfterAssign {
  int m [[uninit]];
  IncDecAfterAssign() { m = 0; ++m; m--; }
};

struct PostIncInInitBeforeAssign {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  PostIncInInitBeforeAssign() { int y = m++; (void)y; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct PostIncInInitAfterAssign {
  int m [[uninit]];
  PostIncInInitAfterAssign() { m = 0; int y = m++; (void)y; }
};

struct IncExplicitThis {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  IncExplicitThis() { ++this->m; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct IncDerefThis {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  IncDerefThis() { ++(*this).m; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

// Incrementing another object's member is not a read of the current object.
struct IncOtherObject {
  int m [[uninit]];
  IncOtherObject(IncOtherObject &o) { m = 0; ++o.m; }
};

template <typename T>
struct IncTmpl {
  T m [[uninit]]; // expected-note {{member 'm' declared here}}
  IncTmpl() { ++m; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};
template struct IncTmpl<int>; // expected-note {{in instantiation of member function 'IncTmpl<int>::IncTmpl' requested here}}

struct IncSuppressedByRule {
  int m [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] IncSuppressedByRule() { ++m; }
};

struct IncSuppressedStmt {
  int m [[uninit]];
  IncSuppressedStmt() {
    // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
    [[profiles::suppress(std::init)]] { ++m; }
  }
};

struct OneBranchThenRead {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  OneBranchThenRead(bool b) {
    if (b)
      m = 1;
    int y = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)y;
  }
};

struct BothBranchesThenRead {
  int m [[uninit]];
  BothBranchesThenRead(bool b) {
    if (b)
      m = 1;
    else
      m = 2;
    int y = m;
    (void)y;
  }
};

struct LoopBodyThenRead {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  LoopBodyThenRead(int n) {
    for (int i = 0; i < n; ++i)
      m = i;
    int y = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)y;
  }
};

// std::byte may be read while uninitialized (paper §4.5).
struct ByteExempt {
  std::byte b [[uninit]];
  ByteExempt() { std::byte c = b; (void)c; }
};

// A member initialized in the (written) member-initializer list is assigned at
// body entry, so a later read is fine and no marker/list-init contradiction is
// introduced.
struct MarkerWithListInit {
  int m [[uninit]];
  MarkerWithListInit() : m(0) { int y = m; (void)y; }
};

// Reads inside the written member-initializer list are checked in execution
// order (declaration order): a member becomes assigned at its own
// initializer, so an earlier member initializer -- or a base initializer,
// which runs before all member initializers -- reading it is a
// read-before-init.
struct InitListReadBefore {
  int o;
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  InitListReadBefore() : o(m), m(5) {} // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct InitListReadAfter {
  int m [[uninit]];
  int o;
  InitListReadAfter() : m(5), o(m) {} // OK: m's initializer runs first
};

struct InitListReadNoInit {
  int o;
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  InitListReadNoInit() : o(m) {} // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct InitListSelfRead {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  InitListSelfRead() : m(m + 1) {} // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct InitBase {
  InitBase(int);
};
struct InitListBaseRead : InitBase {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  InitListBaseRead() : InitBase(m) {} // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] InitListReadSuppressed {
  int o;
  int m [[uninit]];
  InitListReadSuppressed() : o(m) {} // OK: suppressed
};

// A this-capturing lambda created in the ctor body may run immediately, so a
// member read in its body counts as a read at the point the lambda is
// created. Writes in the body earn no assignment credit (the lambda may never
// run), and a lambda stored now but called only after assignment is flagged
// all the same (accepted imprecision).
struct LambdaReadInvoked {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  LambdaReadInvoked() {
    int y = [this] { return m; }(); // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)y;
  }
};

struct LambdaReadAfterAssign {
  int m [[uninit]];
  LambdaReadAfterAssign() {
    m = 1;
    auto l = [this] { return m; }; // OK: m assigned on every path here
    (void)l;
  }
};

struct LambdaWriteNoCredit {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  LambdaWriteNoCredit() {
    auto l = [this] { m = 1; }; // OK: a body write is not a read
    l();
    int y = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)y;
  }
};

struct LambdaNoMemberUse {
  int m [[uninit]];
  LambdaNoMemberUse() {
    auto l = [this] { return 42; }; // OK: body touches no member
    m = l();
  }
};

struct LambdaCompoundRead {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  LambdaCompoundRead() {
    auto l = [this] { m += 1; }; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)l;
  }
};

struct LambdaNestedRead {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  LambdaNestedRead() {
    auto l = [this] { return [this] { return m; }; }; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)l;
  }
};

struct LambdaImplicitThisCapture {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  LambdaImplicitThisCapture() {
    auto l = [&] { return m; }; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)l;
  }
};

// An init-capture's initializer is an ordinary CFG element of the ctor and is
// checked by the plain read arm, independent of the body scan.
struct LambdaCaptureInitRead {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  LambdaCaptureInitRead() {
    auto l = [v = m] { return v; }; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)l;
  }
};

struct LambdaReadSuppressed {
  int m [[uninit]];
  LambdaReadSuppressed() {
    // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
    [[profiles::suppress(std::init)]] {
      auto l = [this] { return m; }; // OK: suppressed
      (void)l;
    }
  }
};

// Strict crediting for plain escapes: inside a constructor body, nothing but
// a whole-member assignment counts as initializing an [[uninit]] member.
// Passing &m to a [[ref_to_uninit]] parameter of an ordinary function,
// calling a member function that assigns it, or letting `this` escape earns
// no credit -- the paper rejects complex constructor code (§5.1) and
// reserves callee-initialization for now_init (§6.2, the [[now_init]] tests
// below); the remedy for a plain callee is [[profiles::suppress]] or the
// [[now_init]] annotation. Contrast the *local*-object passes, which
// conservatively credit any escape
// (safety-profile-init-local-member-read.cpp, test_escape_*).
void init_pointee(int *p [[ref_to_uninit]]);
struct EscapeMemberAddress {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  EscapeMemberAddress() {
    init_pointee(&m);
    int x = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)x;
  }
};

struct EscapeMemberCall {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  void setup() { m = 1; }
  EscapeMemberCall() {
    setup();
    int x = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)x;
  }
};

struct EscapeThis;
void take_this(EscapeThis *);
struct EscapeThis {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  EscapeThis() {
    take_this(this);
    int x = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)x;
  }
};

// The §6.2 exception: a call to a [[now_init]] function initializes the
// storage bound to each of its [[ref_to_uninit]] parameters, so a
// current-object member passed as `&m` (or `m` to a reference parameter)
// becomes assigned at the call. The credit is a real Gen bit in the
// dataflow -- not parse-order -- so §1.2's all-branches rule still governs:
// a [[now_init]] call under one branch does not satisfy a read at the join.
[[now_init]] void now_init_pointee(int *p [[ref_to_uninit]]);
[[now_init]] void now_init_referent(int &r [[ref_to_uninit]]);
struct NowInitMemberAddress {
  int m [[uninit]];
  NowInitMemberAddress() {
    now_init_pointee(&m);
    int x = m; // OK: the [[now_init]] callee initialized m
    (void)x;
  }
};

struct NowInitMemberReference {
  int m [[uninit]];
  NowInitMemberReference() {
    now_init_referent(m);
    int x = m; // OK
    (void)x;
  }
};

struct NowInitReadBeforeCall {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  NowInitReadBeforeCall() {
    int x = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    now_init_pointee(&m);
    (void)x;
  }
};

struct NowInitUnderBranch {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  NowInitUnderBranch(bool b) {
    if (b)
      now_init_pointee(&m);
    int x = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)x;
  }
};

struct NowInitBothBranches {
  int m [[uninit]];
  NowInitBothBranches(bool b) {
    if (b)
      now_init_pointee(&m);
    else
      m = 0;
    int x = m; // OK: initialized on every path
    (void)x;
  }
};

// Only the *marked* parameters carry the promise: an unmarked parameter of
// the same [[now_init]] callee earns nothing -- and handing it &m is already
// the parse-time binding violation.
[[now_init]] void now_init_first(int *p [[ref_to_uninit]], int *q);
struct NowInitUnmarkedParam {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  int o [[uninit]];
  NowInitUnmarkedParam() {
    now_init_first(&o, &m); // expected-error {{pointer to uninitialized memory must be marked '[[ref_to_uninit]]' under profile 'std::init'}}
    int x = o; // OK: bound to the marked parameter
    int y = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)x; (void)y;
  }
};

// Passing `this` itself to a marked parameter hands the callee the whole
// object to initialize: every tracked member is assigned at the call. (The
// implicit object parameter of a member call cannot carry the marker, so a
// member call still earns nothing -- see EscapeMemberCall.)
struct NowInitWholeObject;
[[now_init]] void now_init_object(NowInitWholeObject *obj [[ref_to_uninit]]);
struct NowInitWholeObject {
  int a [[uninit]];
  int b [[uninit]];
  NowInitWholeObject() {
    now_init_object(this);
    int x = a + b; // OK: the whole object was handed over for initialization
    (void)x;
  }
};

// Argument mapping through operator calls: a member operator() receives the
// object as operator-call argument 0 ahead of its declared parameters --
// for a C++23 static operator() too, whose object argument is still
// evaluated -- while an explicit-object operator() declares the object as
// parameter 0 and maps arguments directly.
struct NowInitStaticFunctor {
  [[now_init]] static void operator()(int *p [[ref_to_uninit]]);
};
struct NowInitStaticOperator {
  int m [[uninit]];
  NowInitStaticOperator() {
    NowInitStaticFunctor f;
    f(&m);
    int x = m; // OK: &m bound the static operator()'s marked parameter
    (void)x;
  }
};

struct NowInitExplicitObjectFunctor {
  [[now_init]] void operator()(this NowInitExplicitObjectFunctor &,
                               int *p [[ref_to_uninit]]);
};
struct NowInitExplicitObjectOperator {
  int m [[uninit]];
  NowInitExplicitObjectOperator() {
    NowInitExplicitObjectFunctor f;
    f(&m);
    int x = m; // OK: &m bound the explicit-object operator()'s parameter 1
    (void)x;
  }
};

// A [[now_init]] call inside a written member initializer credits in
// execution order: the call element precedes the initializer's own write, so
// a body read of the passed member is already covered.
[[now_init]] int now_init_count(int *p [[ref_to_uninit]]);
struct NowInitInMemInit {
  int m [[uninit]];
  int n;
  NowInitInMemInit() : n(now_init_count(&m)) {
    int x = m; // OK: credited at the call inside n's initializer
    (void)x;
  }
};

// [[uninit]] members inherited from a non-virtual base with no user-provided
// constructor are tracked like the class's own: nothing can have assigned
// them before the derived body runs. A base with a user-provided constructor
// is trusted (paper §5.1) and its members left alone.
struct BaseRead {
  int bm [[uninit]]; // expected-note {{member 'bm' declared here}}
};
struct DerivedRead : BaseRead {
  DerivedRead() { int y = bm; (void)y; } // expected-error {{member 'bm' is read before initialization under profile 'std::init'}}
};

struct BaseAssign {
  int bm [[uninit]];
};
struct DerivedAssign : BaseAssign {
  DerivedAssign() { bm = 1; int y = bm; (void)y; } // OK
};

struct BaseWritten {
  int bm [[uninit]];
};
struct DerivedWrittenBaseInit : BaseWritten {
  DerivedWrittenBaseInit() : BaseWritten{1} { int y = bm; (void)y; } // OK: written base initializer
};

struct BaseTrusted {
  int bm [[uninit]];
  BaseTrusted() {}
};
struct DerivedTrustedBase : BaseTrusted {
  DerivedTrustedBase() { int y = bm; (void)y; } // OK: base ctor trusted (paper §5.1)
};

struct GrandBase {
  int gm [[uninit]]; // expected-note {{member 'gm' declared here}}
};
struct MidBase : GrandBase {};
struct DerivedTwoLevels : MidBase {
  DerivedTwoLevels() { int y = gm; (void)y; } // expected-error {{member 'gm' is read before initialization under profile 'std::init'}}
};

struct BaseSup {
  int bm [[uninit]];
};
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] DerivedSup : BaseSup {
  DerivedSup() { int y = bm; (void)y; } // OK: suppressed
};

// A delegating constructor's target initializes the members before the body
// runs (paper §5.1), so its body is not analyzed.
struct Delegating {
  int m [[uninit]];
  Delegating() : Delegating(0) { int y = m; (void)y; }
  Delegating(int v) : m(v) {}
};

// A read of another object's member is not a read of the current object.
struct ReadsOtherObject {
  int m [[uninit]];
  ReadsOtherObject() { m = 0; }
  ReadsOtherObject(const ReadsOtherObject &o) { m = o.m; }
};

struct MultipleMembers {
  int a [[uninit]];
  int b [[uninit]]; // expected-note {{member 'b' declared here}}
  int c [[uninit]];
  MultipleMembers() {
    a = 1;
    int x = a; (void)x;
    int y = b; (void)y; // expected-error {{member 'b' is read before initialization under profile 'std::init'}}
    c = 2;
    int z = c; (void)z;
  }
};

struct ExplicitThis {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  ExplicitThis() { int y = this->m; (void)y; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

// Writing through `(*this).m` is an initialization, so a later read of m is
// fine -- the same as `this->m` (the reported false positive).
struct DerefThisWriteThenRead {
  int m [[uninit]];
  DerefThisWriteThenRead() { (*this).m = 1; int y = m; (void)y; }
};

// A real read through `(*this).m` before assignment is still diagnosed.
struct DerefThisReadBeforeInit {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  DerefThisReadBeforeInit() { int y = (*this).m; (void)y; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct OutOfLine {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  OutOfLine();
};
OutOfLine::OutOfLine() { int y = m; (void)y; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}

template <typename T>
struct Tmpl {
  T m [[uninit]]; // expected-note {{member 'm' declared here}}
  Tmpl() { T y = m; (void)y; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};
template struct Tmpl<int>; // expected-note {{in instantiation of member function 'Tmpl<int>::Tmpl' requested here}}

struct SuppressedCtor {
  int m [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] SuppressedCtor() { int y = m; (void)y; }
};

struct SuppressedByRule {
  int m [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] SuppressedByRule() { int y = m; (void)y; }
};

struct SuppressedStmt {
  int m [[uninit]];
  SuppressedStmt() {
    // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
    [[profiles::suppress(std::init)]] { int y = m; (void)y; }
  }
};

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] SuppressedClass {
  int m [[uninit]];
  SuppressedClass() { int y = m; (void)y; }
};
