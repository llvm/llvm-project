// REQUIRES: host-supports-jit
//
// Usage as a lit test (pipes every line below into clang-repl as a separate
// incremental input, the same way fail.cpp / code-undo.cpp do):
//   RUN: cat %s | clang-repl | FileCheck %s
//
// Usage interactively: paste blocks (or the whole file) directly at the
// `clang-repl>` prompt. Each block is a single line on purpose -- clang-repl
// treats one physical line as one Interpreter::Parse() call, and several of
// these tests rely on packing a well-formed declaration and an ill-formed
// one onto the *same* line so both go through Sema together before the
// chunk is discarded as a whole.
//
// -----------------------------------------------------------------------
// Observed results (assertions-enabled build, this checkout, macOS/arm64)
// -----------------------------------------------------------------------
// Running this file as-is through `./bin/clang-repl < ptu-rewind-stress.cpp`
// does NOT make it to the %quit at the bottom:
//  * Test 1 (dangling_name) printed "dangling_name = 7" instead of 99, and
//    the JIT reported
//      error: In incr_module_89, duplicate definition of symbol '_dangling_name'
//    i.e. the rolled-back chunk's *codegen* (which runs incrementally per
//    top-level decl, before the end-of-chunk error check) was never undone
//    either, so the second, successful declaration silently reused the
//    first (supposedly-discarded) global.
//  * Test 3 (Boom<int> template instantiation) reliably aborts the process:
//      Assertion failed: (CP.UndefinedButUsedSize == S.UndefinedButUsed.size()),
//      function restore, file SemaStateStash.cpp, line 457.
//    i.e. real, confirmed, unconditional test-3-and-later block; a debug/
//    assertions build never reaches tests 4-7 in one session because of
//    this abort.
//  * Running tests 4-7 in isolation (skipping 1-3), Test 4 (sizeof on a
//    pre-existing type) produced
//      JIT session error: Symbols not found: [ _sz_known2 ]
//      error: Failed to materialize symbols: ...
//    followed by the process spinning at 100% CPU instead of returning to
//    the prompt for the remaining input (tests 5-7 never ran in that
//    session either).
//  * Running Test 1 completely on its own (no preceding chunks) did NOT
//    reproduce the wrong-value/duplicate-symbol failure -- it printed the
//    correct "dangling_name = 99". This is expected for a poisoned/reused-
//    memory bug: whether a stale pointer's target has been overwritten by
//    something that "looks wrong" depends on what the allocator handed out
//    to *other* code in between, so the manifestation is sensitive to the
//    exact preceding session history, not just to the isolated snippet.
//    That is precisely why this file chains many small scenarios in one
//    session rather than shipping them as independent one-liners.
//
// -----------------------------------------------------------------------
// Background
// -----------------------------------------------------------------------
// Interpreter::Parse (clang/lib/Interpreter/Interpreter.cpp) wraps every
// incremental input in a PTUSlabRollback guard:
//   1. Takes CheckPoint = Ctx.getAllocator().checkPoint()
//   2. Stashes ASTContext/Sema side-table sizes (ASTContextStateStash,
//      SemaStateStash)
//   3. Runs IncrementalParser::Parse()
//   4. On success: commits (keeps everything)
//   5. On failure: SemaState.restore(), ASTCtxState.restore(), then
//      Ctx.getAllocator().restoreToCheckPoint(CheckPoint)
//
// restoreToCheckPoint (llvm/include/llvm/Support/Allocator.h) does not just
// move a pointer back -- it actively memset()s the reclaimed range to 0xCD
// ("poisonMemory") outside of ASan builds. So *any* pointer left behind in a
// side-table that the two StateStash::restore() functions fail to clean up
// is not merely suspect, it is guaranteed to point at either poisoned bytes
// or, once new allocations reuse that space, at a completely unrelated
// object (type confusion).
//
// Reading ASTContextStateStash.cpp and SemaStateStash.cpp shows the
// restore() paths fall into three buckets:
//   (a) real cleanup: erase from the container based on
//       isAfterCheckpoint(ptr, SlabCP)  [handles most Type folding sets]
//   (b) resize()/pop-back style cleanup on trailing-append vectors
//       [FunctionScopes, LateParsedInstantiations, SavedVTableUses, ...]
//   (c) `assert(oldSize == newSize)` with NO actual erase -- compiled away
//       entirely under NDEBUG, so the container is left holding dangling
//       pointers with no diagnostic at all in a release build
//       [StringLiteralCache, KeyFunctions-adjacent maps when the early
//       return below fires, MergedDecls, TemplateInstCallbacks,
//       PendingInstantiations, LateParsedTemplateMap, VTableUses/
//       VTablesUsed, ...]
// Additionally, ASTContextStateStash::restore() opens with:
//       if (CP.TypesSize == Ctx.Types.size()) return;
// which skips *all* of the above (including the folding sets that do have
// real cleanup logic) whenever the number of interned Type nodes happens to
// be unchanged -- even though many other caches (record layouts, key
// functions, ...) can grow without interning a new Type.
//
// Every test below is built to land in one of these gaps. None of the
// "poison" code paths depend on undefined behavior sanitizers to observe --
// under a debug (assertions-enabled) build several of them should abort on
// the assert() itself; under a release build the same inputs should
// eventually crash, print garbage, or misbehave once the poisoned/reused
// memory is dereferenced.
// -----------------------------------------------------------------------

extern "C" int printf(const char *, ...);

// =======================================================================
// Test 1: Sema::IdResolver keeps a dangling entry after a failed parse.
//
// SemaStateStash::restore() (clang/lib/Interpreter/SemaStateStash.cpp)
// walks S.getCurScope()->decls() and calls Scope::RemoveDecl() for every
// decl allocated after the checkpoint -- but the matching
// `S.IdResolver.RemoveDecl(D)` call right above it is commented out. Name
// lookup goes through IdResolver, not just Scope, so `dangling_name`'s
// VarDecl stays "found" by lookup even though its storage is about to be
// poisoned by the allocator rewind.
//
// `int dangling_name = 7;` fully binds (Scope + IdResolver + DeclContext
// lookup map) before `intentional_error_type` is even parsed, because
// IncrementalParser::ParseOrWrapTopLevelDecl only checks
// Diags.hasErrorOccurred() *after* parsing every top-level decl in the
// chunk. So both statements are discarded together, but only Scope forgets
// about `dangling_name`.
// =======================================================================
int dangling_name = 7; intentional_error_type garbage_after_dangling_name;

// Redeclare the same identifier in a fresh, successful chunk. If the stale
// IdResolver entry survived, Sema's redeclaration-merging logic
// (Sema::MergeVarDecl* et al.) may try to compare this new VarDecl against
// the dangling one -- reading 0xCD-poisoned memory, or memory since reused
// by an unrelated allocation.
int dangling_name = 99;
auto t1 = printf("dangling_name = %d\n", dangling_name);
// CHECK: dangling_name = 99

// =======================================================================
// Test 2: Amplify test 1 by repeating the fail/redeclare cycle several
// times on the same identifier. Because every failed chunk takes its
// checkpoint at (almost) the same allocator offset, each failed attempt's
// storage for `stress_var` gets reallocated over the *same* address range
// poisoned by the previous attempt. Any stale IdResolver node left behind
// by an earlier iteration therefore ends up aliasing whatever the *next*
// iteration (or the final, successful declaration) allocates there --
// classic type-confusion setup, and a good candidate to run under ASan.
// =======================================================================
int stress_var = 0; intentional_error_type e0;
int stress_var = 1; intentional_error_type e1;
int stress_var = 2; intentional_error_type e2;
int stress_var = 3; intentional_error_type e3;
int stress_var = 4; intentional_error_type e4;
int stress_var = 42;
auto t2 = printf("stress_var = %d\n", stress_var);
// CHECK: stress_var = 42

// =======================================================================
// Test 3: Sema::PendingInstantiations does not survive a failed parse.
//
// ParseOrWrapTopLevelDecl() returns *before* calling
// LocalInstantiations.perform()/GlobalInstantiations.perform() whenever
// Diags.hasErrorOccurred() -- see the early `return` right after
// CleanUpPTU() in IncrementalParser.cpp. So any instantiation work queued
// while parsing `Boom<int> boom_instance; boom_instance.trigger();` is never
// drained on this path. SemaStateStash::restore() only compares
// S.PendingInstantiations.size() with an assert (it's a std::deque, never
// resized/erased), so a stale PendingImplicitInstantiation entry pointing
// at the (about-to-be-poisoned) `Boom<int>::trigger` specialization can
// leak into whatever the *next* successful chunk's own eager-instantiation
// pass processes.
// =======================================================================
template <typename T> struct Boom { void trigger() { T v; (void)v; } };
Boom<int> boom_instance; boom_instance.trigger(); intentional_error_type boom_garbage;

// This chunk's own GlobalEagerInstantiationScope/LocalEagerInstantiationScope
// pass (constructed fresh in every ParseOrWrapTopLevelDecl call) is where a
// leftover PendingInstantiations entry from the failed chunk above would
// get processed against poisoned memory.
int after_boom_marker = 1;
auto t3 = printf("after_boom_marker = %d\n", after_boom_marker);
// CHECK: after_boom_marker = 1

// =======================================================================
// Test 4: ASTContextStateStash's Types.size() early-return guard skips
// cleanup of caches that don't depend on interning a new Type.
//
//   void ASTContextStateStash::restore(...) {
//     if (CP.TypesSize == Ctx.Types.size())
//       return;
//     ... (all the real per-folding-set cleanup lives below this line) ...
//
// `AlreadyKnown` already exists (its RecordType was interned when it was
// first declared, in the prior committed chunk). Computing sizeof() on it
// only populates Ctx.ASTRecordLayouts (and, since it's a POD aggregate here,
// nothing else) -- it does not intern a new Type, so Ctx.Types.size() is
// unchanged across this failing chunk and the whole restore() body,
// including ASTRecordLayouts's own (otherwise correct) erase-by-checkpoint
// logic, is skipped.
// =======================================================================
struct AlreadyKnown { int a; int b; };
unsigned long sz_known = sizeof(AlreadyKnown); intentional_error_type sizeof_garbage;

// Recomputing sizeof() on the same, still-live type must not dereference a
// stale ASTRecordLayout* left behind by the chunk above.
unsigned long sz_known2 = sizeof(AlreadyKnown);
auto t4 = printf("sizeof(AlreadyKnown) = %lu\n", sz_known2);
// CHECK: sizeof(AlreadyKnown) = 8

// =======================================================================
// Test 5: Virtual dispatch bookkeeping (Sema::VTableUses/VTablesUsed,
// ASTContext::KeyFunctions) is assert-only / guard-gated, same as above,
// but exercised through polymorphic classes and `new` instead of sizeof.
// RecordLayoutBuilder.cpp populates Ctx.KeyFunctions[RD] while laying out
// `Derived` for the `new` expression below; Sema::MarkVTableUsed populates
// VTableUses/VTablesUsed. Both are torn down on a *committed* chunk but
// only assert-compared on a rolled-back one.
// =======================================================================
struct Base { virtual int val() { return 1; } virtual ~Base() {} };
struct Derived : Base { int val() override { return 2; } };
Base* bp = new Derived(); int vv = bp->val(); intentional_error_type vtable_garbage;

// A fresh Base*/Derived pair exercising the same key-function/vtable-use
// bookkeeping. If the discarded attempt above left stale KeyFunctions /
// VTablesUsed state pointing at poisoned memory, codegen for *this* vtable
// can be skipped, corrupted, or crash outright.
Base* bp2 = new Derived();
auto t5 = printf("val = %d\n", bp2->val());
// CHECK: val = 2

// =======================================================================
// Test 6: Ctx.StringLiteralCache has no erase logic at all -- not gated by
// the Types.size() guard (defining a brand-new function interns a new
// FunctionProtoType, so the early return above does *not* fire here), just
// a bare `assert(CP.StringLiteralCacheSize == Ctx.StringLiteralCache.size())`
// with nothing to actually undo the insertion made by
// ASTContext::getPredefinedStringLiteralFromCache when Sema processes
// __PRETTY_FUNCTION__. In an assertions-enabled build this is the test most
// likely to abort immediately and deterministically, independent of memory
// poisoning, purely because the size check itself fails.
// =======================================================================
void uses_predefined_expr() { const char* pf = __PRETTY_FUNCTION__; (void)pf; } intentional_error_type predefined_garbage;

// A second, differently-named function that also uses __PRETTY_FUNCTION__,
// forcing another StringLiteralCache lookup/insert keyed differently from
// the discarded one above.
void uses_predefined_expr_again() { const char* pf = __PRETTY_FUNCTION__; (void)pf; }
int pf_ran = 1;
auto t6 = printf("pf_ran = %d\n", pf_ran);
// CHECK: pf_ran = 1

// =======================================================================
// Test 7: Cross-declaration merging state (Ctx.MergedDecls,
// Ctx.InstantiatedFromUsingShadowDecl) survives a failed using-declaration.
// A `using` declaration creates a UsingShadowDecl tied back to the
// original NS::helper via a side map that SemaStateStash/
// ASTContextStateStash only assert-compare. If the failed attempt's
// UsingShadowDecl (and its bookkeeping entry) is left dangling, a
// subsequent successful `using NS::helper;` shares the same DeclarationName
// and could resolve through, or conflict with, the stale shadow chain.
// =======================================================================
namespace NS { int helper() { return 10; } }
using NS::helper; intentional_error_type merge_garbage;

using NS::helper;
auto t7 = printf("helper() = %d\n", helper());
// CHECK: helper() = 10

%quit
